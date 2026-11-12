#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from tf2_ros import TransformBroadcaster

import yaml, os, time, math, threading
from ament_index_python.packages import get_package_share_directory
import numpy as np

from robo_project.common.map_handler import MapFrameManager
from robo_project.common.basic_types import PoseMeters, PosePixels

# ── Keep-inside-the-room guard ────────────────────────────────────────────────
# Uses the SAME grid, resolution and origin that publish_map() sends on /map
# (and that motion_planner.py plans on), so the guard and the planner always
# agree about what is free space.  Internal convention: 1 = free, 0 = wall.
MAP_RESOLUTION = 0.02       # must match the resolution in publish_map()
FORWARD_LOOKAHEAD_M = 0.10  # don't drive forward if the cell this far ahead is not free

# ── Path follower tuning ──────────────────────────────────────────────────────
WAYPOINT_TOLERANCE_M = 0.25  # intermediate waypoint counts as reached within this distance
WAYPOINT_PASSED_M = 0.40     # ...or within this distance once it is behind the robot
GOAL_TOLERANCE_M = 0.10      # final waypoint (the goal) tolerance
LIN_VEL = 0.3                # m/s when driving
MIN_LIN_VEL = 0.1            # m/s: slow down to this when approaching the goal
MAX_ANG_VEL = 0.8            # rad/s
ANG_GAIN = 1.5               # angular.z = ANG_GAIN * heading error (clamped)
# Turning on the spot (the robot stops driving) only for sharp turns: start above TURN_IN_PLACE_ERR, then keep
# turning until the heading error is below TURN_IN_PLACE_EXIT (hysteresis: no stop-go flicker). Smaller corners
# are taken while driving, slower the larger the heading error (full speed below SLOW_TURN_ERR).
# (was: on the spot above 20 deg -> a short stop at almost every path corner)
# 2026-11-12 (2): corners of 40-75 deg are also driven, as a tight slow curve (speed fraction down to
# MIN_TURN_SPEED_FRAC) - no stop - if the live grid shows room for it; else / above 75 deg: on the spot.
TURN_IN_PLACE_ERR = 1.31     # rad (~75 deg)
TURN_IN_PLACE_EXIT = 0.26    # rad (~15 deg)
SLOW_TURN_ERR = 0.17         # rad (~10 deg): full speed below this heading error
CURVE_CHECK_ERR = 0.70       # rad (~40 deg): above this the curve is only driven if the turn area is free
CURVE_SPEED_FRAC = 0.35      # speed fraction at CURVE_CHECK_ERR ...
MIN_TURN_SPEED_FRAC = 0.15   # ... and at TURN_IN_PLACE_ERR (0.045 m/s: a curve of ~6 cm radius)

# ── Room to turn on the spot ──────────────────────────────────────────────────
# Turning on the spot sweeps (part of) a circle of radius hypot(0.575, 0.30) = 0.65 m (1.15 x 0.60 m body,
# base_footprint at its centre). Before such a turn the runner computes the area the body really sweeps for
# that turn on the LIVE ML local grid /fusion/local_grid (robot frame: right even when the localized pose is
# off). Short way blocked -> the long way round if that is free. Both blocked (tight parking spot, dead end)
# -> manoeuvre: a small search over 0.5 s moves (forward / back / on the spot x left / straight / right) on the
# grid with the real body outline finds the shortest collision-free sequence that ends facing the next waypoint
# (a multi-point turn); it is re-planned every move with the newest grid.
# (without this every body overlap in the tests happened while turning on the spot in a tight spot)
BODY_HALF_L, BODY_HALF_W = 0.575, 0.30
PIVOT_RADIUS_M = math.hypot(BODY_HALF_L, BODY_HALF_W) + 0.08
LOCAL_GRID_MAX_AGE_S = 1.0
LOCAL_OCC_MIN = 50           # occupancy value counted as obstacle (model: >= 50% occupied)
BODY_MARGIN_M = 0.03         # safety margin around the body for these checks
PIVOT_BLOCK_CELLS = 3        # >= this many obstacle cells swept = blocked (ML noise tolerance)
MANOEUVRE_V, MANOEUVRE_W = 0.15, 0.6          # m/s, rad/s of the manoeuvre moves
MANOEUVRE_STEP_S = 0.5       # duration of one move
MANOEUVRE_MARGINS_M = (0.04, 0.0)   # body outline margins tried for the manoeuvre (the 4.7 cm cells add ~2 cm):
                                    # with 4 cm if possible, the bare outline only if there is no other way
MANOEUVRE_MAX_EXPAND = 3000  # search effort limit
MANOEUVRE_MAX_DIST_M = 1.0   # stay within this distance of where the manoeuvre started
RECOVERY_MAX_S = 30.0        # give up (turn on the spot anyway) after this long
FRONT_GUARD_M = 0.12         # obstacle this close in front of the bumper (full width) -> no forward driving:
FRONT_GUARD_REPORT_S = 3.0   # turn towards the waypoint; still blocked after this -> report it to the planner
FRONT_GUARD_CELLS = 3

# ── Stall detection ───────────────────────────────────────────────────────────
# The hand-drawn map is only approximate: Habitat can have an obstacle where
# the sketch shows free space.  While driving forward, compare how far the
# robot was commanded to go with how far it actually moved ALONG ITS HEADING
# (Habitat lets a blocked robot slide sideways along a wall, so plain movement
# is not enough).  If it is far less, report the spot ahead on /blocked_point;
# motion_planner marks it as an obstacle and re-plans from the current pose.
STALL_WINDOW_M = 0.40        # judge after this much commanded forward driving
STALL_MIN_RATIO = 0.30       # stalled if actual forward motion < 30% of commanded
BLOCKED_AHEAD_M = 0.25       # reported obstacle position: this far ahead of the robot


class RunnerNode(Node):
    def __init__(self):
        super().__init__('runner_node')

        print("\n" + "="*50, flush=True)
        print("🚀 RUNNER NODE INITIALIZING (NO ML / NO LOCALIZATION)...", flush=True)
        print("="*50 + "\n", flush=True)

        self.mfm = None

        # QoS perfectly matched to RViz settings
        map_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', map_qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_odom_msg_for_tf = None
        self.rviz_tf_timer = self.create_timer(0.05, self._publish_stable_rviz_tf_timer)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        # Publish the CURRENT target waypoint so a visualizer or other
        # tool can see exactly what runner_node is aiming for right now
        # (not just the full static path from motion_planner).
        self.current_waypoint_pub = self.create_publisher(
            PoseStamped, '/runner_current_waypoint', 10)
        self.blocked_pub = self.create_publisher(PointStamped, '/blocked_point', 10)
        self._reset_stall()

        self.cb_group = ReentrantCallbackGroup()

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.get_odom, 10, callback_group=self.cb_group)
        self.create_subscription(PoseStamped, '/goal_pose', self.rviz_goal_callback, 10, callback_group=self.cb_group)

        # ── Path following: receives the plan from motion_planner.py ───────────
        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/planned_path', self.path_callback, path_qos, callback_group=self.cb_group)

        # ── What the ML model has seen (fusion_obstacle_node): overrides the hand-drawn map in
        #    _is_free_world (the drawing can be ~1 m off from the real room) ────────────────
        self.ml_state = None   # H x W int8 in /map orientation: 100 seen occupied, 0 seen free, -1 not seen
        self.ml_info = None
        self.create_subscription(OccupancyGrid, '/fusion/obstacle_map', self._ml_map_callback, 2,
                                 callback_group=self.cb_group)
        self.local_grid = None   # (wall time, info, H x W int16) of /fusion/local_grid (robot frame)
        self.create_subscription(OccupancyGrid, '/fusion/local_grid', self._local_grid_callback, 2,
                                 callback_group=self.cb_group)
        self.path = []
        self.waypoint_idx = 0
        self._pivoting = False
        self._pivot_dir = None
        self._recovery_t0 = None
        self._move = None
        self._plan = []
        self._front_since = None
        self._loop_lock = threading.Lock()
        self.robot_yaw = 0.0
        self.path_goal_reached = False
        self.create_timer(0.1, self.path_follow_loop, callback_group=self.cb_group)

        self.setup_complete = False
        self.map_published_once = False

        self.startup_timer = self.create_timer(1.0, self.async_setup, callback_group=self.cb_group)
        self.create_timer(1.0, self.publish_map, callback_group=self.cb_group)

    def async_setup(self):
        if self.setup_complete: return
        self.startup_timer.cancel()

        print("⏳ Loading map in the background...", flush=True)
        try:
            self.mfm = MapFrameManager(use_discrete_state_space=True)
            self.setup_complete = True
            print("✅ BACKGROUND LOAD COMPLETE! Map is now actively broadcasting.", flush=True)
        except Exception as e:
            print(f"❌ ERROR DURING STARTUP: {e}", flush=True)

    def _publish_cmd_vel(self, fwd: float, ang: float):
        msg = Twist()
        msg.linear.x  = float(fwd)
        msg.angular.z = float(ang)
        self.cmd_vel_pub.publish(msg)

    def publish(self, twist_msg):
        is_turn = abs(twist_msg.angular.z) > 0.01
        is_move = abs(twist_msg.linear.x) > 0.01

        if is_turn:
            print("🤖 AI COMMAND: Perfect 90-degree turn.", flush=True)
            cmd = Twist()
            cmd.angular.z = math.copysign(0.5, twist_msg.angular.z)
            for _ in range(10):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        elif is_move:
            print("🤖 AI COMMAND: Perfect grid step forward.", flush=True)
            cmd = Twist()
            cmd.linear.x = math.copysign(0.5, twist_msg.linear.x)
            for _ in range(10):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.5)

    def publish_map(self):
        if not self.setup_complete or self.mfm is None:
            return

        grid_map = self.mfm.map
        if grid_map is None: return

        # map -> odom is published by _publish_stable_rviz_tf_timer (20 Hz);
        # not repeated here, so it has a single source.
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        resolution = 0.02
        msg.info.resolution = resolution
        msg.info.height = int(grid_map.shape[0])
        msg.info.width = int(grid_map.shape[1])
        msg.info.origin.position.x = float(-(grid_map.shape[1] / 2.0) * resolution)
        msg.info.origin.position.y = float(-(grid_map.shape[0] / 2.0) * resolution)
        msg.info.origin.orientation.w = 1.0

        # Standard ROS values: 0 = free, 100 = occupied.  OccupancyGrid row 0
        # is the BOTTOM (smallest y) while image row 0 is the top of the
        # sketch, so flip vertically - otherwise RViz shows the map mirrored.
        msg.data = np.where(np.flipud(grid_map) == 1, 0, 100).astype(np.int8).flatten().tolist()
        self.map_pub.publish(msg)

        if not self.map_published_once:
            print("🗺️ MAP BROADCAST SUCCESSFUL! Check RViz now.", flush=True)
            self.map_published_once = True

    def _nearest_free_pose_px(self, pose_px, max_radius: int = 120):
        """
        Snap any pixel pose to the nearest valid free cell on the hand-drawn map.
        Internal map convention: 1 = free, 0 = obstacle.
        """
        if pose_px is None:
            return None
        if self.mfm is None:
            return pose_px

        m = self.mfm.map_with_border
        H, W = m.shape

        r0 = int(round(pose_px.r))
        c0 = int(round(pose_px.c))
        yaw = float(getattr(pose_px, "yaw", 0.0))

        r0 = max(0, min(H - 1, r0))
        c0 = max(0, min(W - 1, c0))

        if m[r0, c0] == 1:
            return PosePixels(r0, c0, yaw)

        best = None
        best_d2 = None

        for rad in range(1, max_radius + 1):
            for dr in range(-rad, rad + 1):
                for dc in (-rad, rad):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < H and 0 <= c < W and m[r, c] == 1:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best_d2:
                            best = (r, c)
                            best_d2 = d2

            for dc in range(-rad + 1, rad):
                for dr in (-rad, rad):
                    r = r0 + dr
                    c = c0 + dc
                    if 0 <= r < H and 0 <= c < W and m[r, c] == 1:
                        d2 = dr * dr + dc * dc
                        if best is None or d2 < best_d2:
                            best = (r, c)
                            best_d2 = d2

            if best is not None:
                r, c = best
                return PosePixels(r, c, yaw)

        return PosePixels(r0, c0, yaw)

    def rviz_goal_callback(self, msg):
        self.goal_received = True
        # motion_planner subscribes to /goal_pose too: it re-plans from the
        # current pose and publishes the new /planned_path, which
        # path_callback() then follows.
        print(f"🎯 Goal received from RViz: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
              f"— waiting for motion_planner to re-plan.", flush=True)

    def _publish_stable_rviz_tf_timer(self):
        """
        Permanent RViz TF chain:
            map -> odom -> base_footprint

        This prevents RViz Fixed Frame 'map' and RobotModel from disappearing.
        """
        try:
            now = self.get_clock().now().to_msg()

            # map -> odom
            t_map_odom = TransformStamped()
            t_map_odom.header.stamp = now
            t_map_odom.header.frame_id = "map"
            t_map_odom.child_frame_id = "odom"
            t_map_odom.transform.translation.x = 0.0
            t_map_odom.transform.translation.y = 0.0
            t_map_odom.transform.translation.z = 0.0
            t_map_odom.transform.rotation.x = 0.0
            t_map_odom.transform.rotation.y = 0.0
            t_map_odom.transform.rotation.z = 0.0
            t_map_odom.transform.rotation.w = 1.0

            # NOTE: odom -> base_footprint is deliberately NOT published here.
            # habitat_bridge already publishes it (same data, from the simulator),
            # and two publishers of the same transform make RViz jitter.
            # runner_node owns ONLY map -> odom.
            self.tf_broadcaster.sendTransform([t_map_odom])

        except Exception as e:
            if not hasattr(self, "_stable_rviz_tf_warned"):
                print(f"⚠️ Stable RViz TF publish failed: {e}", flush=True)
                self._stable_rviz_tf_warned = True

    def get_odom(self, msg: Odometry):
        self.latest_odom_msg_for_tf = msg

        if not self.setup_complete or self.mfm is None:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        odom_pose = PoseMeters(x, y, yaw)
        self.latest_odom_yaw_for_astar = odom_pose.yaw
        self.robot_yaw = yaw

    # ── Path following ────────────────────────────────────────────────────────

    def path_callback(self, msg: Path):
        if not msg.poses:
            return
        new_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if new_path == self.path:
            return  # ignore republishes of the same path (don't reset progress)
        self.path = new_path
        self.waypoint_idx = 0
        self.path_goal_reached = False
        self._pivoting = False
        self._pivot_dir = None
        self._recovery_t0 = None
        self._move = None
        self._plan = []
        self._reset_stall()
        print(f"Path received from motion_planner: {len(self.path)} waypoints. Following now.", flush=True)

    def path_follow_loop(self):
        # one control step at a time (the timer is in a reentrant group; a manoeuvre search can take ~0.5 s)
        if not self._loop_lock.acquire(blocking=False):
            return
        try:
            self._path_follow_step()
        finally:
            self._loop_lock.release()

    def _path_follow_step(self):
        if not self.path or self.path_goal_reached:
            return
        odom_msg = self.latest_odom_msg_for_tf
        if odom_msg is None:
            return

        rx = float(odom_msg.pose.pose.position.x)
        ry = float(odom_msg.pose.pose.position.y)
        ryaw = self.robot_yaw

        # GUARD 1: robot is outside the free area of the sketch (Habitat's floor
        # can extend past a drawn wall).  Don't stop for good: the planned path
        # starts at the nearest free cell, so keep following it back inside;
        # only the look-ahead check (GUARD 2) is skipped while outside.
        outside = not self._is_free_world(rx, ry)
        if outside:
            self._log_throttled(
                "outside",
                f"⚠️ Robot at ({rx:.2f},{ry:.2f}) is outside the free area of the map — "
                f"driving back along the path.")

        # Skip every waypoint already within tolerance (the planner's path is
        # dense - one point per 2 cm cell - so many are passed per tick).
        while self.waypoint_idx < len(self.path):
            wx, wy = self.path[self.waypoint_idx]
            is_last = self.waypoint_idx == len(self.path) - 1
            d = math.hypot(wx - rx, wy - ry)
            if is_last:
                reached = d < GOAL_TOLERANCE_M
            else:
                # Also count a nearby waypoint that is already behind the robot
                # as passed; otherwise the robot circles around it.
                behind = abs(self._wrap_angle(math.atan2(wy - ry, wx - rx) - ryaw)) > math.pi / 2
                reached = d < WAYPOINT_TOLERANCE_M or (behind and d < WAYPOINT_PASSED_M)
            if not reached:
                break
            self.waypoint_idx += 1
            self._log_throttled("wp", f"Waypoint {self.waypoint_idx}/{len(self.path)} reached.")

        if self.waypoint_idx >= len(self.path):
            self._publish_cmd_vel(0.0, 0.0)
            self.path_goal_reached = True
            print(f"GOAL REACHED! Robot stopped at ({rx:.2f},{ry:.2f}).", flush=True)
            return

        wx, wy = self.path[self.waypoint_idx]

        # Publish current target so it can be visualized live
        wp_msg = PoseStamped()
        wp_msg.header.frame_id = 'map'
        wp_msg.header.stamp = self.get_clock().now().to_msg()
        wp_msg.pose.position.x = wx
        wp_msg.pose.position.y = wy
        wp_msg.pose.orientation.w = 1.0
        self.current_waypoint_pub.publish(wp_msg)

        dist = math.hypot(wx - rx, wy - ry)

        angle_to_wp = math.atan2(wy - ry, wx - rx)
        angle_err = self._wrap_angle(angle_to_wp - ryaw)

        # Debug: show exactly what the steering logic is computing,
        # so a direction bug is visible directly in the terminal.
        now = time.time()
        if not hasattr(self, "_last_steer_log") or now - self._last_steer_log > 1.0:
            print(
                f"[steer] robot=({rx:.2f},{ry:.2f}) yaw={math.degrees(ryaw):.0f}° "
                f"-> target=({wx:.2f},{wy:.2f}) dist={dist:.2f}m "
                f"angle_to_wp={math.degrees(angle_to_wp):.0f}° "
                f"angle_err={math.degrees(angle_err):.0f}°",
                flush=True)
            self._last_steer_log = now

        # Proportional steering (a fixed +-0.5 rad/s with a tight threshold
        # overshoots and wiggles left/right around the heading).
        ang = max(-MAX_ANG_VEL, min(MAX_ANG_VEL, ANG_GAIN * angle_err))
        if self._check_stall(rx, ry, ryaw):
            return

        sharp = abs(angle_err) > TURN_IN_PLACE_ERR or (
            abs(angle_err) > CURVE_CHECK_ERR and not self._pivoting
            and self._choose_turn(angle_err) != math.copysign(1.0, angle_err))   # no room for the curve
        if sharp:
            if not self._pivoting:
                self._pivoting = True
                self._pivot_dir = self._choose_turn(angle_err)
                if self._pivot_dir == 0:                 # both ways would hit something: recovery first
                    self._recovery_t0 = time.time()
                    print(f"↩️  No room to turn on the spot at ({rx:.2f},{ry:.2f}) - manoeuvring "
                          f"(multi-point turn).", flush=True)
        elif abs(angle_err) < TURN_IN_PLACE_EXIT:
            self._pivoting = False
            self._pivot_dir = None
            self._recovery_t0 = None
        if self._pivoting and self._recovery_t0 is not None:
            d = self._choose_turn(angle_err)
            if d == 0 and time.time() - self._recovery_t0 < RECOVERY_MAX_S:
                now = time.time()
                if self._move is None or now >= self._move[2]:
                    c, s_ = math.cos(ryaw), math.sin(ryaw)
                    wxr, wyr = c * (wx - rx) + s_ * (wy - ry), -s_ * (wx - rx) + c * (wy - ry)
                    if not self._plan or not self._move_free(*self._plan[0]):
                        self._publish_cmd_vel(0.0, 0.0)          # stand still while searching
                        self._plan = self._plan_manoeuvre(wxr, wyr) or []
                        if self._plan:
                            print(f"   manoeuvre: {len(self._plan)} moves "
                                  f"({' '.join(self._move_name(v, w) for v, w in self._plan)})", flush=True)
                    if self._plan:
                        v, w = self._plan.pop(0)
                        self._move = (v, w, now + MANOEUVRE_STEP_S)
                    else:
                        self._move = None
                if self._move is not None:
                    self._drive_prev = None
                    self._publish_cmd_vel(self._move[0], self._move[1])
                    return
            print(f"   manoeuvre done after {time.time() - self._recovery_t0:.1f} s - turning now.", flush=True)
            self._recovery_t0 = None
            self._move = None
            self._plan = []
            self._pivot_dir = d if d else self._choose_turn(angle_err, force=True)
        if self._pivoting and self._pivot_dir and abs(angle_err) > 0.6 and \
                math.copysign(1.0, angle_err) != self._pivot_dir:
            ang = self._pivot_dir * MAX_ANG_VEL           # the long way round (the short way is blocked)
        if self._pivoting:
            self._publish_cmd_vel(0.0, ang)
        else:
            # GUARD 3 (live ML grid, robot frame): obstacle right in front of the bumper -> no forward driving.
            # Turn towards the waypoint if that is free; blocked for FRONT_GUARD_REPORT_S -> report the obstacle
            # at its real position (from the grid) so the planner goes around it.
            hit = self._front_obstacle()
            if hit is not None:
                now = time.time()
                self._front_since = self._front_since or now
                turn = ang if abs(angle_err) > 0.05 and self._choose_turn(angle_err) == math.copysign(1.0, angle_err) \
                    else 0.0
                self._publish_cmd_vel(0.0, turn)
                self._log_throttled("front", f"🚧 Obstacle within {FRONT_GUARD_M:.2f} m in front of the bumper at "
                                             f"({rx:.2f},{ry:.2f}) - not driving forward.")
                if now - self._front_since > FRONT_GUARD_REPORT_S:
                    c, s_ = math.cos(ryaw), math.sin(ryaw)
                    ox_, oy_ = rx + c * hit[0] - s_ * hit[1], ry + s_ * hit[0] + c * hit[1]
                    self._front_since = None
                    self._report_blocked_at(ox_, oy_)
                return
            self._front_since = None
            # GUARD 2: the cell just ahead is wall/void -> don't step into it.
            ahead_x = rx + FORWARD_LOOKAHEAD_M * math.cos(ryaw)
            ahead_y = ry + FORWARD_LOOKAHEAD_M * math.sin(ryaw)
            if not outside and not self._is_free_world(ahead_x, ahead_y):
                self._publish_cmd_vel(0.0, 0.0)
                self._log_throttled(
                    "blocked",
                    f"🚧 Blocked: cell {FORWARD_LOOKAHEAD_M:.2f} m ahead of "
                    f"({rx:.2f},{ry:.2f}) is not free. Not driving forward.")
                self._report_blocked(rx, ry, ryaw)
                return
            err = abs(angle_err)
            if err <= SLOW_TURN_ERR:
                frac = 1.0
            elif err <= CURVE_CHECK_ERR:
                frac = 1.0 - (1.0 - CURVE_SPEED_FRAC) * (err - SLOW_TURN_ERR) / (CURVE_CHECK_ERR - SLOW_TURN_ERR)
            else:
                frac = CURVE_SPEED_FRAC - (CURVE_SPEED_FRAC - MIN_TURN_SPEED_FRAC) * \
                    min(1.0, (err - CURVE_CHECK_ERR) / (TURN_IN_PLACE_ERR - CURVE_CHECK_ERR))
            lin = LIN_VEL * frac
            if self.waypoint_idx == len(self.path) - 1:
                lin = max(MIN_LIN_VEL, min(lin, dist))  # slow down for the goal
            self._publish_cmd_vel(lin, ang)
            self._drive_prev = (time.time(), rx, ry, ryaw, lin)

    def _reset_stall(self):
        self._drive_prev = None   # (time, x, y, yaw, commanded speed) of the last forward command
        self._stall_expected = 0.0
        self._stall_actual = 0.0

    def _check_stall(self, rx: float, ry: float, ryaw: float) -> bool:
        """True (and obstacle reported) if commanded forward driving did not move
        the robot forward along its heading."""
        now = time.time()
        prev = self._drive_prev
        self._drive_prev = None
        if prev is not None:
            t0, x0, y0, yaw0, v = prev
            self._stall_expected += v * (now - t0)
            self._stall_actual += (rx - x0) * math.cos(yaw0) + (ry - y0) * math.sin(yaw0)
        if self._stall_expected < STALL_WINDOW_M:
            return False
        expected, actual = self._stall_expected, self._stall_actual
        self._reset_stall()
        if actual >= STALL_MIN_RATIO * expected:
            return False
        print(f"⛔ Stalled at ({rx:.2f},{ry:.2f}): drove {actual:.2f} m of {expected:.2f} m commanded — "
              f"something in Habitat blocks the way that is not on the map.", flush=True)
        self._report_blocked(rx, ry, ryaw)
        return True

    def _report_blocked_at(self, x: float, y: float):
        """Stop, publish an obstacle at world point (x, y) and wait for a new path."""
        self._publish_cmd_vel(0.0, 0.0)
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(x), float(y)
        self.blocked_pub.publish(msg)
        print(f"   Reported obstacle at ({x:.2f},{y:.2f}) — asking motion_planner to re-plan.", flush=True)
        self.path = []
        self._reset_stall()

    def _report_blocked(self, rx: float, ry: float, direction: float):
        """Stop, publish the blocked spot in `direction` from the robot and wait for a new path."""
        self._publish_cmd_vel(0.0, 0.0)
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = rx + BLOCKED_AHEAD_M * math.cos(direction)
        msg.point.y = ry + BLOCKED_AHEAD_M * math.sin(direction)
        self.blocked_pub.publish(msg)
        print(f"   Reported obstacle at ({msg.point.x:.2f},{msg.point.y:.2f}) — asking motion_planner to re-plan.",
              flush=True)
        # Pause following until the re-planned path arrives (path_callback).
        self.path = []
        self._reset_stall()

    def _local_grid_callback(self, msg: OccupancyGrid):
        if msg.header.frame_id not in ('base_footprint', 'base_link'):
            return
        a = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.local_grid = (time.time(), msg.info, a)

    def _front_obstacle(self):
        """Robot-frame centre (x, y) of the obstacle cells within FRONT_GUARD_M in front of the bumper (full width)
        on the live local grid, or None (no such obstacle / no fresh grid)."""
        g = self.local_grid
        if g is None or time.time() - g[0] > LOCAL_GRID_MAX_AGE_S:
            return None
        _, info, a = g
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        X, Y = np.meshgrid(ox + (np.arange(a.shape[1]) + 0.5) * res, oy + (np.arange(a.shape[0]) + 0.5) * res)
        sel = (a >= LOCAL_OCC_MIN) & (X >= BODY_HALF_L) & (X <= BODY_HALF_L + FRONT_GUARD_M) & \
            (np.abs(Y) <= BODY_HALF_W + 0.02)
        if int(sel.sum()) < FRONT_GUARD_CELLS:
            return None
        return float(X[sel].mean()), float(Y[sel].mean())

    def _grid_obstacles(self, radius):
        """Robot-frame (x, y) of the obstacle cells of the fresh live local grid within radius, minus the cells
        already under the body (+margin); None without a fresh grid."""
        g = self.local_grid
        if g is None or time.time() - g[0] > LOCAL_GRID_MAX_AGE_S:
            return None
        _, info, a = g
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        c0, c1 = max(0, int((-radius - ox) / res)), min(info.width - 1, int((radius - ox) / res))
        r0, r1 = max(0, int((-radius - oy) / res)), min(info.height - 1, int((radius - oy) / res))
        occ = a[r0:r1 + 1, c0:c1 + 1] >= LOCAL_OCC_MIN
        X, Y = np.meshgrid(ox + (np.arange(c0, c1 + 1) + 0.5) * res, oy + (np.arange(r0, r1 + 1) + 0.5) * res)
        keep = occ & ~self._in_body(X, Y) & (np.hypot(X, Y) <= radius)
        return X[keep], Y[keep]

    @staticmethod
    def _in_body(x, y):
        return (np.abs(x) <= BODY_HALF_L + BODY_MARGIN_M) & (np.abs(y) <= BODY_HALF_W + BODY_MARGIN_M)

    def _turn_hits(self, err):
        """Obstacle cells swept by the body when turning on the spot by err (short way) and the long way round;
        (None, None) without a fresh grid."""
        pts = self._grid_obstacles(PIVOT_RADIUS_M)
        if pts is None:
            return None, None
        px, py = pts
        if px.size == 0:
            return 0, 0
        out = []
        for turn in (err, err - math.copysign(2 * math.pi, err)):
            hit = np.zeros(px.size, bool)
            for th in np.linspace(0.0, turn, max(2, int(abs(turn) / math.radians(5)) + 1)):
                c, s_ = math.cos(th), math.sin(th)           # cell in the frame of the body turned by th
                hit |= self._in_body(c * px + s_ * py, -s_ * px + c * py)
            out.append(int(hit.sum()))
        return out[0], out[1]

    def _choose_turn(self, err, force=False):
        """Direction to turn on the spot: +1 left / -1 right; the short way if it is free, else the long way if
        that is free; 0 = both blocked, or with force the way with fewer obstacle cells."""
        short, long_ = self._turn_hits(err)
        sdir = math.copysign(1.0, err)
        if short is None or short < PIVOT_BLOCK_CELLS:
            return sdir
        if long_ < PIVOT_BLOCK_CELLS or (force and long_ < short):
            if not force:
                self._log_throttled("longway", "   short way round is blocked - turning the other way.")
            return -sdir
        return sdir if force else 0

    @staticmethod
    def _outline_checker(occ, info, margin=0.0):
        """collides(x, y, th): body outline (+margin) at robot-frame pose (x, y, th) touches >= 2 obstacle cells."""
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        H, W = occ.shape
        hl, hw = BODY_HALF_L + margin, BODY_HALF_W + margin
        sx = np.arange(-hl, hl + 1e-9, 0.04)
        sy = np.arange(-hw, hw + 1e-9, 0.04)
        bx = np.concatenate([sx, sx, np.full(sy.size, -hl), np.full(sy.size, hl)])
        by = np.concatenate([np.full(sx.size, -hw), np.full(sx.size, hw), sy, sy])

        def collides(x, y, th):
            c, s_ = math.cos(th), math.sin(th)
            gx = np.floor((x + c * bx - s_ * by - ox) / res).astype(int)
            gy = np.floor((y + s_ * bx + c * by - oy) / res).astype(int)
            ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            return int(occ[gy[ok], gx[ok]].sum()) >= 2
        return collides

    def _move_free(self, v, w):
        """True if the move (v, w) for MANOEUVRE_STEP_S is collision-free on the newest local grid."""
        g = self.local_grid
        if g is None or time.time() - g[0] > LOCAL_GRID_MAX_AGE_S:
            return False
        _, info, a = g
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        X, Y = np.meshgrid(ox + (np.arange(a.shape[1]) + 0.5) * res, oy + (np.arange(a.shape[0]) + 0.5) * res)
        collides = self._outline_checker((a >= LOCAL_OCC_MIN) & ~self._in_body(X, Y), info)
        x = y = th = 0.0
        for _ in range(4):
            dt = MANOEUVRE_STEP_S / 4
            x += v * dt * math.cos(th)
            y += v * dt * math.sin(th)
            th += w * dt
            if collides(x, y, th):
                return False
        return True

    @staticmethod
    def _move_name(v, w):
        return ("fwd" if v > 0 else "back" if v < 0 else "turn") + ("-L" if w > 0 else "-R" if w < 0 else "")

    def _plan_manoeuvre(self, wxr, wyr):
        """Shortest collision-free sequence of moves (v, w) after which the robot faces the waypoint (wxr, wyr)
        (robot frame), searched on the live local grid with the body outline; None if none is found."""
        import heapq
        g = self.local_grid
        if g is None or time.time() - g[0] > LOCAL_GRID_MAX_AGE_S:
            return None
        _, info, a = g
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        H, W = a.shape
        occ = a >= LOCAL_OCC_MIN
        X, Y = np.meshgrid(ox + (np.arange(W) + 0.5) * res, oy + (np.arange(H) + 0.5) * res)
        occ &= ~self._in_body(X, Y)                       # cells already under the body: ignored
        for margin in MANOEUVRE_MARGINS_M:
            plan = self._search_manoeuvre(self._outline_checker(occ, info, margin), wxr, wyr)
            if plan:
                return plan
        return None

    def _search_manoeuvre(self, collides, wxr, wyr):
        import heapq
        def facing(x, y, th):
            return abs(self._wrap_angle(math.atan2(wyr - y, wxr - x) - th)) < TURN_IN_PLACE_EXIT

        moves = [(v, w) for v in (MANOEUVRE_V, 0.0, -MANOEUVRE_V) for w in (MANOEUVRE_W, 0.0, -MANOEUVRE_W)
                 if (v, w) != (0.0, 0.0)]
        sub = 4
        dt = MANOEUVRE_STEP_S / sub
        key = lambda x, y, th: (round(x / 0.05), round(y / 0.05), round(self._wrap_angle(th) / math.radians(10)))
        h = lambda x, y, th: abs(self._wrap_angle(math.atan2(wyr - y, wxr - x) - th)) / MANOEUVRE_W
        start = (0.0, 0.0, 0.0)
        openq = [(h(*start), 0.0, 0, start, None)]
        nodes = {0: (start, None, None)}                  # id -> (state, parent id, move)
        seen = {key(*start)}
        nid, expanded = 0, 0
        while openq and expanded < MANOEUVRE_MAX_EXPAND:
            _, cost, cur, (x, y, th), _ = heapq.heappop(openq)
            if facing(x, y, th):
                plan = []
                while nodes[cur][1] is not None:
                    plan.append(nodes[cur][2])
                    cur = nodes[cur][1]
                return plan[::-1]
            expanded += 1
            for v, w in moves:
                nx, ny, nth = x, y, th
                bad = False
                for _ in range(sub):
                    nx += v * dt * math.cos(nth)
                    ny += v * dt * math.sin(nth)
                    nth += w * dt
                    if collides(nx, ny, nth):
                        bad = True
                        break
                if bad or math.hypot(nx, ny) > MANOEUVRE_MAX_DIST_M:
                    continue
                k = key(nx, ny, nth)
                if k in seen:
                    continue
                seen.add(k)
                prev_move = nodes[cur][2]
                step = MANOEUVRE_STEP_S * (1.5 if v < 0 else 1.0)
                if prev_move is not None and prev_move[0] * v < 0:
                    step += 0.5                            # changing between forward and backward
                nid += 1
                nodes[nid] = ((nx, ny, nth), cur, (v, w))
                heapq.heappush(openq, (cost + step + h(nx, ny, nth), cost + step, nid, (nx, ny, nth), None))
        return None

    # ── Keep-inside-the-room guard ───────────────────────────────────────────

    def _ml_map_callback(self, msg: OccupancyGrid):
        self.ml_info = msg.info
        self.ml_state = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(msg.info.height, msg.info.width)

    def _is_free_world(self, x: float, y: float) -> bool:
        """True if world point (x, y) is a free cell.
        Where the ML model has seen the cell (/fusion/obstacle_map), its view decides (seen free /
        seen occupied); elsewhere the hand-drawn map decides.
        Out-of-bounds counts as NOT free (that is the black void outside the room).
        Uses the same origin/resolution as publish_map(), so it matches /map."""
        ml, info = self.ml_state, self.ml_info
        if ml is not None:
            mc = math.floor((x - info.origin.position.x) / info.resolution)
            mr = math.floor((y - info.origin.position.y) / info.resolution)
            if 0 <= mr < info.height and 0 <= mc < info.width and ml[mr, mc] != -1:
                return bool(ml[mr, mc] == 0)
        if self.mfm is None or self.mfm.map is None:
            return True  # map not loaded yet: don't block anything
        grid = self.mfm.map
        H, W = grid.shape
        col = math.floor(x / MAP_RESOLUTION + W / 2.0)
        row = H - 1 - math.floor(y / MAP_RESOLUTION + H / 2.0)  # image rows are flipped vs /map
        if not (0 <= row < H and 0 <= col < W):
            return False
        return grid[row, col] == 1

    def _log_throttled(self, key: str, text: str, period: float = 1.0):
        now = time.time()
        last = getattr(self, "_last_log_" + key, 0.0)
        if now - last > period:
            print(text, flush=True)
            setattr(self, "_last_log_" + key, now)

    @staticmethod
    def _wrap_angle(a):
        while a > math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


def main(args=None):
    rclpy.init(args=args)
    node = RunnerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

