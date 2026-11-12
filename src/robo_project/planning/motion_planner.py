#!/usr/bin/env python3

import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry

import time

from robo_project.common.basic_types import PosePixels
# Compiled A* + path smoothing: same paths as planning/astar.py, ~30 ms instead of up to ~9 s.
from robo_project.planning.fast_astar import run_astar, run_astar_cost, smooth_path

# ── GOAL in world coordinates (meters) ───────────────────────────────────────
GOAL_X = 0.0   # meters  (default goal; a "2D Goal Pose" clicked in RViz replaces it)
GOAL_Y = 0.0   # meters

# Walls are grown by this much before planning, so the path keeps the robot
# body away from them instead of hugging the wall line.
# Real robot: 1.15 m x 0.60 m (config/vinebot_rig.yaml) -> half-width 0.30 m + 0.05 m margin.
INFLATION_M = 0.35

# Obstacles reported by runner_node on /blocked_point (things that exist in
# Habitat but not on the hand-drawn map) are added as discs of this radius.
BLOCKED_RADIUS_M = 0.25
BLOCKED_EXPIRE_S = 60.0     # reported obstacles are forgotten after this (a wrong report must not close a passage)

# Obstacles seen by the fusion ML model (fusion_obstacle_node, /fusion/obstacle_map, aligned with
# /map) are added to the A* map like walls.  When one lies on the remaining path, re-plan at once.
# Path points closer than ML_CHECK_MIN_DIST_M to the robot are not checked (the robot may already
# be inside a wall's inflation; the new path starts from the nearest free cell anyway).
ML_CHECK_MIN_DIST_M = INFLATION_M + 0.05
REPLAN_MIN_PERIOD_S = 0.5   # at most one ML-triggered re-plan per this period
NO_PATH_RETRY_S = 2.0       # no path: stop the robot and try again after this

# Robot inside a wall's inflation band (Habitat's real room is wider than the sketch, so the robot
# can end up a few cm from a drawn wall): A* starts from the nearest free cell, and runner_node
# would cut straight from the robot to the next waypoint - through the drawn wall, which its
# look-ahead refuses ("Blocked: cell ahead is not free") -> endless re-plan loop. So the first
# waypoint is then a point of the A* path that the robot can reach in a straight line over free
# sketch cells, and far enough that runner_node does not skip it (it skips waypoints < 0.25 m,
# or < 0.40 m when behind it).
ESCAPE_MIN_DIST_M = 0.45
ESCAPE_MAX_DIST_M = 1.5
# The same spot reported blocked this often without the robot moving: stop and retry later.
STUCK_REPEATS = 3

# The hand-drawn map can be off by ~1 m from the real room. What the ML model has SEEN overrides
# the drawing: cells it saw free are free even where a wall is drawn (and cells it saw occupied
# are obstacles). If the drawing still gives no route, A* is run again where drawn walls the model
# has not seen yet - within SOFT_WALL_BAND_M of the drawn room - are passable at SOFT_WALL_COST per
# cell: the robot tries the most likely gap, the model shows on the way whether the wall is real,
# and the planner re-plans. Walls further out (the void outside the drawing) stay blocked.
SOFT_WALL_BAND_M = 1.0
# Goal clearance: the robot is 1.15 x 0.60 m, so its corners are 0.65 m from its centre. A goal closer
# than that to an obstacle makes the body hit it when the robot turns / parks there (seen in the tests at
# goals 0.5-0.6 m from a wall). Such goals are moved up to GOAL_SHIFT_MAX_M to the nearest spot with
# GOAL_CLEARANCE_M of room (or the roomiest spot nearby).
# Turning room: the 1.15 x 0.60 m body sweeps a 0.65 m circle when it turns on the spot, but paths only keep
# INFLATION_M (0.35) from walls so that the robot fits through corridors. Cells closer than PROX_RANGE_M to an
# obstacle cost up to PROX_COST extra (paths keep to the middle when there is room), and the path smoothing
# keeps its corners SMOOTH_CLEARANCE_M from walls where possible (corners are where the robot turns).
PROX_RANGE_M = 0.65
PROX_COST = 3.0
SMOOTH_CLEARANCE_M = 0.55
GOAL_CLEARANCE_M = 0.80     # corner radius 0.65 + 0.15 m (ML map near walls misses ~4% of obstacle cells)
GOAL_SHIFT_MAX_M = 0.50
GOAL_SHIFT_FAR_M = 0.50     # = GOAL_SHIFT_MAX_M: searching further (1.5 m tested) picked spots that only the SKETCH showed
#                             as roomy (unseen by the ML) - in the tests that was worse, so it is off
SOFT_WALL_COST = 20.0


class MotionPlanner(Node):

    def __init__(self):
        super().__init__('motion_planner')
        # room needed at a parking spot (m); larger with localization (estimated pose, sketch ~0.7 m off)
        self.goal_clearance = float(self.declare_parameter('goal_clearance', GOAL_CLEARANCE_M).value)
        self.wait_for_ml_s = float(self.declare_parameter('wait_for_ml_s', 8.0).value)
        self.start_time = time.monotonic()

        path_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.path_pub = self.create_publisher(Path, '/planned_path', path_qos)

        # Publish the goal itself, so any other node (e.g. a visualizer)
        # can read the ACTUAL goal being used instead of hardcoding a
        # separate copy of GOAL_X/GOAL_Y that could drift out of sync.
        self.goal_pub = self.create_publisher(PoseStamped, '/motion_planner_goal', path_qos)

        self.grid = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None
        self.map_height = None
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        self.robot_x = None
        self.robot_y = None
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.goal_x = GOAL_X
        self.goal_y = GOAL_Y
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)

        self.blocked_points = []
        self.create_subscription(PointStamped, '/blocked_point', self.blocked_callback, 10)

        self.ml_occ = None          # bool H x W: obstacles seen by the ML model
        self.ml_free = None         # bool H x W: free space seen by the ML model
        self.ml_inflated = None     # ML obstacles grown by INFLATION_M (what the path must avoid)
        self.far_wall = None        # drawn walls further than SOFT_WALL_BAND_M from the drawn room
        self.ml_warned = False
        self.create_subscription(OccupancyGrid, '/fusion/obstacle_map', self.ml_map_callback, 2)
        self.current_wp = None      # runner_node's current waypoint (x, y)
        self.create_subscription(PoseStamped, '/runner_current_waypoint', self.current_wp_callback, 5)
        self.last_replan = 0.0
        self.no_path_since = None
        self.last_blocked = None    # (point x, y, robot x, y) of the last /blocked_point
        self.blocked_repeats = 0

        self.planned = False
        self.last_path_msg = None
        self.goal_msg = self._build_goal_msg()
        self.goal_pub.publish(self.goal_msg)
        self.create_timer(0.2, self.try_plan)
        self.create_timer(2.0, self.republish_path)

        self.get_logger().info("motion_planner started. Waiting for /map and /odom...")
        self.get_logger().info(f"Goal: x={GOAL_X}m, y={GOAL_Y}m")

    def _build_goal_msg(self):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.goal_x
        msg.pose.position.y = self.goal_y
        msg.pose.orientation.w = 1.0
        return msg

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def map_callback(self, msg: OccupancyGrid):
        if self.grid is not None:
            return  # only need to load the map once
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.grid = np.array(msg.data, dtype=np.int8).reshape(
            self.map_height, self.map_width)
        walls = (self.grid != 0).astype(np.uint8)
        dist_to_room = cv2.distanceTransform(walls, cv2.DIST_L2, 5) * self.map_resolution
        self.far_wall = dist_to_room > SOFT_WALL_BAND_M
        self.get_logger().info(
            f"Map received: {self.map_width}x{self.map_height} "
            f"res={self.map_resolution}m origin=({self.map_origin_x:.2f},"
            f"{self.map_origin_y:.2f})")

    def odom_callback(self, msg: Odometry):
        self.robot_x = float(msg.pose.pose.position.x)
        self.robot_y = float(msg.pose.pose.position.y)

    def goal_callback(self, msg: PoseStamped):
        self.goal_x = float(msg.pose.position.x)
        self.goal_y = float(msg.pose.position.y)
        self.get_logger().info(f"New goal from RViz: x={self.goal_x:.2f}m, y={self.goal_y:.2f}m. Re-planning.")
        self.goal_msg = self._build_goal_msg()
        self.goal_pub.publish(self.goal_msg)
        self.last_path_msg = None  # don't republish the old path meanwhile
        self.planned = False  # try_plan() runs A* again from the current pose

    def blocked_callback(self, msg: PointStamped):
        px, py = float(msg.point.x), float(msg.point.y)
        rx = self.robot_x if self.robot_x is not None else 0.0
        ry = self.robot_y if self.robot_y is not None else 0.0
        lb = self.last_blocked
        same = lb is not None and math.hypot(px - lb[0], py - lb[1]) < 0.05 \
            and math.hypot(rx - lb[2], ry - lb[3]) < 0.05
        self.blocked_repeats = self.blocked_repeats + 1 if same else 0
        self.last_blocked = (px, py, rx, ry)
        if self.blocked_repeats >= STUCK_REPEATS - 1:
            # Re-planning gives the same path again: don't spin in a loop at 5 Hz.
            self.get_logger().error(
                f"Robot stuck at ({rx:.2f},{ry:.2f}): ({px:.2f},{py:.2f}) reported blocked "
                f"{self.blocked_repeats + 1} times without moving. Stopping, retrying in {NO_PATH_RETRY_S:.0f} s.")
            self.no_path_since = time.monotonic()
            self.planned = True
            self._publish_stop()
            return
        self.blocked_points.append((px, py, time.monotonic()))
        self.get_logger().warn(
            f"Obstacle reported at ({msg.point.x:.2f},{msg.point.y:.2f}) "
            f"[{len(self.blocked_points)} total]. Re-planning from the current pose.")
        self.last_path_msg = None
        self.planned = False

    def current_wp_callback(self, msg: PoseStamped):
        self.current_wp = (float(msg.pose.position.x), float(msg.pose.position.y))

    def ml_map_callback(self, msg: OccupancyGrid):
        if self.grid is None:
            return
        if msg.info.width != self.map_width or msg.info.height != self.map_height or \
                abs(msg.info.resolution - self.map_resolution) > 1e-6:
            if not self.ml_warned:
                self.get_logger().error("/fusion/obstacle_map does not match /map - ignored.")
                self.ml_warned = True
            return
        state = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(self.map_height, self.map_width)
        self.ml_occ = state == 100
        self.ml_free = state == 0
        r_px = max(1, int(round(INFLATION_M / self.map_resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
        self.ml_inflated = cv2.dilate(self.ml_occ.astype(np.uint8), kernel) > 0
        if self.planned and self.last_path_msg is not None and self.no_path_since is None \
                and time.monotonic() - self.last_replan > REPLAN_MIN_PERIOD_S:
            hit = self._path_blocked_by_ml()
            if hit is not None:
                self.get_logger().warn(
                    f"ML model: obstacle on the path at ({hit[0]:.2f},{hit[1]:.2f}). Re-planning.")
                self.planned = False

    def _path_blocked_by_ml(self):
        """First point of the remaining path inside an (inflated) ML obstacle, or None."""
        pts = [(p.pose.position.x, p.pose.position.y) for p in self.last_path_msg.poses]
        if not pts or self.robot_x is None:
            return None
        # Remaining path: robot -> runner's current waypoint -> rest.  Without that topic, from the
        # path vertex nearest to the robot.
        k = None
        if self.current_wp is not None:
            d = [math.hypot(x - self.current_wp[0], y - self.current_wp[1]) for x, y in pts]
            if min(d) < 0.05:
                k = int(np.argmin(d))
        if k is None:
            k = int(np.argmin([math.hypot(x - self.robot_x, y - self.robot_y) for x, y in pts]))
        line = [(self.robot_x, self.robot_y)] + pts[k:]
        step = self.map_resolution / 2
        for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for i in range(n + 1):
                x = x0 + (x1 - x0) * i / n
                y = y0 + (y1 - y0) * i / n
                if math.hypot(x - self.robot_x, y - self.robot_y) < ML_CHECK_MIN_DIST_M:
                    continue
                r, c = self.world_to_pixel(x, y)
                if self.ml_inflated[r, c]:
                    return x, y
        return None

    # ── Coordinate conversion ────────────────────────────────────────────────
    # Uses the resolution/origin actually published in /map — never
    # hardcoded separately, so it can't drift out of sync.

    def world_to_pixel(self, x: float, y: float):
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)
        row = max(0, min(self.map_height - 1, row))
        col = max(0, min(self.map_width - 1, col))
        return row, col

    def pixel_to_world(self, row: int, col: int):
        x = self.map_origin_x + col * self.map_resolution
        y = self.map_origin_y + row * self.map_resolution
        return x, y

    # ── Planning ──────────────────────────────────────────────────────────────

    def try_plan(self):
        if self.planned and self.no_path_since is not None \
                and time.monotonic() - self.no_path_since > NO_PATH_RETRY_S:
            self.planned = False   # retry: the ML obstacle map may have changed
        if self.planned:
            return
        if self.grid is None or self.robot_x is None:
            return
        # Do not plan blind: if the ML node is running, wait (max wait_for_ml_s) for its first obstacle map.
        # A first path drawn on the sketch alone can lead straight into an obstacle the model sees a moment
        # later, and the re-plan then turns the long robot on the spot next to it.
        if self.ml_occ is None and time.monotonic() - self.start_time < self.wait_for_ml_s:
            if self.count_publishers('/fusion/obstacle_map') > 0:
                return
        self.planned = True
        self.run_astar_and_publish()

    def republish_path(self):
        self.goal_msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(self.goal_msg)
        if self.last_path_msg is None:
            return
        self.last_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.last_path_msg)

    def run_astar_and_publish(self):
        start_row, start_col = self.world_to_pixel(self.robot_x, self.robot_y)
        goal_row, goal_col = self.world_to_pixel(self.goal_x, self.goal_y)

        self.get_logger().info(
            f"Start: world=({self.robot_x:.2f},{self.robot_y:.2f}) "
            f"-> pixel=({start_row},{start_col}) "
            f"value={self.grid[start_row, start_col]}")
        self.get_logger().info(
            f"Goal:  world=({self.goal_x:.2f},{self.goal_y:.2f}) "
            f"-> pixel=({goal_row},{goal_col}) "
            f"value={self.grid[goal_row, goal_col]}")

        # /map uses the ROS convention (0 = free, 100 = wall, -1 = unknown);
        # A* wants 1 = free, 0 = wall.  Grow the walls by INFLATION_M.
        free = (self.grid == 0).astype(np.uint8)
        n_ml = 0
        if self.ml_occ is not None:
            free[self.ml_free] = 1         # free space seen by the ML model overrides drawn walls
            free[self.ml_occ] = 0          # obstacles seen by the ML model
            n_ml = int(self.ml_occ.sum())
        disc = np.zeros_like(free)         # obstacles reported by runner_node (robot got stuck)
        b_px = int(round(BLOCKED_RADIUS_M / self.map_resolution))
        self.blocked_points = [b for b in self.blocked_points if time.monotonic() - b[2] < BLOCKED_EXPIRE_S]
        for bx, by, _ in self.blocked_points:
            b_row, b_col = self.world_to_pixel(bx, by)
            cv2.circle(disc, (b_col, b_row), b_px, 1, -1)
        free[disc > 0] = 0
        r_px = max(1, int(round(INFLATION_M / self.map_resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
        astar_map = cv2.erode(free, kernel)

        t0 = time.monotonic()
        self.last_replan = t0
        # One search on a cost map: free cells cost 1, drawn walls the ML model has not seen yet
        # (near the drawn room) cost SOFT_WALL_COST per cell - so A* crosses a drawn wall only when
        # that saves a long detour (the drawing can be ~1 m off), see SOFT_WALL_BAND_M.
        cost = self._soft_cost_map(free, disc, kernel)
        passable = np.isfinite(cost).astype(np.uint8)
        s_row, s_col = self._snap_to_free(passable, start_row, start_col)
        g_row, g_col = self._snap_to_free(passable, goal_row, goal_col)
        g_row, g_col = self._safe_goal(free, passable, g_row, g_col)
        self._publish_effective_goal(g_row, g_col)
        path = run_astar_cost(cost, PosePixels(s_row, s_col), PosePixels(g_row, g_col))
        nav_map = astar_map                # map the path is smoothed / checked on
        if path:
            soft = np.zeros_like(astar_map)
            for p in path:
                if cost[int(p.r), int(p.c)] > 1.0:
                    soft[int(p.r), int(p.c)] = 1
            n_soft = int(soft.sum())
            if n_soft:
                # smoothing may only use the unseen-wall cells A* chose (not cut more of them)
                nav_map = (astar_map | cv2.dilate(soft, np.ones((5, 5), np.uint8))).astype(np.uint8)
                self.get_logger().warn(
                    f"Path crosses {n_soft * self.map_resolution:.2f} m of drawn wall the ML model has not "
                    f"seen yet (the drawing can be off). The model will show on the way whether it is "
                    f"real; the planner re-plans if so.")
        else:
            # without numba there is no cost-map search: plain A* on the drawn map + model view
            # (also used when the parking spot is not reachable on the cost map: plain A* to the nearest free
            #  cell of the inflated map, starting from the chosen spot - and announce that goal)
            s_row, s_col = self._snap_to_free(astar_map, start_row, start_col)
            sg_row, sg_col = g_row, g_col
            g_row, g_col = self._snap_to_free(astar_map, sg_row, sg_col)
            path = run_astar(astar_map, PosePixels(s_row, s_col), PosePixels(g_row, g_col), include_diagonals=True)
            if path and (g_row, g_col) != (sg_row, sg_col):
                ox, oy = self.pixel_to_world(sg_row, sg_col)
                nx, ny = self.pixel_to_world(g_row, g_col)
                self.get_logger().warn(
                    f"Parking spot ({ox:.2f},{oy:.2f}) is not reachable with the robot's clearance; driving to the "
                    f"nearest reachable spot ({nx:.2f},{ny:.2f}) instead ({math.hypot(nx - ox, ny - oy):.2f} m away).")
                self._publish_effective_goal(g_row, g_col)

        if not path:
            self.get_logger().error(
                "A* found NO PATH between start and goal (they are not connected on the inflated map, "
                f"{n_ml} ML obstacle cells). Stopping the robot, retrying in {NO_PATH_RETRY_S:.0f} s.")
            self.no_path_since = time.monotonic()
            self._publish_stop()
            return
        self.no_path_since = None

        # Robot inside a wall's inflation band: first waypoint = a point of the path it can
        # reach in a straight line over free cells (see ESCAPE_MIN_DIST_M).
        robot_row, robot_col = self.world_to_pixel(self.robot_x, self.robot_y)
        if nav_map[robot_row, robot_col] == 0:
            fwd = path[::-1]                      # start -> goal
            k = self._escape_index(fwd, robot_row, robot_col, free | (nav_map & (1 - astar_map)))
            if k:
                ex, ey = self.pixel_to_world(int(fwd[k].r), int(fwd[k].c))
                self.get_logger().warn(
                    f"Robot at ({self.robot_x:.2f},{self.robot_y:.2f}) is within {INFLATION_M:.2f} m of a "
                    f"wall: leaving straight to ({ex:.2f},{ey:.2f}) first.")
                path = fwd[k:][::-1]

        # run_astar returns the path REVERSED (goal->start), one cell per
        # step.  smooth_path() keeps only the corners that are needed
        # (line-of-sight on the inflated map), so the robot drives straight
        # segments instead of a 2 cm staircase.
        # shortcuts keep SMOOTH_CLEARANCE_M from walls where possible (the A* cells themselves stay allowed)
        rs = max(1, int(round(SMOOTH_CLEARANCE_M / self.map_resolution)))
        wide = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1)))
        smooth_map = (wide & nav_map).astype(np.uint8)
        for p in path:
            smooth_map[int(p.r), int(p.c)] = 1
        smoothed = smooth_path(smooth_map, path)
        waypoints_px = list(reversed(smoothed))  # start -> goal order

        self.get_logger().info(
            f"Path found: {len(path)} cells, smoothed to {len(waypoints_px)} waypoints "
            f"in {(time.monotonic() - t0) * 1000:.0f} ms ({n_ml} ML obstacle cells on the map)")

        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        for wp in waypoints_px:
            x, y = self.pixel_to_world(int(wp.r), int(wp.c))
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.last_path_msg = msg
        self.path_pub.publish(msg)
        self.get_logger().info(
            f"Path published on /planned_path with {len(msg.poses)} waypoints.")

    def _soft_cost_map(self, free, disc, kernel):
        """Cost per cell: 1 = free (drawing or seen by the model), SOFT_WALL_COST = drawn wall the
        model has not seen yet near the drawn room (and its inflation band), inf = blocked (seen by
        the model, reported by runner_node, or far outside the drawn room). Walls are grown by
        INFLATION_M as in the normal map."""
        hard = (disc > 0) | (self.far_wall & (free == 0))
        if self.ml_occ is not None:
            hard |= self.ml_occ
        hard_inf = cv2.dilate(hard.astype(np.uint8), kernel) > 0
        any_inf = cv2.erode(free, kernel) == 0
        cost = np.ones(free.shape, np.float64)
        cost[any_inf] = SOFT_WALL_COST
        # keep away from obstacles when there is room (turning on the spot needs PROX_RANGE_M)
        obst = ((free == 0) | hard).astype(np.uint8)
        d = cv2.distanceTransform(1 - obst, cv2.DIST_L2, 5) * self.map_resolution
        near = (d < PROX_RANGE_M) & ~any_inf
        frac = np.clip((PROX_RANGE_M - d[near]) / (PROX_RANGE_M - INFLATION_M), 0.0, 1.0)
        cost[near] += PROX_COST * frac ** 2
        cost[hard_inf] = np.inf
        return cost

    def _escape_index(self, fwd, r0, c0, free):
        """Index of the first path cell ESCAPE_MIN..MAX_DIST_M from the robot that the straight
        line from the robot reaches over free cells only (sketch + reported + ML obstacles,
        not inflated - what runner_node's look-ahead checks). None if there is none."""
        res = self.map_resolution
        for i, p in enumerate(fwd):
            d = math.hypot(p.r - r0, p.c - c0) * res
            if d > ESCAPE_MAX_DIST_M:
                break
            if d >= ESCAPE_MIN_DIST_M and self._line_free(free, r0, c0, int(p.r), int(p.c)):
                return i
        return None

    @staticmethod
    def _line_free(free, r0, c0, r1, c1):
        n = int(max(abs(r1 - r0), abs(c1 - c0)) * 4) + 1
        rows = np.round(np.linspace(r0, r1, n + 1)).astype(int)
        cols = np.round(np.linspace(c0, c1, n + 1)).astype(int)
        H, W = free.shape
        if rows.min() < 0 or cols.min() < 0 or rows.max() >= H or cols.max() >= W:
            return False
        return bool(free[rows, cols].all())

    def _publish_stop(self):
        """A one-waypoint path at the robot's position: runner_node stops there instead of
        driving on along the old (now blocked) path."""
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(self.robot_x)
        ps.pose.position.y = float(self.robot_y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
        self.last_path_msg = msg
        self.path_pub.publish(msg)

    def _publish_effective_goal(self, r, c):
        """/motion_planner_goal = where the robot will actually stop: the clicked goal, moved out of an
        obstacle (snap) and away from walls (_safe_goal) if needed. RViz shows it as "Goal"."""
        x, y = self.pixel_to_world(r, c)
        if math.hypot(x - self.goal_x, y - self.goal_y) < 0.03:      # unchanged (within a map cell)
            x, y = self.goal_x, self.goal_y
        self.goal_msg.pose.position.x, self.goal_msg.pose.position.y = x, y
        self.goal_msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(self.goal_msg)

    def _safe_goal(self, free, passable, r, c):
        """Parking cell for the clicked goal. The spot chosen earlier for the same goal is KEPT while it is
        passable and still has (nearly) as much room as the best spot now - re-plans must not move the target
        the runner is driving to (with localization the ML obstacle map shifts slightly) - but it is replaced
        when the ML view shows that it is clearly more cramped than another spot (5 cm hysteresis)."""
        key = (round(self.goal_x, 3), round(self.goal_y, 3))
        cached = getattr(self, "_goal_cache", None)
        prefer = cached[1] if cached and cached[0] == key else None
        nr, nc = self._safe_goal_compute(free, passable, r, c, prefer)
        self._goal_cache = (key, (nr, nc))
        return nr, nc

    def _safe_goal_compute(self, free, passable, r, c, prefer=None):
        """Move the goal cell (r, c) to where the whole robot body fits (goal_clearance, default GOAL_CLEARANCE_M, from every
        obstacle), at most GOAL_SHIFT_MAX_M away; else to the roomiest reachable cell in that radius.
        prefer: earlier choice, kept if it is passable and has the full self.goal_clearance (or, when no spot
        has it, at least the best room - 0.05 m)."""
        res = self.map_resolution
        R = int(round(GOAL_SHIFT_FAR_M / res))
        H, W = free.shape
        r0, r1, c0, c1 = max(0, r - R), min(H, r + R + 1), max(0, c - R), min(W, c + R + 1)
        # clearance of each cell, computed on a padded window so obstacles just outside count too
        pad = int(round(self.goal_clearance / res)) + 1
        pr0, pr1, pc0, pc1 = max(0, r0 - pad), min(H, r1 + pad), max(0, c0 - pad), min(W, c1 + pad)
        clear = cv2.distanceTransform(free[pr0:pr1, pc0:pc1], cv2.DIST_L2, 5) * res
        clear = clear[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0]
        here = float(clear[r - r0, c - c0])
        rr, cc = np.mgrid[r0:r1, c0:c1]
        dist = np.hypot(rr - r, cc - c) * res
        reach = passable[r0:r1, c0:c1] > 0
        ok = (dist <= GOAL_SHIFT_MAX_M) & reach
        # best spot now: full room nearby; else full room further away (outside a narrow alcove - the body
        # cannot fit in it without touching); else the roomiest spot nearby
        if here >= self.goal_clearance:
            nr, nc, room, moved = r, c, here, 0.0
        else:
            good = ok & (clear >= self.goal_clearance)
            good_far = reach & (dist <= GOAL_SHIFT_FAR_M) & (clear >= self.goal_clearance)
            if good.any():
                k = np.argmin(np.where(good, dist, np.inf))
            elif good_far.any():
                k = np.argmin(np.where(good_far, dist, np.inf))
            elif ok.any():
                k = np.argmax(np.where(ok, clear - 0.1 * dist, -np.inf))
            else:
                k = None
            if k is None or float(clear.flat[k]) <= here + 0.02:
                nr, nc, room, moved = r, c, here, 0.0
            else:
                nr, nc, room, moved = int(rr.flat[k]), int(cc.flat[k]), float(clear.flat[k]), float(dist.flat[k])
        # keep the earlier choice unless it is clearly worse
        if prefer is not None and r0 <= prefer[0] < r1 and c0 <= prefer[1] < c1:
            pr, pc = prefer
            if reach[pr - r0, pc - c0] or (pr, pc) == (r, c):
                # full room if any spot has it; hysteresis (5 cm) only when no spot reaches self.goal_clearance
                need = self.goal_clearance if room >= self.goal_clearance else room - 0.05
                if float(clear[pr - r0, pc - c0]) >= need:
                    return pr, pc
        if (nr, nc) == (r, c):
            return r, c
        gx, gy = self.pixel_to_world(r, c)
        nx, ny = self.pixel_to_world(nr, nc)
        if getattr(self, "_goal_shift_logged", None) != (r, c, nr, nc):
            self._goal_shift_logged = (r, c, nr, nc)
            self.get_logger().warn(
                f"Goal ({gx:.2f},{gy:.2f}) is only {here:.2f} m from an obstacle - the 1.15 x 0.60 m robot needs "
                f"{self.goal_clearance:.2f} m to park and turn there. Driving to ({nx:.2f},{ny:.2f}) instead "
                f"({moved:.2f} m away, {room:.2f} m of room).")
        return nr, nc

    def _snap_to_free(self, astar_map, r, c, max_radius=100):
        """Snap to nearest cell where value != 0 (free, per this map's convention)."""
        H, W = astar_map.shape
        r = max(0, min(H - 1, r))
        c = max(0, min(W - 1, c))
        if astar_map[r, c] != 0:
            return r, c
        for radius in range(1, max_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) == radius:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and astar_map[nr, nc] != 0:
                            return nr, nc
        return r, c


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

