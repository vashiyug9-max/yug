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

from robo_project.scripts.basic_types import PosePixels
# Compiled A* + path smoothing: same paths as scripts/astar.py, ~30 ms instead of up to ~9 s.
from robo_project.scripts.fast_astar import run_astar, smooth_path

# ── GOAL in world coordinates (meters) ───────────────────────────────────────
GOAL_X = 0.0   # meters  (default goal; a "2D Goal Pose" clicked in RViz replaces it)
GOAL_Y = 0.0   # meters

# Walls are grown by this much before planning, so the path keeps the robot
# body away from them instead of hugging the wall line.
INFLATION_M = 0.20

# Obstacles reported by runner_node on /blocked_point (things that exist in
# Habitat but not on the hand-drawn map) are added as discs of this radius.
BLOCKED_RADIUS_M = 0.25

# Obstacles seen by the fusion ML model (fusion_obstacle_node, /fusion/obstacle_map, aligned with
# /map) are added to the A* map like walls.  When one lies on the remaining path, re-plan at once.
# Path points closer than ML_CHECK_MIN_DIST_M to the robot are not checked (the robot may already
# be inside a wall's inflation; the new path starts from the nearest free cell anyway).
ML_CHECK_MIN_DIST_M = 0.30
REPLAN_MIN_PERIOD_S = 0.5   # at most one ML-triggered re-plan per this period
NO_PATH_RETRY_S = 2.0       # no path: stop the robot and try again after this


class MotionPlanner(Node):

    def __init__(self):
        super().__init__('motion_planner')

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
        self.ml_inflated = None     # ...grown by INFLATION_M (what the path must avoid)
        self.ml_warned = False
        self.create_subscription(OccupancyGrid, '/fusion/obstacle_map', self.ml_map_callback, 2)
        self.current_wp = None      # runner_node's current waypoint (x, y)
        self.create_subscription(PoseStamped, '/runner_current_waypoint', self.current_wp_callback, 5)
        self.last_replan = 0.0
        self.no_path_since = None

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
        self.blocked_points.append((float(msg.point.x), float(msg.point.y)))
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
        self.ml_occ = np.frombuffer(bytes(msg.data), dtype=np.int8).reshape(self.map_height, self.map_width) == 100
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
        b_px = int(round(BLOCKED_RADIUS_M / self.map_resolution))
        for bx, by in self.blocked_points:
            b_row, b_col = self.world_to_pixel(bx, by)
            cv2.circle(free, (b_col, b_row), b_px, 0, -1)
        n_ml = 0
        if self.ml_occ is not None:
            free[self.ml_occ] = 0          # obstacles seen by the ML model
            n_ml = int(self.ml_occ.sum())
        r_px = max(1, int(round(INFLATION_M / self.map_resolution)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
        astar_map = cv2.erode(free, kernel)

        start_row, start_col = self._snap_to_free(astar_map, start_row, start_col)
        goal_row, goal_col = self._snap_to_free(astar_map, goal_row, goal_col)

        start_px = PosePixels(start_row, start_col)
        goal_px = PosePixels(goal_row, goal_col)

        t0 = time.monotonic()
        self.last_replan = t0
        path = run_astar(astar_map, start_px, goal_px, include_diagonals=True)

        if not path:
            self.get_logger().error(
                "A* found NO PATH between start and goal (they are not connected on the inflated map, "
                f"{n_ml} ML obstacle cells). Stopping the robot, retrying in {NO_PATH_RETRY_S:.0f} s.")
            self.no_path_since = time.monotonic()
            self._publish_stop()
            return
        self.no_path_since = None

        # run_astar returns the path REVERSED (goal->start), one cell per
        # step.  smooth_path() keeps only the corners that are needed
        # (line-of-sight on the inflated map), so the robot drives straight
        # segments instead of a 2 cm staircase.
        smoothed = smooth_path(astar_map, path)
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

