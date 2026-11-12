#!/usr/bin/env python3
"""
Sketch-map particle filter: tracks the robot pose on the hand-drawn map (/map) with odometry and the
ML local occupancy grid (/fusion/local_grid). Continuous heading, no global search jumps.

  ros2 run robo_project sketch_pf_node --ros-args -p odom_topic:=/wheel_odom
  (start pose: parameters init_x / init_y / init_yaw, or "2D Pose Estimate" in RViz = /initialpose)

Inputs
  /map                 runner_node's map (0.02 m/cell, origin and size from msg.info, ROS row order)
  odom_topic           wheel odometry (default /wheel_odom; simulation: sim/tools/wheel_odom_sim.py)
  /fusion/local_grid   ML 64 x 64 grid in base_footprint (0 = free ... 100 = occupied)
  /initialpose         (re)start the filter at a pose (RViz "2D Pose Estimate")
Outputs
  /localization/odom   nav_msgs/Odometry in "map": the pose for runner / planner / ML node, published for
                       EVERY odometry message (PF correction x odometry, odometry timestamp) - remap their
                       /odom to this topic, no code change:  --ros-args -r /odom:=/localization/odom
  /localization/pose   PoseWithCovarianceStamped (estimate + spread), /localization/particles PoseArray

Measurement model: the sketch can be ~1 m off and has no furniture, so occupancy is compared with a
BLURRED sketch (sigma_m) - an ML obstacle close to a drawn wall scores well, free ML space on a drawn
wall scores badly - and the per-cell evidence is tempered (cells of one grid are strongly correlated).
The filter only updates after the robot moved (min_trans / min_rot), like AMCL.

Defaults are for wheels + IMU odometry (robot_localization EKF, ~1% turn error): motion noise 5% / 3%, sketch
weight 5, heading hardly corrected from the sketch. Tested over 12 seeded replays: no divergence (larger
motion noise let the sketch drag the filter off in 2 of 4 runs). Wheel odometry ONLY (tracks over-report
turns): -p noise_trans_per_m:=0.08 -p noise_rot_per_rad:=0.10 -p corr_rot_per_rad:=0.15 -p corr_rot_per_m:=0.05
(heading errors then reach ~25 deg - an IMU is needed for safe autonomous driving).
"""
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    """2D pose a (x, y, yaw) then b in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def limit_step(corr, target, odom, max_t, max_r):
    """New map<-odom correction: the ROBOT's corrected pose (corr x odom) moves towards target x odom by at
    most max_t (m) and max_r (rad) - the published pose never jumps (obstacles mapped just before a
    correction stay consistent). Done on the robot pose, so a heading change does not swing it around."""
    cur, tgt = compose(corr, odom), compose(target, odom)
    dx, dy = tgt[0] - cur[0], tgt[1] - cur[1]
    d = math.hypot(dx, dy)
    k = 1.0 if d <= max_t else max_t / d
    dr = math.atan2(math.sin(tgt[2] - cur[2]), math.cos(tgt[2] - cur[2]))
    dr = max(-max_r, min(max_r, dr))
    new = (cur[0] + k * dx, cur[1] + k * dy, wrap(cur[2] + dr))
    return compose(new, inverse(odom))


class PFCore:
    """The particle filter without ROS (used by the node and by sim/tools/pf_replay.py)."""

    def __init__(self, n=600, sigma_m=0.40, p_floor=0.10, effective_cells=5.0, noise=(0.05, 0.03, 0.05, 0.02),
                 init_xy_std=0.20, init_yaw_std=0.10, seed=0):
        self.n, self.sigma_m, self.p_floor, self.eff_obs = n, sigma_m, p_floor, effective_cells
        self.a_tt, self.a_rr, self.a_rt, self.a_tr = noise
        self.init_xy_std, self.init_yaw_std = init_xy_std, init_yaw_std
        self.rng = np.random.default_rng(seed)
        self.logp = self.log1mp = None
        self.origin = self.res = self.shape = None
        self.parts = self.w = None
        self.neff = float(n)

    def set_map(self, occ, origin_xy, res):
        """occ: H x W bool/0-1 occupied in ROS row order (row 0 = min y)."""
        s = self.sigma_m / res
        k = int(2 * round(3 * s) + 1)
        p_occ = cv2.GaussianBlur(np.asarray(occ, np.float32), (k, k), s)
        p_occ = np.clip(p_occ, self.p_floor, 1.0 - self.p_floor)
        self.logp, self.log1mp = np.log(p_occ).astype(np.float32), np.log(1 - p_occ).astype(np.float32)
        self.origin, self.res, self.shape = (float(origin_xy[0]), float(origin_xy[1])), float(res), p_occ.shape

    def reset(self, pose):
        self.parts = np.column_stack([
            self.rng.normal(pose[0], self.init_xy_std, self.n),
            self.rng.normal(pose[1], self.init_xy_std, self.n),
            wrap(self.rng.normal(pose[2], self.init_yaw_std, self.n))])
        self.w = np.full(self.n, 1.0 / self.n)

    def predict(self, o0, o1):
        d = compose(inverse(o0), o1)                       # body-frame motion from odometry
        trans, rot = math.hypot(d[0], d[1]), abs(d[2])
        n = self.n
        st = self.a_tt * trans + self.a_tr * rot + 0.005
        sr = self.a_rr * rot + self.a_rt * trans + 0.005
        dx = d[0] + self.rng.normal(0, st, n)
        dy = d[1] + self.rng.normal(0, st, n)
        dth = d[2] + self.rng.normal(0, sr, n)
        c, s = np.cos(self.parts[:, 2]), np.sin(self.parts[:, 2])
        self.parts[:, 0] += c * dx - s * dy
        self.parts[:, 1] += s * dx + c * dy
        self.parts[:, 2] = wrap(self.parts[:, 2] + dth)

    def update(self, fx, ly, is_occ):
        """Observed cells in the robot frame (forward, left) and their state (True = occupied)."""
        P = self.parts
        c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
        wx = P[:, 0:1] + c * fx[None] - s * ly[None]
        wy = P[:, 1:2] + s * fx[None] + c * ly[None]
        col = np.floor((wx - self.origin[0]) / self.res).astype(np.int32)
        row = np.floor((wy - self.origin[1]) / self.res).astype(np.int32)
        h, w = self.shape
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        row, col = np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)
        ll = np.where(is_occ[None], self.logp[row, col], self.log1mp[row, col])
        ll = np.where(inside, ll, math.log(0.5)).sum(axis=1)
        lw = np.log(self.w + 1e-300) + (self.eff_obs / len(fx)) * ll
        lw -= lw.max()
        w = np.exp(lw)
        self.w = w / w.sum()
        self.neff = 1.0 / np.sum(self.w ** 2)
        if self.neff < 0.5 * self.n:
            pos = (self.rng.random() + np.arange(self.n)) / self.n
            idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), self.n - 1)
            self.parts = self.parts[idx].copy()
            self.w = np.full(self.n, 1.0 / self.n)

    def estimate(self):
        w = self.w
        x, y = float(np.sum(w * self.parts[:, 0])), float(np.sum(w * self.parts[:, 1]))
        yaw = math.atan2(float(np.sum(w * np.sin(self.parts[:, 2]))), float(np.sum(w * np.cos(self.parts[:, 2]))))
        return (x, y, yaw)

    def spread(self):
        return float(np.sqrt(max(np.cov(self.parts[:, 0], aweights=self.w), np.cov(self.parts[:, 1], aweights=self.w))))


def grid_cells(msg_info, data, stride=2, occ_min=65, free_max=35):
    """ML OccupancyGrid (base_footprint) -> observed cells (forward, left, occupied?)."""
    g = np.asarray(data, np.int16).reshape(msg_info.height, msg_info.width)[::stride, ::stride]
    iy, ix = np.mgrid[0:msg_info.height:stride, 0:msg_info.width:stride]
    fx = msg_info.origin.position.x + (ix + 0.5) * msg_info.resolution
    ly = msg_info.origin.position.y + (iy + 0.5) * msg_info.resolution
    occ_c, free_c = g >= occ_min, (g >= 0) & (g <= free_max)
    sel = occ_c | free_c
    return fx[sel], ly[sel], occ_c[sel]


class SketchPF(Node):
    def __init__(self):
        super().__init__("sketch_pf")
        p = self.declare_parameter
        self.n = int(p("particles", 600).value)
        self.odom_topic = str(p("odom_topic", "/wheel_odom").value)
        self.init = (float(p("init_x", -4.10).value), float(p("init_y", 4.90).value), float(p("init_yaw", 0.0).value))
        self.init_xy_std = float(p("init_xy_std", 0.20).value)
        self.init_yaw_std = float(p("init_yaw_std", 0.10).value)
        self.stride = int(p("cell_stride", 2).value)
        self.min_trans = float(p("min_trans", 0.05).value)
        self.min_rot = float(p("min_rot", 0.05).value)
        # Corrections are blended in proportionally to the motion since the last update (no jumps, and never
        # faster than the ML obstacle map can follow): at most corr_per_m x distance + corr_per_rad x turn (m),
        # and corr_rot_per_rad x turn + corr_rot_per_m x distance (rad).
        self.k_t = float(p("corr_per_m", 0.10).value)
        self.k_tr = float(p("corr_per_rad", 0.05).value)
        self.k_r = float(p("corr_rot_per_rad", 0.02).value)
        self.k_rt = float(p("corr_rot_per_m", 0.005).value)
        self.core = PFCore(
            n=self.n,
            sigma_m=float(p("sigma_m", 0.40).value),                # sketch inaccuracy (blur)
            p_floor=float(p("p_floor", 0.10).value),                # robustness: never trust one cell fully
            effective_cells=float(p("effective_cells", 5.0).value),  # tempering of correlated cells
            noise=(float(p("noise_trans_per_m", 0.05).value), float(p("noise_rot_per_rad", 0.03).value),
                   float(p("noise_rot_per_m", 0.05).value), float(p("noise_trans_per_rad", 0.02).value)),
            init_xy_std=self.init_xy_std, init_yaw_std=self.init_yaw_std)
        self.info = None
        self.started = False
        self.odom_buf = deque(maxlen=400)                           # (t, (x, y, yaw))
        self.last_update_odom = None                                # odom pose of the last PF update
        self.corr = None                                            # map <- odom correction
        self.n_updates = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, latched)
        self.create_subscription(OccupancyGrid, "/map", self.on_map, 10)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 50)
        self.create_subscription(OccupancyGrid, "/fusion/local_grid", self.on_grid, 5)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initialpose, 5)
        self.odom_pub = self.create_publisher(Odometry, "/localization/odom", 20)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/localization/pose", 5)
        self.parts_pub = self.create_publisher(PoseArray, "/localization/particles", 2)
        self.create_timer(0.5, self.publish_particles)
        self.get_logger().info(f"Sketch PF: {self.n} particles, odometry {self.odom_topic}, start "
                               f"({self.init[0]:.2f}, {self.init[1]:.2f}, {math.degrees(self.init[2]):.0f} deg). "
                               f"Waiting for /map ...")

    # ── map ────────────────────────────────────────────────────────────────
    def on_map(self, msg: OccupancyGrid):
        if self.info is not None:
            return
        h, w = msg.info.height, msg.info.width
        occ = np.asarray(msg.data, np.int16).reshape(h, w) != 0
        self.core.set_map(occ, (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution)
        self.info = msg.info
        self.get_logger().info(f"Map {w}x{h} @ {msg.info.resolution} m, origin ({msg.info.origin.position.x:.2f}, "
                               f"{msg.info.origin.position.y:.2f}); sketch blurred with sigma {self.core.sigma_m} m")

    # ── odometry ───────────────────────────────────────────────────────────
    def on_odom(self, msg: Odometry):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        o = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(msg.pose.pose.orientation))
        self.odom_buf.append((t, o))
        if not self.started:
            self.reset(self.init, o)
        est = compose(self.corr, o)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "map"
        out.child_frame_id = "base_footprint"
        out.pose.pose.position.x, out.pose.pose.position.y = est[0], est[1]
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = math.sin(est[2] / 2), math.cos(est[2] / 2)
        out.twist = msg.twist
        self.odom_pub.publish(out)

    def odom_at(self, t):
        if not self.odom_buf:
            return None
        best = min(self.odom_buf, key=lambda e: abs(e[0] - t))
        return best[1] if abs(best[0] - t) < 0.15 else None

    # ── filter ─────────────────────────────────────────────────────────────
    def reset(self, pose, odom):
        self.core.reset(pose)
        self.started = True
        self.last_update_odom = odom
        self.corr = compose(pose, inverse(odom))
        self.get_logger().info(f"PF started at ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)")

    def on_initialpose(self, msg: PoseWithCovarianceStamped):
        if not self.odom_buf:
            return
        q = msg.pose.pose
        self.reset((q.position.x, q.position.y, yaw_of(q.orientation)), self.odom_buf[-1][1])

    def on_grid(self, msg: OccupancyGrid):
        if self.info is None or not self.started:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        o = self.odom_at(t)
        if o is None:
            return
        d = compose(inverse(self.last_update_odom), o)
        if math.hypot(d[0], d[1]) < self.min_trans and abs(d[2]) < self.min_rot:
            return
        fx, ly, is_occ = grid_cells(msg.info, msg.data, self.stride)
        if len(fx) < 20:
            return
        self.core.predict(self.last_update_odom, o)
        self.last_update_odom = o
        self.core.update(fx, ly, is_occ)
        est = self.core.estimate()
        dist, turn = math.hypot(d[0], d[1]), abs(d[2])
        self.corr = limit_step(self.corr, compose(est, inverse(o)), o,
                               self.k_t * dist + self.k_tr * turn + 0.001,
                               self.k_r * turn + self.k_rt * dist + math.radians(0.05))
        self.n_updates += 1
        self.publish_pose(msg.header.stamp, est)
        if self.n_updates % 20 == 0:
            self.get_logger().info(f"PF update {self.n_updates}: ({est[0]:.2f}, {est[1]:.2f}, "
                                   f"{math.degrees(est[2]):.0f} deg), spread {self.core.spread():.2f} m, "
                                   f"Neff {self.core.neff:.0f}")

    # ── outputs ────────────────────────────────────────────────────────────
    def publish_pose(self, stamp, est):
        m = PoseWithCovarianceStamped()
        m.header.stamp = stamp
        m.header.frame_id = "map"
        m.pose.pose.position.x, m.pose.pose.position.y = est[0], est[1]
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(est[2] / 2), math.cos(est[2] / 2)
        cov = np.cov(self.core.parts[:, :2].T, aweights=self.core.w)
        m.pose.covariance[0], m.pose.covariance[1] = cov[0, 0], cov[0, 1]
        m.pose.covariance[6], m.pose.covariance[7] = cov[1, 0], cov[1, 1]
        m.pose.covariance[35] = float(np.cov(self.core.parts[:, 2], aweights=self.core.w))
        self.pose_pub.publish(m)

    def publish_particles(self):
        if not self.started:
            return
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = "map"
        for x, y, th in self.core.parts[::max(1, self.n // 200)]:
            q = Pose()
            q.position.x, q.position.y = float(x), float(y)
            q.orientation.z, q.orientation.w = math.sin(th / 2), math.cos(th / 2)
            pa.poses.append(q)
        self.parts_pub.publish(pa)


def main():
    rclpy.init()
    node = SketchPF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
