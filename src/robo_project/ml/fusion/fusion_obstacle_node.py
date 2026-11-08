#!/usr/bin/env python3
"""
Fusion obstacle node: 4 RGB cameras + 360 deg LiDAR + front depth -> trained FusionOccNet
-> egocentric local occupancy grid -> persistent obstacle map for motion_planner.

The model (fusion_model.py, checkpoint runs/fusion_full_e1.2/best.pt) predicts a 1.2 m x 1.2 m
grid (64 x 64, forward = top row, left = left column) of free-space probability around the robot.
Inputs are prepared with FusionInputs, i.e. exactly as in training, from sensors that share the
same Habitat frame stamp; the pose used to place the prediction is the /odom pose at that stamp.

Every prediction is fused into a world map with the geometry of /map (same size, origin,
resolution) using clamped log-odds: one confident frame marks a cell, later frames that see it
free clear it again, so single wrong frames neither stick nor block for long. The map is published
on /fusion/obstacle_map (100 = obstacle, 0 = free, -1 = never seen); motion_planner adds the
obstacles to its A* map and re-plans as soon as one lies on the current path.

Subscribes:  /camera/{front,right,back,left}/image_raw, /camera/front/depth/image_raw (32FC1),
             /livox/lidar (PointCloud2 in base_footprint), /odom, /map
Publishes:   /fusion/local_grid   (OccupancyGrid in base_footprint: the raw prediction, for RViz)
             /fusion/obstacle_map (OccupancyGrid in map, aligned with /map)

Parameters:
  model_path       checkpoint (default: runs/fusion_full_e1.2/best.pt found above this file)
  rig              sensor rig yaml (default: config/vinebot_rig.yaml)
  device           "cpu" / "cuda" (default: cuda if available)
  use_cameras / use_lidar / use_depth   leave a sensor out (the model was trained with dropout)
  rate             max predictions per second (default 10; CPU inference takes ~0.1 s)
  p_clip           per-frame probabilities are clipped to [p_clip, 1 - p_clip] (default 0.03)
  l_max            log-odds clamp (default 7.0: ~2 confident frames to flip a cell)
  l_occ            a cell is an obstacle above this log-odds (default 1.0)
  l_free           ...and free below this (default -1.0)
  lidar_map        also fuse the raw LiDAR beyond the model's square (default true)
  lidar_map_range  m, how far (default 4.0); lidar_hit / lidar_free: log-odds per hit / free ray
  lidar_free_max_height  m (0.25): LiDAR free space only where its lowest beam is below this height
"""

import math
import time
from array import array
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy

from robo_project.ml.fusion.fusion_model import FusionInputs, FusionPredictor, default_fusion_model
from robo_project.common.rig import CAM_ORDER, load_rig, rig_signature, lidar_to_base, drop_self_hits


def stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_rgb(msg: Image) -> np.ndarray:
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
    img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch)[:, :, :3]
    return np.ascontiguousarray(img[:, :, ::-1] if msg.encoding.startswith("bgr") else img)


def decode_depth(msg: Image) -> np.ndarray:
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width].copy()
    if msg.encoding in ("16UC1", "mono16"):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return d.astype(np.float32) / 1000.0
    raise ValueError(f"Unsupported depth encoding {msg.encoding}")


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class FusionObstacleNode(Node):
    def __init__(self):
        super().__init__("fusion_obstacle_node")
        p = self.declare_parameter
        p("model_path", "")
        p("rig", "")
        p("device", "")
        p("use_cameras", True)
        p("use_lidar", True)
        p("use_depth", True)
        p("rate", 10.0)
        p("p_clip", 0.03)
        p("l_max", 7.0)
        p("l_occ", 1.0)
        p("l_free", -1.0)
        p("lidar_map", True)
        p("lidar_map_range", 4.0)
        p("lidar_hit", 2.0)
        p("lidar_free", 0.7)
        p("lidar_free_max_height", 0.25)
        g = lambda n: self.get_parameter(n).value

        path = str(g("model_path")) or default_fusion_model()
        if not path:
            raise FileNotFoundError("Fusion checkpoint not found; pass model_path:=/path/to/best.pt")
        self.model = FusionPredictor(path, str(g("device")) or None)
        self.extent = self.model.extent
        self.rig = load_rig(str(g("rig")))
        self.inputs = FusionInputs(self.rig, self.extent)
        trained_for = self.model.cfg.get("rig_sensors")
        if trained_for is None:
            self.get_logger().warn("This checkpoint does not record its sensor rig (old model): it was trained "
                                   "with all sensors at 1.0 m - check it matches config/vinebot_rig.yaml.")
        elif trained_for != rig_signature(self.rig):
            self.get_logger().warn(f"Model was trained for another sensor rig than {self.rig['path']}: "
                                   "predictions will be worse. Re-train with ml/fusion/collect_data.py + train_fusion.py.")
        # Long-range LiDAR layer: the model only sees +-extent/2 around the robot centre (0.6 m for
        # the 1.2 m model - the real robot's bumper is at 0.575 m), the LiDAR sees far. Beyond the
        # model's square, cells along each LiDAR ray are marked free up to the first hit and the hit
        # occupied, out to lidar_map_range; inside the square the model decides.
        self.lidar_map = bool(g("lidar_map"))
        self.lidar_range = float(g("lidar_map_range"))
        self.lidar_hit, self.lidar_free = float(g("lidar_hit")), float(g("lidar_free"))
        lid = self.rig.get("lidar", {})
        self.lidar_xy = (float(lid.get("x", 0.0)), float(lid.get("y", 0.0)))
        self.band = self.rig.get("obstacle_band", [0.10, 1.80])
        self.lidar_z = float(lid.get("height", 0.45))
        prof = lid.get("min_elevation_deg")
        self.min_el = np.radians(np.asarray(prof, np.float64)) if prof else None
        self.lidar_free_max_h = float(g("lidar_free_max_height"))
        self.use = {"cams": bool(g("use_cameras")), "lidar": bool(g("use_lidar")), "depth": bool(g("use_depth"))}
        self.period = 1.0 / max(float(g("rate")), 0.1)
        pc = float(g("p_clip"))
        self.l_cap = math.log((1 - pc) / pc)
        self.l_max, self.l_occ, self.l_free = float(g("l_max")), float(g("l_occ")), float(g("l_free"))
        self.n = 64
        self.res = self.extent / self.n

        self.rgb = {}                     # cam -> (ns, msg)
        self.depth = None                 # (ns, msg)
        self.cloud = None                 # (ns, msg)
        self.poses = deque(maxlen=400)    # (ns, x, y, yaw) from /odom
        self.last_done_ns = -1
        self.last_run = 0.0
        self.n_pred = 0
        self.t_pred = 0.0
        self.waiting_ns, self.waiting_since = -1, 0.0
        self.n_incomplete = 0             # frames predicted with a sensor missing (timeout)
        self.warned_frame = False

        self.map_info = None              # geometry of /map
        self.logodds = None               # H x W float32
        self.state = None                 # H x W int8 published
        self.last_pub = 0.0

        for cam in CAM_ORDER:
            self.create_subscription(Image, f"/camera/{cam}/image_raw",
                                     lambda m, cam=cam: self.rgb.__setitem__(cam, (stamp_ns(m.header.stamp), m)), 2)
        self.create_subscription(Image, "/camera/front/depth/image_raw",
                                 lambda m: setattr(self, "depth", (stamp_ns(m.header.stamp), m)), 2)
        self.create_subscription(PointCloud2, "/livox/lidar",
                                 lambda m: setattr(self, "cloud", (stamp_ns(m.header.stamp), m)), qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 50)
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, 10)

        self.grid_pub = self.create_publisher(OccupancyGrid, "/fusion/local_grid", 2)
        self.obst_pub = self.create_publisher(OccupancyGrid, "/fusion/obstacle_map", 2)
        self.create_timer(0.02, self.tick)
        self.create_timer(1.0, self.heartbeat)
        on = [k for k, v in self.use.items() if v]
        self.get_logger().info(f"Fusion model {path} on {self.model.device}, local grid {self.extent} m, "
                               f"sensors {on}. Waiting for /map ...")

    # ── inputs ───────────────────────────────────────────────────────────
    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose
        self.poses.append((stamp_ns(msg.header.stamp), q.position.x, q.position.y, yaw_of(q.orientation)))

    def map_cb(self, msg: OccupancyGrid):
        if self.map_info is not None:
            return
        self.map_info = msg.info
        h, w = msg.info.height, msg.info.width
        self.logodds = np.zeros((h, w), np.float32)
        self.state = np.full((h, w), -1, np.int8)
        self.get_logger().info(f"Obstacle map {w}x{h} @ {msg.info.resolution} m aligned with /map. "
                               f"Publishing /fusion/obstacle_map")

    def pose_at(self, ns: int):
        if not self.poses:
            return None
        best = min(self.poses, key=lambda s: abs(s[0] - ns))
        if abs(best[0] - ns) > 100_000_000:          # no pose within 0.1 s of the frame
            return None
        return best[1:]

    # ── main loop ────────────────────────────────────────────────────────
    def tick(self):
        if "front" not in self.rgb or time.monotonic() - self.last_run < self.period:
            return
        ns = self.rgb["front"][0]
        if ns == self.last_done_ns:
            return
        # Wait (max 0.1 s) until every enabled sensor delivered this same frame.
        complete = ((not self.use["cams"] or all(c in self.rgb and self.rgb[c][0] == ns for c in CAM_ORDER))
                    and (not self.use["lidar"] or (self.cloud is not None and self.cloud[0] == ns))
                    and (not self.use["depth"] or (self.depth is not None and self.depth[0] == ns)))
        if not complete:
            if self.waiting_ns != ns:
                self.waiting_ns, self.waiting_since = ns, time.monotonic()
                return
            if time.monotonic() - self.waiting_since < 0.1:
                return
            self.n_incomplete += 1
        self.last_done_ns = ns
        self.last_run = time.monotonic()

        rgb = {}
        if self.use["cams"]:
            # only images of this same frame (the bridge stamps all sensors of a frame alike)
            rgb = {c: decode_rgb(m) for c, (t, m) in self.rgb.items() if t == ns}
        lidar_grid = depth_grid = pts = None
        if self.use["lidar"] and self.cloud is not None and self.cloud[0] == ns:
            cmsg = self.cloud[1]
            pts = read_points_numpy(cmsg, field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
            if cmsg.header.frame_id not in ("", "base_footprint"):
                # real robot: the Livox driver publishes in the sensor's own frame (livox_frame);
                # move the points to base_footprint with the measured mounting in vinebot_rig.yaml
                pts = lidar_to_base(pts, self.rig)
                if not self.warned_frame:
                    lid = self.rig["lidar"]
                    self.get_logger().info(f"/livox/lidar in frame '{cmsg.header.frame_id}': transformed to "
                                           f"base_footprint with the rig LiDAR mounting (height {lid['height']} m, "
                                           f"yaw {lid.get('yaw_deg', 0.0)} deg)")
                    self.warned_frame = True
            pts = drop_self_hits(pts, self.rig)        # returns from the robot's own masts / mounts
            lidar_grid = self.inputs.lidar_grid(pts)
        if self.use["depth"] and self.depth is not None and self.depth[0] == ns:
            depth_grid = self.inputs.depth_grid(decode_depth(self.depth[1]))

        t = time.monotonic()
        free = self.model.predict(rgb, lidar_grid, depth_grid)
        self.t_pred += time.monotonic() - t
        self.n_pred += 1
        self.publish_local(free, self.rgb["front"][1].header.stamp)

        pose = self.pose_at(ns)
        if pose is not None and self.logodds is not None:
            changed = self.integrate(free, pose)
            if self.lidar_map and pts is not None:
                changed = self.integrate_lidar(pts, pose) or changed
            if changed:
                self.publish_map()

    def integrate(self, free: np.ndarray, pose) -> bool:
        """Fuse one prediction into the world log-odds map. Returns True if an obstacle cell changed."""
        info = self.map_info
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        H, W = self.logodds.shape
        rx, ry, yaw = pose
        half = self.extent / 2
        reach = half * math.sqrt(2)
        c0 = max(0, int((rx - reach - ox) / res))
        c1 = min(W, int((rx + reach - ox) / res) + 1)
        r0 = max(0, int((ry - reach - oy) / res))
        r1 = min(H, int((ry + reach - oy) / res) + 1)
        if c0 >= c1 or r0 >= r1:
            return False
        cols = np.arange(c0, c1)
        rows = np.arange(r0, r1)
        wx = ox + (cols[None, :] + 0.5) * res - rx          # map pixel centres relative to the robot
        wy = oy + (rows[:, None] + 0.5) * res - ry
        c, s = math.cos(yaw), math.sin(yaw)
        fx = c * wx + s * wy                                 # forward
        ly = -s * wx + c * wy                                # left
        inside = (np.abs(fx) < half) & (np.abs(ly) < half)
        gi = np.clip(((half - fx) / self.res).astype(int), 0, self.n - 1)
        gj = np.clip(((half - ly) / self.res).astype(int), 0, self.n - 1)
        p_free = free[gi, gj]
        l = np.clip(np.log(np.clip(1.0 - p_free, 1e-6, 1.0) / np.clip(p_free, 1e-6, 1.0)), -self.l_cap, self.l_cap)

        win = self.logodds[r0:r1, c0:c1]
        win[inside] = np.clip(win[inside] + l[inside].astype(np.float32), -self.l_max, self.l_max)
        st = self.state[r0:r1, c0:c1]
        old_occ = st == 100
        new = np.where(win > self.l_occ, 100, np.where(win < self.l_free, 0, st)).astype(np.int8)
        st[:] = new
        return bool(((st == 100) != old_occ).any())

    def integrate_lidar(self, pts: np.ndarray, pose, n_bearings: int = 720, hit_tol: float = 0.05) -> bool:
        """LiDAR layer (up to lidar_range): along each
        bearing, the cell at the nearest obstacle return is occupied (everywhere) and the cells before it
        are free (outside the model's square only),
        cells behind it are left alone. Returns True if an obstacle cell changed."""
        sx, sy = self.lidar_xy
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        keep = (z >= self.band[0]) & (z <= self.band[1]) & np.isfinite(x) & np.isfinite(y)
        # blind zone around the sensor (self-returns of the mount; Mid-360 ~0.1 m). Not the whole
        # body box: with one return per bearing, dropping a close wall would mark the space behind it free.
        keep &= np.hypot(x - sx, y - sy) >= 0.15
        x, y = x[keep], y[keep]
        nearest = np.full(n_bearings, np.inf)
        if len(x):
            b = ((np.arctan2(y - sy, x - sx) + np.pi) / (2 * np.pi) * n_bearings).astype(int) % n_bearings
            np.minimum.at(nearest, b, np.hypot(x - sx, y - sy))
            filled = nearest.copy()                      # fill single empty bins between returns
            for k in (1, 2):
                cand = np.minimum(np.roll(nearest, k), np.roll(nearest, -k))
                upd = ~np.isfinite(filled) & np.isfinite(cand)
                filled[upd] = cand[upd]
            nearest = filled

        info = self.map_info
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        H, W = self.logodds.shape
        rx, ry, yaw = pose
        R = self.lidar_range
        c0, c1 = max(0, int((rx - R - ox) / res)), min(W, int((rx + R - ox) / res) + 1)
        r0, r1 = max(0, int((ry - R - oy) / res)), min(H, int((ry + R - oy) / res) + 1)
        if c0 >= c1 or r0 >= r1:
            return False
        wx = ox + (np.arange(c0, c1)[None, :] + 0.5) * res - rx
        wy = oy + (np.arange(r0, r1)[:, None] + 0.5) * res - ry
        c, s = math.cos(yaw), math.sin(yaw)
        fx = c * wx + s * wy                                 # forward (base frame)
        ly = -s * wx + c * wy                                # left
        half = self.extent / 2
        outside_model = ~((np.abs(fx) < half) & (np.abs(ly) < half))
        dx, dy = fx - sx, ly - sy
        rng = np.hypot(dx, dy)
        b = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * n_bearings).astype(int) % n_bearings
        near = nearest[b]
        in_range = rng <= R
        # LiDAR hits are real surfaces: add them everywhere, also inside the model's square (obstacles of
        # model and LiDAR combined). Free space from the LiDAR only outside the square (inside, the model
        # decides - it also sees what the LiDAR misses, e.g. low or thin obstacles).
        hit = in_range & (np.abs(rng - near) <= hit_tol)
        free = outside_model & in_range & (rng < near - hit_tol)
        # The real Mid-360 looks only 4-7 deg down (rig lidar.min_elevation_deg): close to the robot its
        # lowest beam passes OVER low obstacles. A cell is free only where that beam is low enough to have
        # seen an obstacle of lidar_free_max_height (otherwise "no return" says nothing about low objects).
        if self.min_el is not None:
            sector = ((np.degrees(np.arctan2(dy, dx)) + 180.0) // 10).astype(int) % len(self.min_el)
            beam_z = self.lidar_z + rng * np.tan(self.min_el[sector])
            free &= beam_z <= self.lidar_free_max_h

        win = self.logodds[r0:r1, c0:c1]
        win[free] = np.maximum(win[free] - self.lidar_free, -self.l_max)
        win[hit] = np.minimum(win[hit] + self.lidar_hit, self.l_max)
        st = self.state[r0:r1, c0:c1]
        old_occ = st == 100
        st[:] = np.where(win > self.l_occ, 100, np.where(win < self.l_free, 0, st)).astype(np.int8)
        return bool(((st == 100) != old_occ).any())

    def publish_local(self, free: np.ndarray, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_footprint"
        msg.info.resolution = float(self.res)
        msg.info.width = msg.info.height = self.n
        msg.info.origin.position.x = -self.extent / 2
        msg.info.origin.position.y = -self.extent / 2
        msg.info.origin.orientation.w = 1.0
        # data[iy, ix]: x = ix (forward), y = iy (left)  <-  free[row, col] with row 0 = front, col 0 = left
        occ = np.clip(np.round((1.0 - free[::-1, ::-1].T) * 100.0), 0, 100).astype(np.int8)
        msg.data = occ.ravel().tolist()
        self.grid_pub.publish(msg)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info = self.map_info
        msg.data = array('b', self.state.tobytes())
        self.obst_pub.publish(msg)
        self.last_pub = time.monotonic()

    def heartbeat(self):
        if self.state is not None and time.monotonic() - self.last_pub > 0.9:
            self.publish_map()
        if self.n_pred >= 50:
            occ = int((self.state == 100).sum()) if self.state is not None else 0
            self.get_logger().info(f"{self.n_pred} predictions ({self.n_incomplete} with a sensor missing), "
                                   f"{self.t_pred / self.n_pred * 1000:.0f} ms each, "
                                   f"{occ} obstacle cells in the map", throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = FusionObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
