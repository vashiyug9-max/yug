#!/usr/bin/env python3
"""
Which cells of an egocentric local grid the robot's sensors can actually see (no ROS imports).

Used for the 1 / 0 / -1 target of FusionOccNet (fusion_data.py) and for the published grid
(fusion_obstacle_node.py):
   1 = free, 0 = occupied   cell is SEEN by at least one sensor
  -1 = unknown              no sensor can see it: outside every camera's field of view and the LiDAR's,
                            or hidden behind an occupied cell (behind a wall / furniture)
A cell is seen from a sensor when it lies inside that sensor's horizontal field of view and the straight
line from the sensor to the cell crosses no occupied cell (the cell itself may be occupied: the face of an
obstacle is seen). Sensors = the 4 cameras and the LiDAR of the rig (config/vinebot_rig.yaml). Cells under
the robot body are known (the robot stands on free floor).

Grid convention (same as RangeLocalOcc / the label): n x n, robot in the centre, row 0 = front (+x),
column 0 = left (+y).
"""

import math

import numpy as np

UNKNOWN = -1.0


def sensor_views(rig: dict):
    """[(x, y, yaw, half_fov), ...] of the rig's cameras and LiDAR in the base frame (load_rig() dict)."""
    views = []
    for cam in rig["cameras"].values():
        views.append((float(cam.get("x", 0.0)), float(cam.get("y", 0.0)), float(cam["yaw"]), float(cam["hfov"]) / 2))
    lid = rig["lidar"]
    views.append((float(lid.get("x", 0.0)), float(lid.get("y", 0.0)), 0.0, math.pi))
    return views


def body_half(rig: dict):
    robot = rig.get("robot", {}) if isinstance(rig.get("robot"), dict) else {}
    return float(robot.get("length", 1.15)) / 2, float(robot.get("width", 0.60)) / 2


class Visibility:
    """Precomputed ray samples for an n x n grid of side `extent`; call visible(free) per grid."""

    def __init__(self, n: int, extent: float, views, body=(0.575, 0.30), step_frac: float = 0.5):
        self.n, self.extent = n, extent
        res = extent / n
        self.res = res
        coords = (np.arange(n) + 0.5) * res - extent / 2
        cx = -coords[:, None] * np.ones((1, n))
        cy = -coords[None, :] * np.ones((n, 1))
        self.body = (np.abs(cx) <= body[0]) & (np.abs(cy) <= body[1])
        target = np.arange(n * n).reshape(n, n)
        self.rays = []          # per sensor: (cells in view [flat idx], sample cell idx [cells x K])
        for sx, sy, yaw, half in views:
            dx, dy = cx - sx, cy - sy
            ang = np.arctan2(dy, dx) - yaw
            ang = np.arctan2(np.sin(ang), np.cos(ang))
            inview = (np.abs(ang) <= half + 1e-9).ravel()
            dist = np.hypot(dx, dy).ravel()[inview]
            k = max(2, int(math.ceil(dist.max() / (res * step_frac))) if dist.size else 2)
            t = (np.arange(1, k) / k)[None, :]                     # samples strictly between sensor and cell
            px = sx + t * dx.ravel()[inview][:, None]
            py = sy + t * dy.ravel()[inview][:, None]
            rows = np.floor((extent / 2 - px) / res).astype(np.int64)
            cols = np.floor((extent / 2 - py) / res).astype(np.int64)
            inside = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
            idx = np.where(inside, rows * n + cols, -1)
            idx[idx == target.ravel()[inview][:, None]] = -1        # the cell itself never blocks its own view
            self.rays.append((np.nonzero(inview)[0], idx))

    def visible(self, free: np.ndarray) -> np.ndarray:
        """free: n x n (1 = free, 0 = occupied, values in between thresholded at 0.5) -> bool n x n seen."""
        occ = np.append((np.asarray(free, np.float32) < 0.5).ravel(), False)   # index -1 -> "not occupied"
        seen = self.body.ravel().copy()
        for cells, idx in self.rays:
            blocked = occ[idx].any(axis=1)
            seen[cells[~blocked]] = True
        return seen.reshape(self.n, self.n)


def with_unknown(free: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """1 / 0 grid + seen mask -> 1 / 0 / -1 grid (float32)."""
    out = np.asarray(free, np.float32).copy()
    out[~seen] = UNKNOWN
    return out
