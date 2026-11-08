"""
Realistic Livox Mid-360 simulation for Habitat, shared by sim/habitat_bridge.py (live) and
ml/fusion/collect_data.py (training data), so both see exactly the same LiDAR.

The old simulated LiDAR was a flat 2D scan of the navmesh: it saw every obstacle at any height.
The real Mid-360 (measured, see config/vinebot_rig.yaml) sits at the FRONT of the robot at 0.45 m
and looks only ~4-7 deg downward, and the robot's own deck blocks low beams to the rear. So it
does NOT see low obstacles close to the robot. This module renders 4 depth cameras (90 deg each)
at the LiDAR position and keeps only rays inside the measured beam pattern:
  - lower edge per azimuth from rig lidar.min_elevation_deg (10-deg sectors from -180 deg)
  - upper edge rig lidar.max_elevation_deg
  - range noise (lidar.range_noise), random subsampling to lidar.points_per_scan
Output points are in base_footprint (x forward, y left, z up from the ground), as the ROS nodes
get them on the real robot after the rig transform.
"""

import math

import numpy as np

DIRS = (("lidar_f", 0.0), ("lidar_l", 90.0), ("lidar_b", 180.0), ("lidar_r", -90.0))
WIDTH = 128          # px per 90 deg  (~0.7 deg / px)
HEIGHT = 160         # px: vertical field of view 2*atan(160/128) = 102 deg = +-51 deg, no pitch needed


def lidar_sensor_specs(rig: dict):
    """Habitat depth sensors at the LiDAR position (add to the agent's sensor_specifications)."""
    import habitat_sim
    lid = rig["lidar"]
    specs = []
    for uuid, yaw_deg in DIRS:
        cam = habitat_sim.CameraSensorSpec()
        cam.uuid = uuid
        cam.sensor_type = habitat_sim.SensorType.DEPTH
        cam.resolution = [HEIGHT, WIDTH]
        # rig x forward / y left -> Habitat agent frame (forward = -Z, left = -X), as for the cameras
        cam.position = [-float(lid.get("y", 0.0)), float(lid["height"]), -float(lid.get("x", 0.0))]
        cam.orientation = [0.0, math.radians(yaw_deg), 0.0]
        cam.hfov = 90.0
        specs.append(cam)
    return specs


class LidarModel:
    """Depth images of the 4 LiDAR sensors -> Nx3 points in base_footprint with the real beam pattern."""

    def __init__(self, rig: dict, seed=None):
        lid = rig["lidar"]
        self.origin = np.array([float(lid.get("x", 0.0)), float(lid.get("y", 0.0)), float(lid["height"])])
        self.max_range = float(lid.get("max_range", 15.0))
        self.noise = float(lid.get("range_noise", 0.02))
        self.n_keep = int(lid.get("points_per_scan", 20000))
        prof = lid.get("min_elevation_deg")
        self.min_el = np.radians(np.asarray(prof, np.float64)) if prof else np.full(36, math.radians(-7.0))
        self.max_el = math.radians(float(lid.get("max_elevation_deg", 52.0)))
        self.rng = np.random.default_rng(seed)
        # per-pixel ray directions in camera axes (forward, right, up) for a 90 deg camera
        f = (WIDTH / 2) / math.tan(math.radians(45.0))
        u = (np.arange(WIDTH) + 0.5 - WIDTH / 2) / f              # right
        v = (HEIGHT / 2 - (np.arange(HEIGHT) + 0.5)) / f          # up
        self.u, self.v = np.meshgrid(u, v)                        # HEIGHT x WIDTH

    def points(self, depths: dict) -> np.ndarray:
        """depths: uuid -> HxW z-depth (m, 0 = nothing) of the sensors in DIRS."""
        out = []
        for uuid, yaw_deg in DIRS:
            d = np.asarray(depths[uuid], np.float64)
            ok = d > 0.05
            dd, uu, vv = d[ok], self.u[ok], self.v[ok]
            fwd, right, up = dd, uu * dd, vv * dd
            psi = math.radians(yaw_deg)
            c, s = math.cos(psi), math.sin(psi)
            # camera forward (c, s), camera right (s, -c) in robot axes
            out.append(np.stack([c * fwd + s * right, s * fwd - c * right, up], 1))
        rel = np.concatenate(out) if out else np.zeros((0, 3))
        horiz = np.hypot(rel[:, 0], rel[:, 1])
        rng = np.linalg.norm(rel, axis=1)
        el = np.arctan2(rel[:, 2], horiz)
        az = np.arctan2(rel[:, 1], rel[:, 0])
        sector = ((np.degrees(az) + 180.0) // 10).astype(int) % len(self.min_el)
        keep = (el >= self.min_el[sector]) & (el <= self.max_el) & (rng <= self.max_range)
        rel, rng = rel[keep], rng[keep]
        if len(rel) > self.n_keep:
            idx = self.rng.choice(len(rel), self.n_keep, replace=False)
            rel, rng = rel[idx], rng[idx]
        if self.noise > 0 and len(rel):
            rel = rel * (1.0 + self.rng.normal(0.0, self.noise, len(rel)) / np.maximum(rng, 1e-3))[:, None]
        return rel + self.origin


def points_to_scan(pts_base: np.ndarray, origin_xy, band, n_bearings: int = 720, max_range: float = 15.0):
    """Points in base_footprint -> nearest obstacle-band return per bearing, measured from the LiDAR
    position origin_xy (inf = none). This is all the model's LiDAR grid needs (RangeLocalOcc keeps the
    nearest band return per bearing), so training samples store it instead of the full cloud."""
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    z = pts_base[:, 2]
    p = pts_base[(z >= band[0]) & (z <= band[1])]
    scan = np.full(n_bearings, np.inf)
    if len(p):
        r = np.hypot(p[:, 0] - ox, p[:, 1] - oy)
        a = np.arctan2(p[:, 1] - oy, p[:, 0] - ox)
        b = ((a + np.pi) / (2 * np.pi) * n_bearings).astype(int) % n_bearings
        ok = r <= max_range
        np.minimum.at(scan, b[ok], r[ok])
    return scan
