#!/usr/bin/env python3
"""
PyTorch dataset for the fusion model (samples are collected with collect_data.py).

Each sample (one .npz) stores the raw sensors of the VineBot rig at a random pose:
  rgb_front / rgb_right / rgb_back / rgb_left   JPEG bytes
  depth_front                                   float16 metres (front RGB-D camera)
  scan                                          float32 LiDAR ranges, n_rays bearings over 360 deg (inf = none)
  label                                         uint8 egocentric free-space map, 6 m x 6 m at 5 cm (1 = free)
The geometric grids are computed from scan/depth at training time with the same code the robot runs,
so the grid extent (3 x coarse cell size) can be chosen per training run without re-collecting.
"""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from robo_project.ml.fusion.lidar_local_occ import RangeLocalOcc, depth_to_points, scan_to_points
from robo_project.common.rig import CAM_ORDER, load_rig

LABEL_EXTENT = 6.0
LABEL_RES = 0.05


# ── dataset ──────────────────────────────────────────────────────────────────

def split_files(files, val_frac: float, val_scenes=()):
    """Deterministic train/val split: by scene name if val_scenes is given, else by file-name hash."""
    train, val = [], []
    for f in files:
        if val_scenes:
            (val if Path(f).parent.name in val_scenes else train).append(f)
        else:
            h = int(hashlib.md5(Path(f).name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            (val if h < val_frac else train).append(f)
    return train, val


class FusionDataset:
    """
    Yields (pano, geo, label) for FusionOccNet, computing the geometric grids from the raw sensors.
    augment=True adds sensor dropout, missing LiDAR sectors (glass / occlusion), noise, colour jitter
    and left-right mirroring. `drop` forces a sensor off ("lidar", "depth", "cams") for evaluation.
    """

    def __init__(self, files, extent: float, rig: dict, grid: int = 64, augment: bool = False,
                 drop=(), p_drop_lidar=0.25, p_drop_depth=0.2, p_drop_cams=0.15, seed: int = 0):
        self.files = list(files)
        self.extent, self.grid, self.rig, self.augment, self.drop = extent, grid, rig, augment, set(drop)
        self.p = dict(lidar=p_drop_lidar, depth=p_drop_depth, cams=p_drop_cams)
        self.size = int(rig["image_size"])
        band = rig.get("obstacle_band", [0.10, 1.80])
        self.lidar_occ = RangeLocalOcc(grid, extent, band[0], band[1], float(rig["lidar"]["max_range"]))
        front = rig["cameras"]["front"]
        self.front = front
        self.depth_occ = RangeLocalOcc(grid, extent, band[0], band[1], float(rig["depth_max"]), fov=front["hfov"])
        self.seed = seed

    def __len__(self):
        return len(self.files)

    def raw(self, i):
        d = np.load(self.files[i])
        rgb = {c: cv2.cvtColor(cv2.imdecode(d[f"rgb_{c}"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for c in CAM_ORDER}
        # scan_origin: LiDAR position (x, y) the scan is measured from (new data: the real front mount);
        # old data has no origin = scan from the robot centre
        origin = d["scan_origin"].astype(np.float64) if "scan_origin" in d.files else np.zeros(2)
        return rgb, d["depth_front"].astype(np.float32), d["scan"].astype(np.float32), d["label"], origin

    def label_grid(self, label):
        n = int(round(self.extent / LABEL_RES))
        c0 = (label.shape[0] - n) // 2
        crop = label[c0:c0 + n, c0:c0 + n].astype(np.float32)
        return cv2.resize(crop, (self.grid, self.grid), interpolation=cv2.INTER_AREA)

    def __getitem__(self, i):
        from robo_project.ml.fusion.fusion_model import make_geo, make_pano
        rng = np.random.default_rng(None if self.augment else self.seed + i)
        rgb, depth, scan, label, origin = self.raw(i)
        angles = np.linspace(-np.pi, np.pi, len(scan), endpoint=False)
        drop = set(self.drop)
        if self.augment:
            for k, pk in self.p.items():
                if rng.random() < pk:
                    drop.add(k)
            if "lidar" in drop and "cams" in drop:      # never remove both main sensors
                drop.discard("cams")
            if rng.random() < 0.5:                      # mirror left <-> right
                rgb = {"front": rgb["front"][:, ::-1], "back": rgb["back"][:, ::-1],
                       "left": rgb["right"][:, ::-1], "right": rgb["left"][:, ::-1]}
                depth = depth[:, ::-1]
                scan = np.interp(-angles, angles, scan, period=2 * np.pi)
                origin[1] = -origin[1]
                label = label[:, ::-1]
            if rng.random() < 0.5:                      # LiDAR sectors without returns (glass, occlusion)
                for _ in range(rng.integers(1, 4)):
                    c, w = rng.uniform(-np.pi, np.pi), np.deg2rad(rng.uniform(15, 60))
                    scan[np.abs(np.angle(np.exp(1j * (angles - c)))) < w / 2] = np.inf
            scan = scan + rng.normal(0, 0.02, scan.shape)
            depth = depth * rng.normal(1.0, 0.01, depth.shape)
            depth[rng.random(depth.shape) < 0.05] = 0.0
            gain = rng.uniform(0.8, 1.2)
            rgb = {c: np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8) for c, im in rgb.items()}

        # sensor_z: the blind-zone filter uses the 3D distance from the LiDAR (at the rig height), as on
        # the robot. Without it, wall hits 10-15 cm away (Habitat's agent is thin) were dropped.
        lidar_grid = None if "lidar" in drop else self.lidar_occ.compute(
            scan_to_points(angles, scan) + np.array([origin[0], origin[1], 0.0]), tuple(origin),
            sensor_z=float(self.rig["lidar"]["height"]))[0]
        cam_x = float(self.front.get("x", 0.0))      # camera ahead of the robot centre (rig x)
        depth_grid = None if "depth" in drop else self.depth_occ.compute(
            depth_to_points(depth, self.front["hfov"], float(self.front["height"]), cam_x=cam_x,
                            max_depth=float(self.rig["depth_max"])), (cam_x, 0.0))[0]
        pano = make_pano({} if "cams" in drop else rgb, self.size)
        geo = make_geo(lidar_grid, depth_grid, self.grid)
        return pano, geo, self.label_grid(np.ascontiguousarray(label))[None]

