#!/usr/bin/env python3
"""
Collect training data for the fusion model in Habitat with the sensor rig in config/vinebot_rig.yaml
(heights AND positions of every camera, LiDAR at the rig height).

  conda activate habitat
  cd ~/robo_project_ws && source install/setup.bash
  python3 -m robo_project.ml.fusion.collect_data --scenes src/environment/IMR_lab8.glb \
      src/environment/Denmark.glb --samples-per-scene 20000 --out data/fusion_mid360

Each sample (one .npz, same format as before, read by fusion_data.FusionDataset):
  rgb_front / rgb_right / rgb_back / rgb_left   JPEG bytes (quality 90)
  depth_front                                   float16 metres (front RGB-D camera)
  scan                                          float32 nearest obstacle-band LiDAR return per bearing (n_rays
                                                over 360 deg, inf = none), measured from the LiDAR position
  scan_origin                                   float32 (x, y) of the LiDAR in base_footprint (rig: front mount)
LiDAR: the realistic Mid-360 model of sim/lidar_sim.py (front mount, measured beam pattern: it does not
see low obstacles close to the robot), with the robot's self-returns removed as on the real robot.
  label                                         uint8 egocentric free-space map, 6 m x 6 m at 5 cm (1 = free)
Poses: random points on the floor where the real robot fits (>= min_clearance from walls), random
heading over the full circle (the old data was mostly at 90-degree headings).
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from robo_project.common.rig import CAM_ORDER, drop_self_hits, load_rig
from robo_project.ml.fusion.lidar_local_occ import ego_crop
from robo_project.sim.lidar_sim import LidarModel, lidar_sensor_specs, points_to_scan

FINE_RES = 0.025      # m/px of the navmesh top-down view (LiDAR ray casting + labels)
LABEL_EXTENT = 6.0    # m, label side (cropped to the model extent at training time)
LABEL_RES = 0.05      # m/px of the label


def make_sim(scene: str, rig: dict, gpu_id: int):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.enable_physics = False
    backend.gpu_device_id = gpu_id
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    size = int(rig["image_size"])
    sensors = []
    for name in CAM_ORDER:
        cam_cfg = rig["cameras"][name]
        kinds = [("", habitat_sim.SensorType.COLOR)]
        if cam_cfg.get("depth"):
            kinds.append(("_depth", habitat_sim.SensorType.DEPTH))
        for suffix, kind in kinds:
            cam = habitat_sim.CameraSensorSpec()
            cam.uuid = name + suffix
            cam.sensor_type = kind
            cam.resolution = [size, size]
            # rig x forward / y left -> Habitat agent frame (forward = -Z, left = -X), as in sim/habitat_bridge.py
            cam.position = [-float(cam_cfg.get("y", 0.0)), float(cam_cfg["height"]), -float(cam_cfg.get("x", 0.0))]
            cam.orientation = [0.0, float(cam_cfg["yaw"]), 0.0]
            cam.hfov = float(cam_cfg["hfov_deg"])
            sensors.append(cam)
    sensors += lidar_sensor_specs(rig)
    agent_cfg.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    if not sim.pathfinder.is_loaded:
        navmesh = Path(scene).with_suffix(".navmesh")
        if navmesh.exists():
            sim.pathfinder.load_nav_mesh(str(navmesh))
    return sim


def floor_height(pf, n: int = 3000) -> float:
    """Height of the floor = the most common height of navigable points (desk tops etc. are fewer)."""
    ys = np.array([float(pf.get_random_navigable_point()[1]) for _ in range(n)])
    bins = np.round(ys / 0.1)
    vals, counts = np.unique(bins, return_counts=True)
    return float(np.median(ys[bins == vals[np.argmax(counts)]]))


def collect(scene: str, n: int, out_dir: Path, rig: dict, seed: int, gpu_id: int, min_clearance: float):
    rng = np.random.default_rng(seed)
    sim = make_sim(scene, rig, gpu_id)
    sim.seed(seed)
    pf = sim.pathfinder
    pf.seed(seed)
    agent = sim.initialize_agent(0)
    floor_y = floor_height(pf)
    lower = pf.get_bounds()[0]
    x_min, z_min = float(lower[0]), float(lower[2])
    topdown = np.asarray(pf.get_topdown_view(FINE_RES, floor_y)).astype(np.float32)
    lid = rig["lidar"]
    lidar = LidarModel(rig, seed)
    origin = np.array([float(lid.get("x", 0.0)), float(lid.get("y", 0.0))], np.float32)
    band = rig.get("obstacle_band", [0.10, 1.80])
    n_lab = int(round(LABEL_EXTENT / LABEL_RES))
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(scene).stem
    done = tries = 0
    t0 = time.time()
    while done < n:
        tries += 1
        pt = np.asarray(pf.get_random_navigable_point())
        if abs(float(pt[1]) - floor_y) > 0.3:
            continue
        # navmesh points are already ~agent radius from walls; this keeps the real robot body off them
        if pf.distance_to_closest_obstacle(pt, max_search_radius=2.0) < min_clearance:
            continue
        habitat_yaw = rng.uniform(-math.pi, math.pi)
        state = habitat_sim.AgentState()
        state.position = pt
        state.rotation = quat_from_angle_axis(habitat_yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        ros_yaw = habitat_yaw + math.pi / 2                       # map yaw (as in the bridge / training)
        row_m, col_m = float(pt[2]) - z_min, float(pt[0]) - x_min
        pts = drop_self_hits(lidar.points(obs), rig)
        ranges = points_to_scan(pts, origin, band, int(lid.get("n_rays", 720)), float(lid["max_range"]))
        label = (ego_crop(topdown, FINE_RES, row_m, col_m, ros_yaw, LABEL_EXTENT, n_lab) > 0.5).astype(np.uint8)
        sample = {f"rgb_{c}": np.frombuffer(cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(obs[c][:, :, :3]),
                                                                             cv2.COLOR_RGB2BGR),
                                                         [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), np.uint8)
                  for c in CAM_ORDER}
        sample["depth_front"] = np.asarray(obs["front_depth"], np.float32).astype(np.float16)
        sample["scan"] = ranges.astype(np.float32)
        sample["scan_origin"] = origin
        sample["label"] = label
        np.savez_compressed(out_dir / f"{name}_{seed}_{done:06d}.npz", **sample)
        done += 1
        if done % 1000 == 0:
            print(f"  {name}: {done}/{n}  ({done / (time.time() - t0):.0f} samples/s, "
                  f"{tries / done:.1f} tries per sample)", flush=True)
    sim.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--samples-per-scene", type=int, default=20000)
    ap.add_argument("--out", default="data/fusion_mid360")
    ap.add_argument("--rig", default="")
    ap.add_argument("--min-clearance", type=float, default=0.20,
                    help="m from the navmesh edge (+0.1 m agent radius = distance to walls)")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rig = load_rig(args.rig)
    print(f"rig: {rig['path']}")
    for i, scene in enumerate(args.scenes):
        print(f"Collecting {args.samples_per_scene} samples from {scene}", flush=True)
        collect(scene, args.samples_per_scene, Path(args.out) / Path(scene).stem, rig,
                args.seed + 1000 * i, args.gpu_id, args.min_clearance)


if __name__ == "__main__":
    main()
