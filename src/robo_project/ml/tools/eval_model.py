#!/usr/bin/env python3
"""
How good is the fusion model on a recorded run?

For every /fusion/local_grid prediction in the bag, the ground truth is rebuilt exactly as the
training labels were (fusion_data.py: navmesh top-down -> ego_crop 6 m / 0.05 m -> centre crop of
`extent` -> resize to 64 x 64), at the robot pose of that frame. Baselines from the same frame:
  lidar  : LiDAR grid only (what the LiDAR alone gives, no learning)
  rule   : LiDAR + front depth, rule fusion (no learning)
Metrics (same as train_fusion.py):  cell3 = 3x3 cell accuracy,  iou = occupied-space IoU on 64 x 64
plus safety numbers on the 64 x 64 grid: miss = GT-occupied cells predicted free (dangerous),
false alarm = GT-free cells predicted occupied (detours).

usage: eval_model.py <bag_dir> <navmesh> <workspace_src_robo_project> <out_dir>
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, navmesh, pkg, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
sys.path.insert(0, str(pkg.parent))
from robo_project.ml.fusion.fusion_model import FusionInputs, rule_fuse            # noqa: E402
from robo_project.ml.fusion.lidar_local_occ import ego_crop                       # noqa: E402
from robo_project.common.rig import load_rig                                   # noqa: E402
from sensor_msgs_py.point_cloud2 import read_points_numpy               # noqa: E402

FINE_RES, LABEL_EXTENT, LABEL_RES, G = 0.025, 6.0, 0.05, 64


def read(topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    out = {t: [] for t in topics}
    while r.has_next():
        t, d, _ = r.read_next()
        out[t].append(deserialize_message(d, get_message(types[t])))
    return out


def ns(st):
    return st.sec * 1_000_000_000 + st.nanosec


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def metrics(pred_free, label):
    p3 = cv2.resize(pred_free, (3, 3), interpolation=cv2.INTER_AREA) > 0.5
    l3 = cv2.resize(label, (3, 3), interpolation=cv2.INTER_AREA) > 0.5
    po, lo = pred_free < 0.5, label < 0.5
    union = (po | lo).sum()
    iou = (po & lo).sum() / union if union else 1.0
    return dict(cell3=(p3 == l3).mean(), iou=iou, miss=(lo & ~po).sum(), occ=lo.sum(),
                fa=(po & ~lo).sum(), free=(~lo).sum(), has_obst=bool(lo.any()))


data = read(["/fusion/local_grid", "/odom", "/livox/lidar", "/camera/front/depth/image_raw"])
odom = data["/odom"]
o_ns = np.array([ns(m.header.stamp) for m in odom])
lidar = {ns(m.header.stamp): m for m in data["/livox/lidar"]}
depth = {ns(m.header.stamp): m for m in data["/camera/front/depth/image_raw"]}

import habitat_sim                                                      # noqa: E402
pf = habitat_sim.PathFinder()
pf.load_nav_mesh(navmesh)
lower = pf.get_bounds()[0]
p0 = odom[0].pose.pose.position
floor_y = float(pf.snap_point(np.array([p0.x, float(lower[1]) + 0.5, -p0.y], np.float32))[1])
topdown = np.asarray(pf.get_topdown_view(FINE_RES, floor_y)).astype(np.float32)
x_min, z_min = float(lower[0]), float(lower[2])

rig = load_rig(str(pkg / "config" / "vinebot_rig.yaml"))
grids = data["/fusion/local_grid"]
if not grids:
    sys.exit("no /fusion/local_grid in the bag")
EXTENT = float(grids[0].info.resolution * grids[0].info.width)   # model grid side (1.2 m, 3.0 m, ...)
fi = FusionInputs(rig, EXTENT)
n_lab = int(round(LABEL_EXTENT / LABEL_RES))
n_crop = int(round(EXTENT / LABEL_RES))
c0 = (n_lab - n_crop) // 2

res = {"model": [], "lidar": [], "rule": []}
skipped = 0
for m in data["/fusion/local_grid"]:
    t = ns(m.header.stamp)
    k = int(np.argmin(np.abs(o_ns - t)))
    if abs(o_ns[k] - t) > 30_000_000 or t not in lidar:
        skipped += 1
        continue
    pose = odom[k].pose.pose
    yaw = yaw_of(pose.orientation)
    lab = ego_crop(topdown, FINE_RES, -pose.position.y - z_min, pose.position.x - x_min, yaw, LABEL_EXTENT, n_lab)
    label = cv2.resize((lab > 0.5).astype(np.float32)[c0:c0 + n_crop, c0:c0 + n_crop], (G, G),
                       interpolation=cv2.INTER_AREA)
    occ = np.array(m.data, np.float32).reshape(G, G)                    # [iy(left), ix(forward)]
    pred = 1.0 - occ.T[::-1, ::-1] / 100.0                              # -> row 0 = front, col 0 = left
    pts = read_points_numpy(lidar[t], field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
    lg = fi.lidar_grid(pts)
    dg = None
    if t in depth:
        dm = depth[t]
        d = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.step // 4)[:, :dm.width]
        dg = fi.depth_grid(d)
    res["model"].append(metrics(pred, label))
    res["lidar"].append(metrics(lg, label))
    res["rule"].append(metrics(rule_fuse(lg, dg), label))

out_dir.mkdir(parents=True, exist_ok=True)
lines = [f"bag: {bag}", f"model grid: {EXTENT:.1f} m", f"frames evaluated: {len(res['model'])} (skipped {skipped} without pose/LiDAR match)",
         f"ground truth: navmesh top-down, same label pipeline as training (fusion_data.py)", "",
         f"{'':24s}{'cell3':>8s}{'IoU*':>8s}{'IoU obst':>10s}{'missed obst.':>14s}{'false alarm':>13s}"]
names = {"model": "FusionOccNet (all)", "lidar": "LiDAR only (no ML)", "rule": "LiDAR+depth rule"}
for key, rows in res.items():
    cell3 = np.mean([r["cell3"] for r in rows])
    obst = [r for r in rows if r["has_obst"]]
    iou = np.mean([r["iou"] for r in obst]) if obst else float("nan")
    miss = sum(r["miss"] for r in rows) / max(1, sum(r["occ"] for r in rows))
    fa = sum(r["fa"] for r in rows) / max(1, sum(r["free"] for r in rows))
    iou_all = np.mean([r["iou"] for r in rows])
    lines.append(f"{names[key]:24s}{cell3:8.1%}{iou_all:8.1%}{iou:10.1%}{miss:14.2%}{fa:13.2%}")
lines += ["", "IoU* = training-style (all frames, empty frames count 1.0, comparable to report.json)",
          f"IoU obst = averaged over frames with an obstacle in view ({len(obst)} of {len(res['model'])})",
          "missed obst. = share of truly occupied cells predicted free; false alarm = share of free cells "
          "predicted occupied"]
report = "\n".join(lines)
(out_dir / "model_eval.txt").write_text(report + "\n")
print(report)
