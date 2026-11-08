# Re-run the fusion model offline on a recorded bag WITHOUT cameras (LiDAR + depth only) and compare to
# ground truth, with the same label pipeline and metrics as eval_model.py. Also prints LiDAR alone.
# usage (from ~/robo_project_ws, conda habitat + install sourced):
#   python3 src/robo_project/ml/tools/eval_model_no_cameras.py bags/my_run [model.pt]
import math, sys
from pathlib import Path
import cv2, numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from robo_project.ml.fusion.fusion_model import FusionInputs, FusionPredictor, default_fusion_model
from robo_project.ml.fusion.lidar_local_occ import ego_crop
from robo_project.common.rig import load_rig
from sensor_msgs_py.point_cloud2 import read_points_numpy
import habitat_sim
bag = sys.argv[1]
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
r.set_filter(rosbag2_py.StorageFilter(topics=["/odom", "/livox/lidar", "/camera/front/depth/image_raw"]))
data = {"/odom": [], "/livox/lidar": {}, "/camera/front/depth/image_raw": {}}
ns = lambda s: s.sec * 1_000_000_000 + s.nanosec
while r.has_next():
    t, d, _ = r.read_next(); m = deserialize_message(d, get_message(types[t]))
    if t == "/odom": data[t].append(m)
    else: data[t][ns(m.header.stamp)] = m
odom = data["/odom"]; o_ns = np.array([ns(m.header.stamp) for m in odom])
pf = habitat_sim.PathFinder(); pf.load_nav_mesh("src/environment/IMR_lab8.navmesh"); lo = pf.get_bounds()[0]
p0 = odom[0].pose.pose.position
fy = float(pf.snap_point(np.array([p0.x, float(lo[1]) + .5, -p0.y], np.float32))[1])
td = np.asarray(pf.get_topdown_view(0.025, fy)).astype(np.float32)
rig = load_rig(); model = FusionPredictor(sys.argv[2] if len(sys.argv) > 2 else default_fusion_model())
fi = FusionInputs(rig, model.extent)
n_crop = int(round(model.extent / 0.05)); c0 = (120 - n_crop) // 2
print(f"model: {sys.argv[2] if len(sys.argv) > 2 else default_fusion_model()} (grid {model.extent} m)")
yaw_of = lambda q: math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
def metrics(pred, lab):
    po, lo_ = pred < 0.5, lab < 0.5; u = (po | lo_).sum()
    p3 = cv2.resize(pred, (3, 3), interpolation=cv2.INTER_AREA) > .5; l3 = cv2.resize(lab, (3, 3), interpolation=cv2.INTER_AREA) > .5
    return (p3 == l3).mean(), ((po & lo_).sum() / u if u else 1.0), (lo_ & ~po).sum(), lo_.sum(), (po & ~lo_).sum(), (~lo_).sum()
res = {"no cameras (LiDAR+depth)": [], "LiDAR only": []}
keys = sorted(data["/livox/lidar"])[::3]
for t in keys:
    k = int(np.argmin(np.abs(o_ns - t)))
    if abs(o_ns[k] - t) > 30_000_000: continue
    pose = odom[k].pose.pose
    lab = ego_crop(td, 0.025, -pose.position.y - float(lo[2]), pose.position.x - float(lo[0]), yaw_of(pose.orientation), 6.0, 120)
    label = cv2.resize((lab > .5).astype(np.float32)[c0:c0 + n_crop, c0:c0 + n_crop], (64, 64), interpolation=cv2.INTER_AREA)
    pts = read_points_numpy(data["/livox/lidar"][t], field_names=("x", "y", "z"), skip_nans=True).astype(np.float64)
    lg = fi.lidar_grid(pts); dg = None
    if t in data["/camera/front/depth/image_raw"]:
        dm = data["/camera/front/depth/image_raw"][t]
        dg = fi.depth_grid(np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.step // 4)[:, :dm.width])
    res["no cameras (LiDAR+depth)"].append(metrics(model.predict({}, lg, dg), label))
    res["LiDAR only"].append(metrics(lg, label))
print(f"frames: {len(keys)}")
for name, rows in res.items():
    a = np.array(rows, dtype=float)
    print(f"{name:26s} cell3 {a[:,0].mean():.1%}  IoU* {a[:,1].mean():.1%}  missed obst. {a[:,2].sum()/max(1,a[:,3].sum()):.2%}  false alarm {a[:,4].sum()/max(1,a[:,5].sum()):.2%}")
