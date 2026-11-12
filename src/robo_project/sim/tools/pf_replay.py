#!/usr/bin/env python3
"""
Offline replay of the sketch particle filter on a recorded simulation run (fast tuning / testing).

  python3 src/robo_project/sim/tools/pf_replay.py BAG [key=value ...] [--sweep]

The bag needs /odom (true pose), /fusion/local_grid and /map (run_regression.sh records them).
Wheel odometry is generated from the true motion with the same error model as wheel_odom_sim.py
(trans_scale, rot_scale, trans_noise, rot_noise), so the filter sees what it sees live.
map_npy=FILE: use this occupied grid (ROS row order, /map size) instead of the sketch.
Filter settings as key=value: sigma_m, p_floor, effective_cells, n, ntt, nrr, nrt, ntr, min_trans, min_rot.
Prints the position / heading error of the estimate (corrected odometry, as published on
/localization/odom) at every true pose, and of the wheel odometry alone.
"""
import math
import sys

import numpy as np
import rosbag2_py
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.serialization import deserialize_message

from robo_project.localization.sketch_pf_node import PFCore, compose, grid_cells, inverse, limit_step, yaw_of


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    odom, grids, the_map = [], [], None
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic == "/odom":
            m = deserialize_message(data, Odometry)
            odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                         (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))))
        elif topic == "/fusion/local_grid":
            m = deserialize_message(data, OccupancyGrid)
            grids.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.info, np.asarray(m.data, np.int16)))
        elif topic == "/map" and the_map is None:
            the_map = deserialize_message(data, OccupancyGrid)
    odom.sort(key=lambda e: e[0])
    grids.sort(key=lambda e: e[0])
    return odom, grids, the_map


def wheel_odometry(odom, trans_scale=1.03, rot_scale=1.08, trans_noise=0.02, rot_noise=0.03, seed=1):
    """Same model as sim/tools/wheel_odom_sim.py."""
    rng = np.random.default_rng(seed)
    out, pose, prev = [], [0.0, 0.0, 0.0], None
    for t, cur in odom:
        if prev is not None:
            d = compose(inverse(prev), cur)
            dist = math.hypot(d[0], d[1])
            fwd = d[0] * trans_scale + rng.normal(0, trans_noise * dist + 1e-6)
            lat = d[1] * trans_scale
            dth = d[2] * rot_scale + rng.normal(0, rot_noise * abs(d[2]) + 0.005 * dist + 1e-6)
            pose = list(compose(tuple(pose), (fwd, lat, dth)))
        prev = cur
        out.append((t, tuple(pose)))
    return out


def run(odom, grids, the_map, cfg):
    core = PFCore(n=int(cfg.get("n", 600)), sigma_m=cfg.get("sigma_m", 0.40), p_floor=cfg.get("p_floor", 0.10),
                  effective_cells=cfg.get("effective_cells", 5.0),
                  noise=(cfg.get("ntt", 0.05), cfg.get("nrr", 0.03), cfg.get("nrt", 0.05), cfg.get("ntr", 0.02)),
                  seed=int(cfg.get("seed", 0)))
    info = the_map.info
    occ = np.asarray(the_map.data, np.int16).reshape(info.height, info.width) != 0
    if cfg.get("map_npy"):                       # replace the sketch, e.g. by the true floor plan (same geometry)
        occ = np.load(cfg["map_npy"])
    core.set_map(occ, (info.origin.position.x, info.origin.position.y), info.resolution)
    wheel = wheel_odometry(odom, cfg.get("trans_scale", 1.03), cfg.get("rot_scale", 1.08),
                           cfg.get("trans_noise", 0.02), cfg.get("rot_noise", 0.03), int(cfg.get("odom_seed", 1)))
    wt = np.array([t for t, _ in wheel])
    core.reset(odom[0][1])
    corr = compose(odom[0][1], inverse(wheel[0][1]))
    last = wheel[0][1]
    gi, errs, yerrs, jumps = 0, [], [], []
    for k, (t, w) in enumerate(wheel):
        while gi < len(grids) and grids[gi][0] <= t:
            gt_, ginfo, gdata = grids[gi]
            gi += 1
            j = int(np.argmin(np.abs(wt - gt_)))
            if abs(wt[j] - gt_) > 0.15:
                continue
            o = wheel[j][1]
            d = compose(inverse(last), o)
            if math.hypot(d[0], d[1]) < cfg.get("min_trans", 0.05) and abs(d[2]) < cfg.get("min_rot", 0.05):
                continue
            fx, ly, is_occ = grid_cells(ginfo, gdata, int(cfg.get("stride", 2)))
            if len(fx) < 20:
                continue
            core.predict(last, o)
            last = o
            core.update(fx, ly, is_occ)
            target = compose(core.estimate(), inverse(o))
            dist, turn = math.hypot(d[0], d[1]), abs(d[2])
            new = limit_step(corr, target, o,
                             cfg.get("k_t", 0.10) * dist + cfg.get("k_tr", 0.05) * turn + 0.001,
                             cfg.get("k_r", 0.02) * turn + cfg.get("k_rt", 0.005) * dist + math.radians(0.05)) \
                if cfg.get("k_t", 0.10) > 0 else target
            a_, b_ = compose(corr, o), compose(new, o)          # jump of the published robot pose
            jumps.append(math.hypot(b_[0] - a_[0], b_[1] - a_[1]))
            corr = new
        est = compose(corr, w)
        g = odom[k][1]
        errs.append(math.hypot(est[0] - g[0], est[1] - g[1]))
        yerrs.append(abs(math.degrees(math.atan2(math.sin(est[2] - g[2]), math.cos(est[2] - g[2])))))
    run.jumps = np.array(jumps) if jumps else np.zeros(1)
    return np.array(errs), np.array(yerrs), wheel


def dead_reckoning_error(odom, wheel):
    corr = compose(odom[0][1], inverse(wheel[0][1]))
    e = [math.hypot(compose(corr, w)[0] - g[0], compose(corr, w)[1] - g[1]) for (_, g), (_, w) in zip(odom, wheel)]
    return np.array(e)


def fmt(e):
    return f"mean {e.mean():.2f}  p95 {np.percentile(e, 95):.2f}  max {e.max():.2f}"


def main():
    bag = sys.argv[1]
    cfg = {}
    for a in sys.argv[2:]:
        if "=" in a:
            k, v = a.split("=", 1)
            cfg[k] = v if k == "map_npy" else float(v)
    odom, grids, the_map = read(bag)
    path = sum(math.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1]) for a, b in zip(odom, odom[1:]))
    print(f"{len(odom)} poses, {len(grids)} ML grids, {odom[-1][0] - odom[0][0]:.0f} s, {path:.1f} m driven")
    sweep = [{}] if "--sweep" not in sys.argv else [
        dict(sigma_m=s, effective_cells=e, ntt=nt, nrr=nr)
        for s in (0.25, 0.4, 0.6) for e in (10, 25, 60) for nt, nr in ((0.10, 0.10), (0.15, 0.25))]
    for extra in sweep:
        c = dict(cfg, **extra)
        e, y, wheel = run(odom, grids, the_map, c)
        j = run.jumps
        print(f"{extra or cfg}:  position {fmt(e)} m | heading mean {y.mean():.1f} max {y.max():.1f} deg | "
              f"pose jumps: max {j.max():.3f} m, {np.sum(j > 0.05)} over 5 cm", flush=True)
    print(f"wheel odometry only: position {fmt(dead_reckoning_error(odom, wheel))} m")


if __name__ == "__main__":
    main()
