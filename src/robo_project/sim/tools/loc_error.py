#!/usr/bin/env python3
"""
Localization accuracy from a recording (simulation): estimated pose vs the true pose.

  python3 src/robo_project/sim/tools/loc_error.py BAG [OUT_DIR]

Needs /odom (true pose from Habitat), /localization/odom (estimate) and optionally /wheel_odom in the bag.
Matches messages by timestamp (the estimate carries the odometry stamp). Prints position / heading error
of the estimate, and of raw wheel odometry aligned at the start (= what you get without localization).
Writes OUT_DIR/localization_error.txt and .png.
"""
import math
import sys
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    want = {"/odom", "/localization/odom", "/wheel_odom"}
    out = {k: {} for k in want}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic in want:
            m = deserialize_message(data, Odometry)
            t = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            out[topic][t] = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
    return out


def stats(err):
    return f"mean {np.mean(err):.3f}  median {np.median(err):.3f}  p95 {np.percentile(err, 95):.3f}  max {np.max(err):.3f}"


def main():
    bag = sys.argv[1]
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else Path(bag).parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    d = read(bag)
    gt, est, wh = d["/odom"], d["/localization/odom"], d["/wheel_odom"]
    common = sorted(set(gt) & set(est))
    if not common:
        sys.exit("no matching /odom and /localization/odom stamps in the bag")
    G = np.array([gt[t] for t in common]); E = np.array([est[t] for t in common])
    pe = np.hypot(G[:, 0] - E[:, 0], G[:, 1] - E[:, 1])
    ye = np.degrees(np.abs(np.angle(np.exp(1j * (G[:, 2] - E[:, 2])))))
    T = (np.array(common) - common[0]) / 1e9
    path = np.sum(np.hypot(np.diff(G[:, 0]), np.diff(G[:, 1])))
    lines = [f"bag {bag}: {len(common)} poses over {T[-1]:.0f} s, robot drove {path:.1f} m",
             f"LOCALIZATION   position error (m): {stats(pe)}",
             f"               heading error (deg): {stats(ye)}"]
    wc = sorted(set(gt) & set(wh))
    if wc:
        G2 = np.array([gt[t] for t in wc]); W = np.array([wh[t] for t in wc])
        # align wheel odometry to the true start pose (what a robot started at a known pose would have)
        a = G2[0, 2] - W[0, 2]; c, s = math.cos(a), math.sin(a)
        wx = G2[0, 0] + c * (W[:, 0] - W[0, 0]) - s * (W[:, 1] - W[0, 1])
        wy = G2[0, 1] + s * (W[:, 0] - W[0, 0]) + c * (W[:, 1] - W[0, 1])
        we = np.hypot(G2[:, 0] - wx, G2[:, 1] - wy)
        lines.append(f"WHEEL ODOMETRY only (no localization), position error (m): {stats(we)}")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "localization_error.txt").write_text(txt + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot(G[:, 0], G[:, 1], "k-", label="true")
        ax[0].plot(E[:, 0], E[:, 1], "g-", label="localization")
        if wc:
            ax[0].plot(wx, wy, "r--", label="wheel odometry only")
        ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("paths (map frame)")
        ax[1].plot(T, pe, "g-", label="localization")
        if wc:
            ax[1].plot((np.array(wc) - wc[0]) / 1e9, we, "r--", label="wheel odometry only")
        ax[1].set_xlabel("s"); ax[1].set_ylabel("position error (m)"); ax[1].legend(); ax[1].grid()
        fig.tight_layout(); fig.savefig(out_dir / "localization_error.png", dpi=90)
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
