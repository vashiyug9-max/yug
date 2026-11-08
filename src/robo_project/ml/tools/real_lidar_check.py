#!/usr/bin/env python3
"""
Check a real Livox recording (LVX2 file from Livox Viewer 2) against our LiDAR pipeline.

  python3 src/robo_project/ml/tools/real_lidar_check.py ~/Downloads/lidar_data results/real_lidar

The points go through exactly what fusion_obstacle_node does with /livox/lidar on the real robot:
rig mounting (config/vinebot_rig.yaml: height, yaw) -> base_footprint, robot self-returns removed,
FusionInputs.lidar_grid (obstacle band, 3 m grid of the model). Writes
  report.txt   self-returns, the beam pattern per direction (rig lidar.min_elevation_deg for the
               simulation), lowest visible obstacle height vs distance
  frames.png   top view of the points and the model's LiDAR input grid at a few moments
No ROS needed. An LVX2 file has no extension by default: the name does not matter.
"""

import struct
import sys
from pathlib import Path

import numpy as np

from robo_project.common.rig import drop_self_hits, lidar_to_base, load_rig
from robo_project.ml.fusion.fusion_model import FusionInputs

POINT = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4"), ("refl", "u1"), ("tag", "u1")])   # data type 1, mm
FRAME_S = 0.05


def read_lvx2(path):
    """-> (memmap bytes, list of (start, end) byte ranges of the frames, header info)."""
    b = np.memmap(path, dtype=np.uint8, mode="r")
    if bytes(b[:10]) != b"livox_tech":
        raise ValueError(f"{path} is not an LVX2 file (no 'livox_tech' signature)")
    dur_ms, ndev = struct.unpack_from("<IB", b, 24)
    devs = []
    off = 29
    for _ in range(ndev):
        sn = bytes(b[off:off + 16]).rstrip(b"\0").decode(errors="replace")
        rpy_xyz = struct.unpack_from("<6f", b, off + 39)
        devs.append(dict(serial=sn, extrinsic_rpy_deg=rpy_xyz[:3], extrinsic_xyz=rpy_xyz[3:]))
        off += 63
    frames = []
    while off + 24 <= len(b):
        _, nxt, _ = struct.unpack_from("<QQQ", b, off)
        if nxt <= off or nxt > len(b):
            break
        frames.append((off, nxt))
        off = nxt
    return b, frames, dict(frame_ms=dur_ms, devices=devs)


def frame_points(b, fr) -> np.ndarray:
    """Points of one frame in the sensor frame (m), zero / invalid returns dropped."""
    off, nxt = fr
    p, chunks = off + 24, []
    while p + 27 <= nxt:
        dtype, length = struct.unpack_from("<BI", b, p + 17)
        if dtype == 1:
            chunks.append(np.frombuffer(bytes(b[p + 27:p + 27 + length]), POINT))
        p += 27 + length
    if not chunks:
        return np.zeros((0, 3))
    a = np.concatenate(chunks)
    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64) / 1000.0
    return xyz[np.linalg.norm(xyz, axis=1) > 0.05]


def window(b, frames, t, n=4):
    i = min(int(t / FRAME_S), len(frames) - n)
    return np.concatenate([frame_points(b, frames[j]) for j in range(i, i + n)])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "results/real_lidar")
    out.mkdir(parents=True, exist_ok=True)
    rig = load_rig("")
    b, frames, info = read_lvx2(src)
    dur = len(frames) * FRAME_S
    lines = [f"file {src}: {len(frames)} frames, {dur:.1f} s, devices {info['devices']}",
             f"rig {rig['path']}: lidar height {rig['lidar']['height']} m, yaw {rig['lidar'].get('yaw_deg', 0.0)} deg"]

    # self-returns: points inside the robot body (rig mounting -> base_footprint)
    sample_t = np.linspace(2, dur - 2, 20)
    n_raw = n_self = 0
    for t in sample_t:
        pts = lidar_to_base(window(b, frames, t), rig)
        n_raw += len(pts)
        n_self += len(pts) - len(drop_self_hits(pts, rig))
    lines.append(f"self-returns removed (inside the {rig['robot']['length']} x {rig['robot']['width']} m body): "
                 f"{100 * n_self / max(n_raw, 1):.2f}% of the points")
    # mounting check: the robot's deck / e-stop are BEHIND the front LiDAR, so the persistent returns
    # close to the sensor must lie behind it once the rig yaw is applied. Otherwise the yaw is wrong
    # (e.g. the Livox driver already applies the 90 deg extrinsic -> set lidar.yaw_deg to 0).
    lx = float(rig["lidar"].get("x", 0.0)); cnt = {}
    for t in sample_t:
        p = lidar_to_base(window(b, frames, t, 2), rig)
        near = p[np.hypot(p[:, 0] - lx, p[:, 1]) < 0.8]
        for key in set(map(tuple, np.floor(near / 0.05).astype(int))):
            cnt[key] = cnt.get(key, 0) + 1
    pers = np.array([np.array(k) * 0.05 + 0.025 for k, c in cnt.items() if c >= 0.9 * len(sample_t)])
    if len(pers):
        dx = float(np.median(pers[:, 0] - lx)); dy = float(np.median(pers[:, 1]))
        ok = dx < -0.1 and abs(dy) < abs(dx)
        lines.append(f"mounting yaw check: robot's own returns at dx {dx:+.2f} m, dy {dy:+.2f} m from the LiDAR -> "
                     + ("behind it, as on the robot: yaw OK" if ok else
                        "NOT behind it: lidar.yaw_deg is probably wrong (driver already rotates the points?)"))
    else:
        lines.append("mounting yaw check: no persistent self-returns found (cannot check)")
    # (the floor height cannot be measured reliably here: a glossy floor returns almost nothing at
    #  grazing angles and mirrors points below the floor)

    # beam pattern: lowest elevation of returns beyond the robot body, per 10-deg azimuth sector
    # (robot axes, origin at the LiDAR) -> rig lidar.min_elevation_deg used by the simulation
    lid = rig["lidar"]
    P = np.concatenate([window(b, frames, t, 4) for t in np.linspace(3, dur - 3, 60)])
    yaw = np.radians(float(lid.get("yaw_deg", 0.0)))
    c, s_ = np.cos(yaw), np.sin(yaw)
    q = np.c_[c * P[:, 0] - s_ * P[:, 1], s_ * P[:, 0] + c * P[:, 1], P[:, 2]]
    r = np.hypot(q[:, 0], q[:, 1])
    az = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
    el = np.degrees(np.arctan2(q[:, 2], r))
    prof = []
    for lo in range(-180, 180, 10):
        m = (r > 1.3) & (az >= lo) & (az < lo + 10)
        prof.append(round(float(np.percentile(el[m], 0.5)), 1) if m.sum() > 200 else float("nan"))
    lines.append("beam pattern (lowest elevation per 10-deg sector from -180 deg, 0 = forward) - "
                 "rig lidar.min_elevation_deg:")
    lines.append("   " + str(prof))
    h = float(lid["height"])
    front = np.nanmax([prof[17], prof[18]])                 # sectors -10..+10 deg
    lines.append(f"straight ahead the lowest beam is {front:.1f} deg -> the LiDAR sees obstacles only ABOVE:")
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        lines.append(f"   {d:.1f} m ahead of the LiDAR: {max(0.0, h + d * np.tan(np.radians(front))) * 100:4.0f} cm")
    lines.append("   lower obstacles close to the robot are only seen by the cameras / front depth.")
    report = "\n".join(lines)
    (out / "report.txt").write_text(report + "\n")
    print(report)

    # pictures: points (top view, obstacle band in colour) + the model's LiDAR input grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inputs = FusionInputs(rig, 3.0)
    band = rig.get("obstacle_band", [0.10, 1.80])
    ts = np.linspace(5, dur - 5, 4)
    fig, ax = plt.subplots(2, len(ts), figsize=(4.2 * len(ts), 8.4))
    L, W = rig["robot"]["length"], rig["robot"]["width"]
    for k, t in enumerate(ts):
        p = drop_self_hits(lidar_to_base(window(b, frames, t), rig), rig)
        inb = (p[:, 2] >= band[0]) & (p[:, 2] <= band[1])
        a = ax[0, k]
        a.scatter(-p[~inb, 1], p[~inb, 0], s=0.2, c="0.75")
        a.scatter(-p[inb, 1], p[inb, 0], s=0.4, c=p[inb, 2], cmap="viridis", vmin=band[0], vmax=band[1])
        a.add_patch(plt.Rectangle((-W / 2, -L / 2), W, L, fill=False, ec="r", lw=1.5))
        a.add_patch(plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, ec="b", ls="--"))
        a.set_xlim(-5, 5); a.set_ylim(-5, 5); a.set_aspect("equal")
        a.set_title(f"t = {t:.0f} s  (forward = up)")
        g = inputs.lidar_grid(p)
        ax[1, k].imshow(g, cmap="gray", vmin=0, vmax=1)
        ax[1, k].set_title("model LiDAR input, 3 m (white = free)")
        ax[1, k].axis("off")
    ax[0, 0].set_ylabel("m  (grey: floor / ceiling, colour: obstacle band)")
    fig.tight_layout()
    fig.savefig(out / "frames.png", dpi=90)
    print(f"wrote {out}/report.txt and {out}/frames.png")


if __name__ == "__main__":
    main()
