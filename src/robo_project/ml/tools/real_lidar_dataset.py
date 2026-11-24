#!/usr/bin/env python3
"""
Training samples from a REAL Livox Mid-360 recording (LVX2 from Livox Viewer 2) for FusionOccNet.

  python3 -m robo_project.ml.tools.real_lidar_dataset ~/Downloads/lidar_data data/real_lidar

The recording has no cameras, no odometry and no ground truth, so the targets are built from the
recording itself:
  1. points of every 0.25 s window -> base_footprint (rig mounting), robot self-returns removed
  2. scan matching (2D ICP of the obstacle-band points against the map built so far) -> robot path
  3. occupancy map of the whole drive: LiDAR rays free up to their hit, the hit occupied (log-odds)
  4. per window: the scan in the training format (nearest obstacle-band return per bearing, from the
     LiDAR position) and a 6 x 6 m label cut out of that map around the robot, robot-forward up:
        1 = free, 0 = occupied, -1 = unknown (never covered by the LiDAR)
Writes OUT/train (first 80 % of the drive) and OUT/test (last 20 %, never trained on) as .npz files with
fields scan, scan_origin, label (int8 with -1), real (=1); no camera / depth fields (the training masks them
as missing). OUT/map.png shows the built map and the path, OUT/report.txt the numbers.

Limits: the LiDAR sees obstacles only inside its height band (it misses low objects close to the robot),
so these targets teach the LiDAR part only; scan matching drifts slowly over a long drive.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from robo_project.common.rig import drop_self_hits, lidar_to_base, load_rig
from robo_project.ml.tools.real_lidar_check import FRAME_S, frame_points, read_lvx2
from robo_project.sim.lidar_sim import points_to_scan

WIN = 5              # frames per window (5 x 0.05 s = 0.25 s)
BAND = (0.15, 1.80)  # m above the floor counted as obstacles (0.15: the mount is tilted ~2 deg, far floor returns)
MAP_RES = 0.05
MAX_RANGE = 8.0      # m used for the map
LABEL_N, LABEL_RES = 120, 0.05


def se2(x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1.0]])


def transform(T, p2):
    return p2 @ T[:2, :2].T + T[:2, 2]


def voxel(p2, res=0.05):
    if len(p2) == 0:
        return p2
    k = np.floor(p2 / res).astype(np.int64)
    _, idx = np.unique(k[:, 0] * 1_000_003 + k[:, 1], return_index=True)
    return p2[idx]


def icp(src, tree, dst, T0, iters=30):
    """Point-to-point 2D ICP of src (robot frame) onto dst (world) from the guess T0 -> (T, inlier share)."""
    T = T0.copy()
    for it in range(iters):
        p = transform(T, src)
        d, j = tree.query(p, distance_upper_bound=0.6 if it < 10 else 0.25)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            return T, 0.0
        a, b = p[ok], dst[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        H = (a - ma).T @ (b - mb)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1] *= -1
            R = Vt.T @ U.T
        t = mb - R @ ma
        D = np.eye(3)
        D[:2, :2], D[:2, 2] = R, t
        T = D @ T
        if np.hypot(*t) < 1e-4 and abs(np.arctan2(R[1, 0], R[0, 0])) < 1e-5:
            break
    d, _ = tree.query(transform(T, src), distance_upper_bound=0.15)
    return T, float(np.isfinite(d).mean())


def main():
    src, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
    rig = load_rig("")
    lid_xy = (float(rig["lidar"].get("x", 0.0)), float(rig["lidar"].get("y", 0.0)))
    b, frames, info = read_lvx2(src)
    n_win = len(frames) // WIN
    print(f"{src}: {len(frames)} frames -> {n_win} windows of {WIN * FRAME_S:.2f} s", flush=True)

    # 1 + 2: windows -> base frame, scan matching against the growing map
    scans, poses, quality = [], [], []
    map_pts = np.zeros((0, 2))
    tree = None
    T = np.eye(3)
    vel = np.eye(3)
    for w in range(n_win):
        pts = np.concatenate([frame_points(b, frames[j]) for j in range(w * WIN, (w + 1) * WIN)])
        pts = drop_self_hits(lidar_to_base(pts, rig), rig)
        z = pts[:, 2]
        band = pts[(z >= BAND[0]) & (z <= BAND[1]) & (np.hypot(pts[:, 0] - lid_xy[0], pts[:, 1] - lid_xy[1]) < MAX_RANGE)]
        p2 = voxel(band[:, :2])
        scans.append(points_to_scan(pts, lid_xy, BAND, 720, MAX_RANGE))
        if tree is None:
            q = 1.0
        else:
            T_new, q = icp(p2, tree, map_pts, T @ vel)
            if q > 0.5:
                vel = np.linalg.inv(T) @ T_new
                T = T_new
            else:                                           # bad match: keep the motion guess
                T = T @ vel
        poses.append(T.copy())
        quality.append(q)
        if tree is None or w % 4 == 0:                      # add a keyframe every second
            map_pts = voxel(np.vstack([map_pts, transform(T, p2)]))
            tree = cKDTree(map_pts)
        if w % 100 == 0:
            print(f"  window {w}/{n_win}: pose ({T[0, 2]:.2f}, {T[1, 2]:.2f}, "
                  f"{np.degrees(np.arctan2(T[1, 0], T[0, 0])):.0f} deg), match {q:.0%}, map {len(map_pts)} pts", flush=True)

    # 3: occupancy map from the rays (bearing scans) of every window
    xy = np.array([[P[0, 2], P[1, 2]] for P in poses])
    lo_xy = xy.min(0) - MAX_RANGE - 1
    hi_xy = xy.max(0) + MAX_RANGE + 1
    W, H = (np.ceil((hi_xy - lo_xy) / MAP_RES)).astype(int)
    logodds = np.zeros((H, W), np.float32)
    ang = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    steps = np.arange(0.0, MAX_RANGE, MAP_RES * 0.7)
    for P, sc in zip(poses, scans):
        th = np.arctan2(P[1, 0], P[0, 0])
        ox, oy = transform(P, np.array([lid_xy]))[0]
        hit = np.isfinite(sc)
        r = np.where(hit, sc, MAX_RANGE)
        a = ang + th
        free_r = steps[None, :] < (r[:, None] - 0.1)
        fx = ox + steps[None, :] * np.cos(a)[:, None]
        fy = oy + steps[None, :] * np.sin(a)[:, None]
        ci = ((fx - lo_xy[0]) / MAP_RES).astype(int)[free_r]
        ri = ((fy - lo_xy[1]) / MAP_RES).astype(int)[free_r]
        cell = np.unique(ri * W + ci)
        logodds.ravel()[cell] -= 0.4
        hx = ox + r[hit] * np.cos(a[hit])
        hy = oy + r[hit] * np.sin(a[hit])
        cell = np.unique(((hy - lo_xy[1]) / MAP_RES).astype(int) * W + ((hx - lo_xy[0]) / MAP_RES).astype(int))
        logodds.ravel()[cell] += 0.9
    world = np.full((H, W), -1, np.int8)               # -1 unknown, 1 free, 0 occupied
    world[logodds < -1.0] = 1
    world[logodds > 1.0] = 0

    # 4: one sample per window
    n_train = int(0.8 * n_win)
    for d in ("train", "test"):
        (out / d).mkdir(parents=True, exist_ok=True)
    coords = (np.arange(LABEL_N) + 0.5) * LABEL_RES - LABEL_N * LABEL_RES / 2
    fx = -coords[:, None] * np.ones((1, LABEL_N))       # row 0 = front
    ly = -coords[None, :] * np.ones((LABEL_N, 1))       # col 0 = left
    body = (np.abs(fx) <= 0.575) & (np.abs(ly) <= 0.30)
    stats = []
    for w, (P, sc) in enumerate(zip(poses, scans)):
        th = np.arctan2(P[1, 0], P[0, 0])
        wx = P[0, 2] + fx * np.cos(th) - ly * np.sin(th)
        wy = P[1, 2] + fx * np.sin(th) + ly * np.cos(th)
        ci = ((wx - lo_xy[0]) / MAP_RES).astype(int)
        ri = ((wy - lo_xy[1]) / MAP_RES).astype(int)
        inside = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        label = np.full((LABEL_N, LABEL_N), -1, np.int8)
        label[inside] = world[ri[inside], ci[inside]]
        label[body] = 1                                  # the robot stands on free floor
        stats.append(((label == 1).mean(), (label == 0).mean(), (label == -1).mean()))
        np.savez_compressed(out / ("train" if w < n_train else "test") / f"real_{w:05d}.npz",
                            scan=sc.astype(np.float32), scan_origin=np.array(lid_xy, np.float32),
                            label=label, real=np.int8(1))

    # map picture + report
    import cv2
    img = np.full((H, W, 3), 128, np.uint8)
    img[world == 1] = (255, 255, 255)
    img[world == 0] = (0, 0, 0)
    for i in range(1, len(xy)):
        p0 = (int((xy[i - 1, 0] - lo_xy[0]) / MAP_RES), int((xy[i - 1, 1] - lo_xy[1]) / MAP_RES))
        p1 = (int((xy[i, 0] - lo_xy[0]) / MAP_RES), int((xy[i, 1] - lo_xy[1]) / MAP_RES))
        cv2.line(img, p0, p1, (0, 0, 255) if i < n_train else (0, 160, 0), 2)
    cv2.imwrite(str(out / "map.png"), img[::-1])
    st = np.array(stats)
    q = np.array(quality)
    dist = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    rep = (f"{src}: {n_win} windows ({len(frames) * FRAME_S:.0f} s), path {dist:.1f} m\n"
           f"scan matching: median match {np.median(q):.0%}, windows below 50 %: {(q < 0.5).sum()}\n"
           f"map {W} x {H} cells at {MAP_RES} m: free {np.mean(world == 1):.0%}, occupied {np.mean(world == 0):.0%}\n"
           f"labels (mean): free {st[:, 0].mean():.0%}, occupied {st[:, 1].mean():.0%}, unknown {st[:, 2].mean():.0%}\n"
           f"samples: {n_train} train, {n_win - n_train} test -> {out}\n")
    (out / "report.txt").write_text(rep)
    print(rep, flush=True)


if __name__ == "__main__":
    main()
