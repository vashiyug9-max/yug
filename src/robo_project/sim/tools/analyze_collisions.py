#!/usr/bin/env python3
"""
Collision analysis of a recorded run (rosbag2, sqlite3).

Two independent checks:
 1. SIM CONTACT  - Habitat's own collision: the bridge moves the agent with pathfinder.try_step,
    which clips motion at the navmesh edge. /odom carries the commanded velocity actually applied
    (twist.linear.x), so commanded vs. actual displacement reveals every contact / wall slide.
 2. BODY OVERLAP - Habitat models the robot as a 0.10 m-radius cylinder (navmesh agent_radius).
    The real VineBot footprint (0.86 x 0.53 m, base_footprint at the middle axle) is checked
    against the scene's obstacles = navigable area grown back by agent_radius.

usage: analyze_collisions.py <bag_dir> <navmesh> <out_dir>
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

RES = 0.02                                   # m/px of the top-down view
FOOT_X = (-0.451, 0.409)                     # VineBot body in base_footprint (from the URDF mesh bounds)
FOOT_Y = (-0.264, 0.266)
WIN_S = 0.5                                  # sim-contact window
MIN_CMD_M = 0.03                             # ...with at least this much commanded forward motion
CONTACT_RATIO = 0.5                          # ...and less than this fraction actually driven
PEN_TOL = 0.02                               # body overlap below this (m) = map discretisation noise


def read_bag(path, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = {t: [] for t in topics}
    while r.has_next():
        topic, data, t_ns = r.read_next()
        out[topic].append((t_ns * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def group(times, flags, gap=1.0):
    ev, cur = [], None
    for t, f in zip(times, flags):
        if f:
            if cur and t - cur[1] <= gap:
                cur[1] = t
            else:
                cur = [t, t]
                ev.append(cur)
    return ev


def main():
    bag, navmesh, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_bag(bag, ["/odom", "/blocked_point", "/goal_pose"])
    odom = data["/odom"]
    if len(odom) < 10:
        sys.exit("not enough /odom in the bag")
    t0 = odom[0][0]
    T = np.array([t for t, _ in odom]) - t0
    X = np.array([m.pose.pose.position.x for _, m in odom])
    Y = np.array([m.pose.pose.position.y for _, m in odom])
    YAW = np.array([yaw_of(m.pose.pose.orientation) for _, m in odom])
    V = np.array([m.twist.twist.linear.x for _, m in odom])

    # ── 1. sim contact ──────────────────────────────────────────────────
    dt = np.diff(T, prepend=T[0])
    cmd = np.abs(V) * np.clip(dt, 0, 0.2)
    dx, dy = np.diff(X, prepend=X[0]), np.diff(Y, prepend=Y[0])
    # motion actually driven in the commanded direction (forward or reverse)
    act = np.maximum(0.0, np.sign(V) * (dx * np.cos(YAW) + dy * np.sin(YAW)))
    csum, asum = np.cumsum(cmd), np.cumsum(act)
    j = np.searchsorted(T, T - WIN_S)
    wc, wa = csum - csum[j], asum - asum[j]
    contact = (wc > MIN_CMD_M) & (wa < CONTACT_RATIO * wc)
    sim_events = group(T, contact)

    # ── 2. body overlap with the scene ──────────────────────────────────
    import habitat_sim
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh)
    radius = float(pf.nav_mesh_settings.agent_radius)
    start = np.array([X[0], 0.0, -Y[0]], dtype=np.float32)
    lower = pf.get_bounds()[0]
    start[1] = float(lower[1]) + 0.5
    floor_y = float(pf.snap_point(start)[1])
    nav = np.asarray(pf.get_topdown_view(RES, floor_y)).astype(np.uint8)    # [z, x]
    x_min, z_min = float(lower[0]), float(lower[2])
    r_px = max(1, int(round(radius / RES)))
    free_real = cv2.dilate(nav, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1,) * 2))
    obst = (free_real == 0).astype(np.uint8)
    pen_map = cv2.distanceTransform(obst, cv2.DIST_L2, 5) * RES             # depth inside an obstacle
    clr_map = cv2.distanceTransform(free_real, cv2.DIST_L2, 5) * RES        # clearance in free space

    fx, fy = np.meshgrid(np.arange(FOOT_X[0], FOOT_X[1] + 1e-9, RES), np.arange(FOOT_Y[0], FOOT_Y[1] + 1e-9, RES))
    fx, fy = fx.ravel(), fy.ravel()
    edge = (np.isclose(fx, fx.min()) | np.isclose(fx, fx.max()) | np.isclose(fy, fy.min()) | np.isclose(fy, fy.max()))

    def to_px(wx, wy):
        c = ((wx - x_min) / RES).astype(int)
        r = ((-wy - z_min) / RES).astype(int)
        ok = (r >= 0) & (c >= 0) & (r < nav.shape[0]) & (c < nav.shape[1])
        return r.clip(0, nav.shape[0] - 1), c.clip(0, nav.shape[1] - 1), ok

    step = max(1, len(T) // 4000)                                            # <= ~4000 poses checked
    idx = np.arange(0, len(T), step)
    pen = np.zeros(len(idx))
    clr = np.zeros(len(idx))
    for k, i in enumerate(idx):
        c, s = math.cos(YAW[i]), math.sin(YAW[i])
        wx = X[i] + c * fx - s * fy
        wy = Y[i] + s * fx + c * fy
        r, cc, ok = to_px(wx, wy)
        p = np.where(ok, pen_map[r, cc], 1.0)
        pen[k] = p.max()
        e = edge & ok
        clr[k] = clr_map[r[e], cc[e]].min() if e.any() else 0.0
    body_hit = pen > PEN_TOL
    body_events = group(T[idx], body_hit)

    # ── report ──────────────────────────────────────────────────────────
    dist = float(np.hypot(np.diff(X), np.diff(Y)).sum())
    lines = [f"bag: {bag}", f"duration {T[-1]:.1f} s, distance driven {dist:.2f} m, "
             f"goals clicked {len(data['/goal_pose'])}, /blocked_point reports {len(data['/blocked_point'])}",
             f"navmesh agent_radius = {radius:.2f} m (Habitat's collision model of the robot)", "",
             f"1. SIM CONTACT (Habitat clipped the motion: driven < {CONTACT_RATIO:.0%} of commanded over {WIN_S} s)",
             f"   events: {len(sim_events)}"]
    for a, b in sim_events:
        i = np.searchsorted(T, a)
        lines.append(f"   t={a:7.1f}-{b:7.1f} s  at ({X[i]:.2f},{Y[i]:.2f})")
    lines += ["", f"2. BODY OVERLAP (0.86 x 0.53 m footprint inside scene obstacles by > {PEN_TOL * 100:.0f} cm)",
              f"   events: {len(body_events)}, time overlapping: {body_hit.mean() * 100:.1f}% of the run, "
              f"max depth {pen.max() * 100:.1f} cm",
              f"   min body clearance when not overlapping: "
              f"{(clr[~body_hit].min() * 100 if (~body_hit).any() else float('nan')):.1f} cm"]
    for a, b in body_events:
        kk = np.searchsorted(T[idx], a)
        sel = (T[idx] >= a) & (T[idx] <= b)
        i = idx[kk]
        lines.append(f"   t={a:7.1f}-{b:7.1f} s  at ({X[i]:.2f},{Y[i]:.2f})  max depth {pen[sel].max() * 100:.1f} cm")
    report = "\n".join(lines)
    (out_dir / "collision_report.txt").write_text(report + "\n")
    print(report)

    # plot: scene, trajectory, events
    img = np.where(free_real[..., None] > 0, 235, 70).astype(np.uint8).repeat(3, axis=2)
    img[(nav == 0) & (free_real > 0)] = (200, 200, 255)                     # navmesh margin
    r, c, _ = to_px(X, Y)
    for k in range(1, len(r)):
        cv2.line(img, (c[k - 1], r[k - 1]), (c[k], r[k]), (200, 120, 0), 2)
    for k in np.flatnonzero(body_hit):
        cv2.circle(img, (c[idx[k]], r[idx[k]]), 4, (0, 140, 255), -1)        # orange: body overlap
    for k in np.flatnonzero(contact):
        cv2.circle(img, (c[k], r[k]), 5, (0, 0, 255), -1)                    # red: sim contact
    for _, m in data["/blocked_point"]:
        rr, cc, _ = to_px(np.array([m.point.x]), np.array([m.point.y]))
        cv2.drawMarker(img, (int(cc[0]), int(rr[0])), (160, 0, 160), cv2.MARKER_TILTED_CROSS, 10, 2)
    cv2.circle(img, (c[0], r[0]), 7, (0, 160, 0), -1)
    cv2.circle(img, (c[-1], r[-1]), 7, (0, 0, 0), 2)
    cv2.imwrite(str(out_dir / "collision_map.png"), cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    print(f"\nwrote {out_dir / 'collision_report.txt'} and {out_dir / 'collision_map.png'}")


if __name__ == "__main__":
    main()
