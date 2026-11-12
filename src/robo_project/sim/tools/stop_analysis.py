#!/usr/bin/env python3
"""
How smoothly the robot drives: counts the stops / pauses in a recorded run.

  /usr/bin/python3 src/robo_project/sim/tools/stop_analysis.py BAG [--detail]
  --detail: every stop with its time, duration, what the robot did, time since the last NEW path and the
            turn angle of the path corner nearest to the robot (localized pose)

From /cmd_vel (what the runner commands) between each /goal_pose and the moment the robot is parked:
  stops        the forward speed drops to 0 for at least 0.3 s while the robot is still on its way
               - turn: it turns on the spot instead of driving   - halt: it stands completely still
  paused       share of the driving time without forward speed
  speed        average speed over the ground (true /odom)
"""
import math
import sys

import numpy as np
import rosbag2_py
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.serialization import deserialize_message

MIN_STOP_S = 0.3


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    cmd, goals, odom = [], [], []
    while r.has_next():
        topic, data, t = r.read_next()
        t *= 1e-9
        if topic == "/cmd_vel":
            m = deserialize_message(data, Twist)
            cmd.append((t, m.linear.x, m.angular.z))
        elif topic == "/goal_pose":
            goals.append(t)
        elif topic == "/odom":
            m = deserialize_message(data, Odometry)
            odom.append((t, m.pose.pose.position.x, m.pose.pose.position.y))
    return np.array(cmd), goals, np.array(odom)


def detail(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    cmd, paths, pose, goals, prev = [], [], [], [], None
    loc_topic = None
    while r.has_next():
        topic, data, t = r.read_next()
        t *= 1e-9
        if topic == "/cmd_vel":
            m = deserialize_message(data, Twist)
            cmd.append((t, m.linear.x, m.angular.z))
        elif topic == "/planned_path":
            m = deserialize_message(data, Path)
            pts = [(p.pose.position.x, p.pose.position.y) for p in m.poses]
            if pts != prev:
                paths.append((t, pts))
                prev = pts
        elif topic == "/goal_pose":
            goals.append(t)
        elif topic in ("/localization/odom", "/odom") and loc_topic in (None, topic):
            if topic == "/localization/odom" and loc_topic is None and pose:
                pose = []                                   # prefer the localized pose (what the runner uses)
            m = deserialize_message(data, Odometry)
            pose.append((t, m.pose.pose.position.x, m.pose.pose.position.y))
            if topic == "/localization/odom":
                loc_topic = topic
    cmd, pose = np.array(cmd), np.array(pose)
    pt = np.array([p[0] for p in paths])
    t0 = cmd[0, 0]
    print("   t     dur  what            new path   corner  (robot -> nearest path corner)")
    i, moving = 0, False
    while i < len(cmd):
        if abs(cmd[i, 1]) > 0.01:
            moving = True
            i += 1
            continue
        j = i
        while j < len(cmd) and abs(cmd[j, 1]) <= 0.01:
            j += 1
        dur = cmd[min(j, len(cmd) - 1), 0] - cmd[i, 0]
        if moving and dur >= MIN_STOP_S and j < len(cmd):
            t = cmd[i, 0]
            turn = float(np.sum(cmd[i:j - 1, 2] * np.diff(cmd[i:j, 0])))
            rev = np.any(cmd[i:j, 1] < -0.01)
            what = "goal start" if any(0 <= t - g < 3.0 for g in goals) else \
                "reversing" if rev else ("turn %4.0f deg" % math.degrees(turn) if abs(turn) > 0.05 else "standstill")
            k = np.searchsorted(pt, t) - 1
            since = t - pt[k] if k >= 0 else float("nan")
            corner = float("nan")
            if k >= 0 and len(pose):
                x, y = pose[min(np.searchsorted(pose[:, 0], t), len(pose) - 1), 1:]
                pts = paths[k][1]
                best = None
                for a, b, c in zip(pts, pts[1:], pts[2:]):
                    d = math.hypot(b[0] - x, b[1] - y)
                    if best is None or d < best[0]:
                        h1, h2 = math.atan2(b[1] - a[1], b[0] - a[0]), math.atan2(c[1] - b[1], c[0] - b[0])
                        best = (d, math.degrees(abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)))))
                if best:
                    corner = best[1]
            print(f"{t - t0:6.1f} {dur:5.1f}s  {what:14s}  {since:5.1f}s ago  {corner:4.0f} deg")
        i = j


def main():
    if "--detail" in sys.argv:
        detail(sys.argv[1])
        return
    cmd, goals, odom = read(sys.argv[1])
    goals = sorted(goals) + [float("inf")]
    tot = dict(turn=0, halt=0, drive=0.0, paused=0.0, dist=0.0, time=0.0)
    print("goal  time   dist  speed   stops(turn/halt)  paused")
    for gi in range(len(goals) - 1):
        seg = cmd[(cmd[:, 0] >= goals[gi]) & (cmd[:, 0] < goals[gi + 1])]
        mv = np.nonzero(np.abs(seg[:, 1]) > 0.01)[0] if len(seg) else []
        if len(mv) == 0:
            continue
        seg = seg[mv[0]:mv[-1] + 1]                       # from the first to the last forward command
        t0, t1 = seg[0, 0], seg[-1, 0]
        turns = halts = 0
        paused = 0.0
        i = 0
        while i < len(seg):
            if abs(seg[i, 1]) > 0.01:
                i += 1
                continue
            j = i
            while j < len(seg) and abs(seg[j, 1]) <= 0.01:
                j += 1
            dur = seg[min(j, len(seg) - 1), 0] - seg[i, 0]
            if dur >= MIN_STOP_S:
                paused += dur
                if np.any(np.abs(seg[i:j, 2]) > 0.05):
                    turns += 1
                else:
                    halts += 1
            i = j
        o = odom[(odom[:, 0] >= t0) & (odom[:, 0] <= t1)]
        dist = float(np.sum(np.hypot(np.diff(o[:, 1]), np.diff(o[:, 2])))) if len(o) > 1 else 0.0
        T = t1 - t0
        print(f"{gi:3d}  {T:5.1f}s {dist:5.1f}m {dist / max(T, 1e-6):5.2f}m/s   {turns + halts:3d} ({turns}/{halts})"
              f"        {100 * paused / max(T, 1e-6):4.0f}%")
        tot["turn"] += turns
        tot["halt"] += halts
        tot["paused"] += paused
        tot["dist"] += dist
        tot["time"] += T
    print(f"ALL  {tot['time']:5.0f}s {tot['dist']:5.1f}m {tot['dist'] / max(tot['time'], 1e-6):5.2f}m/s   "
          f"{tot['turn'] + tot['halt']:3d} ({tot['turn']}/{tot['halt']})        "
          f"{100 * tot['paused'] / max(tot['time'], 1e-6):4.0f}%")


if __name__ == "__main__":
    main()
