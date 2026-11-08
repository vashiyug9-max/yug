#!/usr/bin/env python3
"""Replay a goal sequence on /goal_pose: next goal once the robot is within TOL of the current one
and has stopped, or after TIMEOUT. The first goal is motion_planner's default (0, 0), not sent.
"Current one" = where motion_planner says it drives to (/motion_planner_goal): a clicked goal too close
to a wall for the robot's body is moved up to 0.5 m (planner _safe_goal). The distance to the clicked
goal is reported too."""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

GOALS = [(0.0, 0.0, False), (4.38, -3.89, True), (5.00, 0.72, True), (-3.74, 5.32, True),
         (-3.33, -4.04, True), (-3.84, 5.59, True)]
TOL, TIMEOUT = 0.30, 150.0

rclpy.init()
node = rclpy.create_node("goal_sequencer")
pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
state = {"x": None, "y": None, "v": 0.0, "w": 0.0, "goal": None}


def odom(m):
    state.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y, v=m.twist.twist.linear.x, w=m.twist.twist.angular.z)


node.create_subscription(Odometry, "/odom", odom, 10)
qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
node.create_subscription(PoseStamped, "/motion_planner_goal",
                         lambda m: state.update(goal=(m.pose.position.x, m.pose.position.y)), qos)
while state["x"] is None:
    rclpy.spin_once(node, timeout_sec=0.5)

results = []
for i, (gx, gy, send) in enumerate(GOALS):
    if send:
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = gx, gy
        msg.pose.orientation.w = 1.0
        for _ in range(3):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.1)
    t0 = time.monotonic()
    print(f"[goal {i}] -> ({gx:.2f},{gy:.2f})", flush=True)
    still_since = None
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        tx, ty = gx, gy
        if state["goal"] and math.hypot(state["goal"][0] - gx, state["goal"][1] - gy) <= 0.6:
            tx, ty = state["goal"]                       # the planner's effective goal for this click
        d = math.hypot(state["x"] - tx, state["y"] - ty)
        stopped = abs(state["v"]) < 1e-3 and abs(state["w"]) < 1e-3
        still_since = (still_since or time.monotonic()) if (d < TOL and stopped) else None
        if still_since and time.monotonic() - still_since > 1.0:
            results.append((i, gx, gy, "REACHED", time.monotonic() - t0, d,
                            math.hypot(state["x"] - gx, state["y"] - gy)))
            break
        if time.monotonic() - t0 > TIMEOUT:
            results.append((i, gx, gy, "TIMEOUT", TIMEOUT, d, math.hypot(state["x"] - gx, state["y"] - gy)))
            break
    print(f"[goal {i}] {results[-1][3]} after {results[-1][4]:.1f} s (final distance {results[-1][5]:.2f} m)", flush=True)

print("\nSUMMARY")
for i, gx, gy, res, t, d, dc in results:
    print(f"  goal {i} ({gx:6.2f},{gy:6.2f}): {res:8s} {t:6.1f} s  dist {d:.2f} m"
          + (f"  (parked {dc:.2f} m from the clicked goal: too close to a wall)" if dc - d > 0.05 else ""))
print("SEQUENCE DONE", flush=True)
node.destroy_node()
rclpy.shutdown()
