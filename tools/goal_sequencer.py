#!/usr/bin/env python3
"""Replay a goal sequence on /goal_pose: next goal once the robot is within TOL of the current one
and has stopped, or after TIMEOUT. The first goal is motion_planner's default (0, 0), not sent."""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

GOALS = [(0.0, 0.0, False), (4.38, -3.89, True), (5.00, 0.72, True), (-3.74, 5.32, True),
         (-3.33, -4.04, True), (-3.84, 5.59, True)]
TOL, TIMEOUT = 0.30, 150.0

rclpy.init()
node = rclpy.create_node("goal_sequencer")
pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
state = {"x": None, "y": None, "v": 0.0, "w": 0.0}


def odom(m):
    state.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y, v=m.twist.twist.linear.x, w=m.twist.twist.angular.z)


node.create_subscription(Odometry, "/odom", odom, 10)
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
        d = math.hypot(state["x"] - gx, state["y"] - gy)
        stopped = abs(state["v"]) < 1e-3 and abs(state["w"]) < 1e-3
        still_since = (still_since or time.monotonic()) if (d < TOL and stopped) else None
        if still_since and time.monotonic() - still_since > 1.0:
            results.append((i, gx, gy, "REACHED", time.monotonic() - t0, d))
            break
        if time.monotonic() - t0 > TIMEOUT:
            results.append((i, gx, gy, "TIMEOUT", TIMEOUT, d))
            break
    print(f"[goal {i}] {results[-1][3]} after {results[-1][4]:.1f} s (final distance {results[-1][5]:.2f} m)", flush=True)

print("\nSUMMARY")
for i, gx, gy, res, t, d in results:
    print(f"  goal {i} ({gx:6.2f},{gy:6.2f}): {res:8s} {t:6.1f} s  dist {d:.2f} m")
print("SEQUENCE DONE", flush=True)
node.destroy_node()
rclpy.shutdown()
