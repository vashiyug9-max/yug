#!/usr/bin/env python3
"""
Send ONE goal (after the planner's start-up plan) and report how the robot got there.

  python3 src/robo_project/sim/tools/one_goal.py GOAL_X GOAL_Y [TIME_LIMIT_S]

Prints: final position, time, distance to the clicked goal and the distance driven. The run ends when
the robot has stood still for 4 s (or at the time limit).
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


def main():
    gx, gy = float(sys.argv[1]), float(sys.argv[2])
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    rclpy.init()
    n = rclpy.create_node("one_goal")
    pub = n.create_publisher(PoseStamped, "/goal_pose", 10)
    st = {"x": None, "d": 0.0, "v": 0.0, "w": 0.0}

    def cb(m):
        p = m.pose.pose.position
        if st["x"] is not None:
            st["d"] += math.hypot(p.x - st["x"], p.y - st["y"])
        st.update(x=p.x, y=p.y, v=m.twist.twist.linear.x, w=m.twist.twist.angular.z)

    n.create_subscription(Odometry, "/odom", cb, 10)
    while st["x"] is None:
        rclpy.spin_once(n, timeout_sec=0.5)
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.1)
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.pose.position.x, msg.pose.position.y = gx, gy
    msg.pose.orientation.w = 1.0
    for _ in range(3):
        pub.publish(msg)
        rclpy.spin_once(n, timeout_sec=0.1)
    st["d"] = 0.0
    t0, still = time.monotonic(), None
    while time.monotonic() - t0 < limit:
        rclpy.spin_once(n, timeout_sec=0.1)
        moving = abs(st["v"]) > 1e-3 or abs(st["w"]) > 1e-3
        still = None if moving else (still or time.monotonic())
        if still and time.monotonic() - still > 4.0 and time.monotonic() - t0 > 6:
            break
    t = (still if still else time.monotonic()) - t0
    print(f"  robot at ({st['x']:.2f},{st['y']:.2f}) after {t:.0f} s, "
          f"{math.hypot(st['x'] - gx, st['y'] - gy):.2f} m from the clicked goal, drove {st['d']:.1f} m", flush=True)
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
