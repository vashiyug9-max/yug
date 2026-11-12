#!/usr/bin/env python3
"""
Simulated wheel odometry: what the Roboteq track encoders would report, for testing localization in
Habitat (the bridge's /odom is the TRUE pose, which a real robot never has).

  python3 src/robo_project/sim/tools/wheel_odom_sim.py --ros-args -p trans_scale:=1.03 -p rot_scale:=1.08

Integrates the true body-frame motion between /odom messages with errors:
  trans_scale   distance scale error (1.03 = encoders report 3% too much)
  rot_scale     heading scale error (tracks slip in turns: encoders over-report rotation)
  trans_noise / rot_noise  random noise per metre / per radian of motion
and publishes /wheel_odom (frame "odom_wheel", starts at 0, 0, 0 like a real robot). No TF.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class WheelOdomSim(Node):
    def __init__(self):
        super().__init__("wheel_odom_sim")
        p = self.declare_parameter
        self.ts = float(p("trans_scale", 1.03).value)
        self.rs = float(p("rot_scale", 1.08).value)
        self.tn = float(p("trans_noise", 0.02).value)
        self.rn = float(p("rot_noise", 0.03).value)
        self.rng = np.random.default_rng(int(p("seed", 1).value))
        self.prev = None
        self.pose = [0.0, 0.0, 0.0]
        self.pub = self.create_publisher(Odometry, "/wheel_odom", 50)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.get_logger().info(f"Wheel odometry: distance x{self.ts}, rotation x{self.rs}, "
                               f"noise {self.tn} m/m, {self.rn} rad/rad -> /wheel_odom")

    def on_odom(self, m: Odometry):
        cur = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        if self.prev is not None:
            px, py, pth = self.prev
            dxw, dyw = cur[0] - px, cur[1] - py
            fwd = math.cos(pth) * dxw + math.sin(pth) * dyw
            lat = -math.sin(pth) * dxw + math.cos(pth) * dyw
            dth = math.atan2(math.sin(cur[2] - pth), math.cos(cur[2] - pth))
            d = math.hypot(fwd, lat)
            fwd = fwd * self.ts + self.rng.normal(0, self.tn * d + 1e-6)
            dth = dth * self.rs + self.rng.normal(0, self.rn * abs(dth) + 0.005 * d + 1e-6)
            x, y, th = self.pose
            self.pose = [x + math.cos(th) * fwd - math.sin(th) * lat * self.ts,
                         y + math.sin(th) * fwd + math.cos(th) * lat * self.ts,
                         math.atan2(math.sin(th + dth), math.cos(th + dth))]
        self.prev = cur
        out = Odometry()
        out.header.stamp = m.header.stamp
        out.header.frame_id = "odom_wheel"
        out.child_frame_id = "base_footprint"
        out.pose.pose.position.x, out.pose.pose.position.y = self.pose[0], self.pose[1]
        out.pose.pose.orientation.z, out.pose.pose.orientation.w = math.sin(self.pose[2] / 2), math.cos(self.pose[2] / 2)
        out.twist = m.twist
        self.pub.publish(out)


def main():
    rclpy.init()
    node = WheelOdomSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
