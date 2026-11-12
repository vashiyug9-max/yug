#!/usr/bin/env python3
"""
Simulated IMU gyro for testing the real-robot odometry chain in Habitat:
true yaw rate (from the bridge's true /odom) x gyro_scale + bias + noise -> /imu/data (sensor_msgs/Imu).

  python3 src/robo_project/sim/tools/imu_sim.py --ros-args -p gyro_scale:=1.01

Parameters: gyro_scale (1.01), bias (0.002 rad/s), noise (0.01 rad/s per sample), rate: one message per /odom.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ImuSim(Node):
    def __init__(self):
        super().__init__("imu_sim")
        p = self.declare_parameter
        self.scale = float(p("gyro_scale", 1.01).value)
        self.bias = float(p("bias", 0.002).value)
        self.noise = float(p("noise", 0.01).value)
        self.rng = np.random.default_rng(int(p("seed", 2).value))
        self.prev = None
        self.pub = self.create_publisher(Imu, "/imu/data", qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, 50)
        self.get_logger().info(f"IMU gyro: scale {self.scale}, bias {self.bias} rad/s, noise {self.noise} -> /imu/data")

    def on_odom(self, m: Odometry):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        yaw = yaw_of(m.pose.pose.orientation)
        if self.prev is not None and t > self.prev[0]:
            dt = t - self.prev[0]
            wz = math.atan2(math.sin(yaw - self.prev[1]), math.cos(yaw - self.prev[1])) / dt
            out = Imu()
            out.header.stamp = m.header.stamp
            out.header.frame_id = "base_footprint"         # (real robot: imu_link, TF from our robot model)
            out.orientation_covariance[0] = -1.0                   # no orientation (gyro only)
            out.angular_velocity.z = wz * self.scale + self.bias + self.rng.normal(0, self.noise)
            out.linear_acceleration.z = 9.81
            self.pub.publish(out)
        self.prev = (t, yaw)


def main():
    rclpy.init()
    node = ImuSim()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
