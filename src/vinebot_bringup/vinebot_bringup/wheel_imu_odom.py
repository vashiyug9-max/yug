#!/usr/bin/env python3
"""
Wheels + IMU odometry (the same fusion as config/ekf.yaml, without needing robot_localization).

  ros2 run vinebot_bringup wheel_imu_odom
  (autonomy.launch.py starts it automatically when robot_localization is not installed)

Tracked robots slip in turns, so the encoders report the turn badly (~8% in the tests) but the distance well.
This node takes the FORWARD DISTANCE from the wheel odometry and the TURN from the IMU gyro:
  /wheel_odom (nav_msgs/Odometry, Roboteq)  -> distance driven between two messages (pose difference)
  /imu/data   (sensor_msgs/Imu)             -> angular_velocity.z integrated over the same time
and publishes /odometry/filtered (frame odom -> base_footprint, no TF), the input of sketch_pf_node.
If the IMU stops (> imu_timeout s) it falls back to the wheel heading and warns.

Parameters: wheel_topic, imu_topic, out_topic, imu_timeout (0.3 s),
            imu_yaw_sign (1.0; -1.0 if the IMU is mounted upside down - check: turn left = positive),
            gyro_bias_s (2.0: the gyro bias is averaged while the robot stands still at start; afterwards it is
            updated whenever the wheels report standstill for > 1 s - gyros drift with temperature).
robot_localization's EKF cannot estimate a gyro bias, so this node is the default (odometry:=builtin).
"""
import math
from collections import deque

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def stamp_s(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class WheelImuOdom(Node):
    def __init__(self):
        super().__init__("wheel_imu_odom")
        p = self.declare_parameter
        wheel_topic = p("wheel_topic", "/wheel_odom").value
        imu_topic = p("imu_topic", "/imu/data").value
        out_topic = p("out_topic", "/odometry/filtered").value
        self.imu_timeout = float(p("imu_timeout", 0.3).value)
        self.sign = float(p("imu_yaw_sign", 1.0).value)
        self.bias_s = float(p("gyro_bias_s", 2.0).value)
        self.gyro = deque(maxlen=200)         # (t, integrated yaw, wz)
        self.bias, self.bias_samples, self.bias_t0 = 0.0, [], None
        self.prev = None                      # (t, wheel pose)
        self.pose = [0.0, 0.0, 0.0]
        self.last_imu_wall = None
        self.fallback_warned = False
        self.still = True                     # robot standing still (gyro bias is learned only then)
        self.still_since = 0.0                # stamp since when the wheels report no motion (None = moving)
        self.prev_gyro = None                 # integrated gyro heading at the previous wheel message
        self.n_wheel = 0
        self.pub = self.create_publisher(Odometry, out_topic, 50)
        self.create_subscription(Odometry, wheel_topic, self.on_wheel, 50)
        self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)
        self.get_logger().info(f"Odometry: distance from {wheel_topic}, turn from {imu_topic} gyro -> {out_topic}")

    def on_imu(self, m: Imu):
        t = stamp_s(m.header)
        wz = self.sign * m.angular_velocity.z
        self.last_imu_wall = self.get_clock().now().nanoseconds * 1e-9
        if self.bias_t0 is None:
            self.bias_t0 = t
        if self.prev is not None and self.still_since is not None:
            if t - self.bias_t0 < self.bias_s:                # start: plain average
                self.bias_samples.append(wz)
                self.bias = sum(self.bias_samples) / len(self.bias_samples)
            elif t - self.still_since > 1.0:                 # standing still again: follow the bias drift
                self.bias += 0.01 * (wz - self.bias)
        if self.gyro:                                     # running integral of the (bias-corrected) rate
            t_prev, yaw_prev, w_prev = self.gyro[-1]
            if t <= t_prev:
                return
            yaw = yaw_prev + 0.5 * (w_prev + wz - 2 * self.bias) * (t - t_prev)
        else:
            yaw = 0.0
        self.gyro.append((t, yaw, wz))

    def gyro_yaw_at(self, t):
        """Integrated gyro heading at time t (interpolated; up to 0.1 s extrapolated with the last rate)."""
        if not self.gyro:
            return None
        t_last, yaw_last, w_last = self.gyro[-1]
        if t >= t_last:
            return yaw_last + (w_last - self.bias) * min(t - t_last, 0.1)
        for (ta, ya, _), (tb, yb, _) in zip(reversed(list(self.gyro)[:-1]), reversed(list(self.gyro)[1:])):
            if ta <= t <= tb:
                return ya + (yb - ya) * (t - ta) / (tb - ta)
            if tb < t:
                break
        return self.gyro[0][1] if t < self.gyro[0][0] else None

    def on_wheel(self, m: Odometry):
        t = stamp_s(m.header)
        cur = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        if self.prev is None:
            self.prev = (t, cur)
            self.prev_gyro = self.gyro_yaw_at(t)
            self.publish(m, 0.0)
            return
        t0, (px, py, pth) = self.prev
        dxw, dyw = cur[0] - px, cur[1] - py
        fwd = math.cos(pth) * dxw + math.sin(pth) * dyw
        wheel_dth = wrap(cur[2] - pth)
        self.still = abs(fwd) < 1e-4 and abs(wheel_dth) < 1e-4
        self.still_since = (self.still_since or t) if self.still else None
        now = self.get_clock().now().nanoseconds * 1e-9
        dth = None
        g = self.gyro_yaw_at(t) if self.last_imu_wall is not None and now - self.last_imu_wall < self.imu_timeout \
            else None
        if g is not None and self.prev_gyro is not None:
            dth = g - self.prev_gyro                      # cumulative: a turn missed now is caught up next time
        self.prev_gyro = g
        if dth is None:
            dth = wheel_dth
            if not self.fallback_warned and self.prev_gyro is None and self.n_wheel > 50:
                self.get_logger().warn("No IMU data - using the wheel heading (larger heading errors in turns).")
                self.fallback_warned = True
        elif self.fallback_warned:
            self.get_logger().info("IMU data again - turn from the gyro.")
            self.fallback_warned = False
        self.n_wheel += 1
        x, y, th = self.pose
        mid = th + dth / 2
        self.pose = [x + math.cos(mid) * fwd, y + math.sin(mid) * fwd, wrap(th + dth)]
        self.prev = (t, cur)
        self.publish(m, dth / (t - t0) if t > t0 else 0.0)

    def publish(self, m, wz):
        out = Odometry()
        out.header.stamp = m.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_footprint"
        out.pose.pose.position.x, out.pose.pose.position.y = self.pose[0], self.pose[1]
        out.pose.pose.orientation.z = math.sin(self.pose[2] / 2)
        out.pose.pose.orientation.w = math.cos(self.pose[2] / 2)
        out.twist.twist.linear.x = m.twist.twist.linear.x
        out.twist.twist.angular.z = wz
        self.pub.publish(out)


def main():
    rclpy.init()
    node = WheelImuOdom()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
