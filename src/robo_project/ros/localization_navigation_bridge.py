#!/usr/bin/env python3
"""CMN navigation preview: use coarse-map pose plus *relative* odometry only.

Never injects absolute /odom position as a localization estimate. Projects
odom(t0)^-1 * odom(t) into the latest accepted localizer pose; rejects big
pose jumps. Planner/runner can subscribe to /localization/nav_odom via remap.
This is not proof of accurate localization or safe autonomous navigation.
"""
import math
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


def stamp(msg):
    return float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9


def angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def quat(theta, q):
    q.x = q.y = 0.0
    q.z, q.w = math.sin(theta/2), math.cos(theta/2)


def relative_propagation(anchor_pose, anchor_odom, current_odom):
    # body-frame odometry increment; not global simulator position as input.
    ax, ay, ah = anchor_odom
    cx, cy, ch = current_odom
    px, py, ph = anchor_pose
    dx, dy = cx-ax, cy-ay
    forward = math.cos(ah)*dx + math.sin(ah)*dy
    left = -math.sin(ah)*dx + math.cos(ah)*dy
    return (px + math.cos(ph)*forward - math.sin(ph)*left,
            py + math.sin(ph)*forward + math.cos(ph)*left,
            angle(ph + angle(ch-ah)))


class CmnNavigationBridge(Node):
    def __init__(self):
        super().__init__('cmn_navigation_bridge')
        self.history = deque(maxlen=1200)
        self.anchor = None  # (map_pose, reference_odom)
        self.last_good_stamp = None
        self.stable_count = 0
        self.map_ok = False
        self.last_warning_stamp = -1.0

        self.odom_pub = self.create_publisher(Odometry, '/localization/nav_odom', 10)
        self.stable_pub = self.create_publisher(Bool, '/localization/pose_stable', 10)
        self.marker_pub = self.create_publisher(Marker, '/localization/robot_marker', 10)
        self.raw_pub = self.create_publisher(Marker, '/localization/raw_pose_marker', 10)
        self.goal_pub = self.create_publisher(Marker, '/localization/goal_marker', 10)
        self.tf = TransformBroadcaster(self)

        self.create_subscription(OccupancyGrid, '/map', self.on_map, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 30)
        self.create_subscription(PoseStamped, '/localization/provisional_pose', self.on_raw, 10)
        self.create_subscription(PoseStamped, '/motion_planner_goal', self.on_goal, 10)
        self.get_logger().info('CMN NAV PREVIEW: localization -> nav odom (relative-motion only).')

    def on_map(self, msg):
        # Same sketch grid conventions as localization_map_step342 and the runner.
        correct = (
            msg.info.width == 768 and msg.info.height == 1024 and
            abs(float(msg.info.resolution) - 0.02) < 1e-6 and
            abs(msg.info.origin.position.x + 7.68) < 0.03 and
            abs(msg.info.origin.position.y + 10.24) < 0.03 and
            msg.header.frame_id == 'map'
        )
        if not correct and self.map_ok:
            self.get_logger().error('MAP GEOMETRY CHANGED: nav pose paused.')
        if not correct and not self.map_ok:
            t = self.get_clock().now().nanoseconds / 1e9
            if t-self.last_warning_stamp > 10:
                self.get_logger().error('Expected 768x1024 sketch, 0.02m/cell, centred map. Not publishing nav pose.')
                self.last_warning_stamp = t
        self.map_ok = correct

    @staticmethod
    def make_marker(x, y, heading, name, ident, rgb, kind=Marker.ARROW):
        m = Marker()
        m.header.frame_id = 'map'  # the sketch's coordinate frame, not Habitat odom
        m.ns = name
        m.id = ident
        m.type = kind
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.08
        quat(heading, m.pose.orientation)
        m.scale.x = 0.60 if kind == Marker.ARROW else 0.32
        m.scale.y = 0.19 if kind == Marker.ARROW else 0.32
        m.scale.z = 0.15 if kind == Marker.ARROW else 0.32
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = 1.0
        return m

    def on_goal(self, msg):
        # Planner owns actual goal (may shift it away from walls).
        p = msg.pose.position
        m = self.make_marker(p.x, p.y, 0., 'cmn_goal', 3, (0.2, 0.5, 1.), Marker.SPHERE)
        m.header.stamp = self.get_clock().now().to_msg()
        self.goal_pub.publish(m)

    def on_raw(self, msg):
        if msg.header.frame_id != 'cmn_sketch_map':
            self.get_logger().error('Unsupported localizer frame; no silent frame relabeling.')
            return
        if not self.map_ok or not self.history:
            return
        ts = stamp(msg)
        odom_entry = min(self.history, key=lambda v: abs(v[0]-ts))
        gap = abs(odom_entry[0]-ts)
        if gap > 0.25:
            self.get_logger().warn(f'SKIP localization frame: no odom near measurement ({gap:.2f}s).')
            return
        raw = (float(msg.pose.position.x), float(msg.pose.position.y), yaw(msg.pose.orientation))
        od = odom_entry[1:]
        marker = self.make_marker(*raw, 'cmn_raw', 2, (1., 0.3, 0.3))
        marker.header.stamp = self.get_clock().now().to_msg()
        self.raw_pub.publish(marker)
        if self.anchor is not None:
            old = relative_propagation(*self.anchor, od)
            pos_jump = math.hypot(raw[0]-old[0], raw[1]-old[1])
            yaw_jump = abs(math.degrees(angle(raw[2]-old[2])))
            if pos_jump > 0.75 or yaw_jump > 50.0:
                self.stable_count = 0
                self.get_logger().error(
                    f'REJECT global pose jump: {pos_jump:.2f}m / {yaw_jump:.1f}deg. '
                    'Raw red marker retained; navigation motion NOT approved.')
                return
        self.anchor = (raw, od)
        self.last_good_stamp = ts
        self.stable_count += 1
        self.get_logger().info(f'ACCEPT localizer estimate ({raw[0]:.2f},{raw[1]:.2f},'
                               f'{math.degrees(raw[2]):.1f}deg); stable samples={self.stable_count}')

    def on_odom(self, msg):
        t = stamp(msg)
        p = msg.pose.pose.position
        od = (float(p.x), float(p.y), yaw(msg.pose.pose.orientation))
        self.history.append((t, *od))
        age = float('inf') if self.last_good_stamp is None else t-self.last_good_stamp
        ready = self.map_ok and self.anchor is not None and 0 <= age < 20 and self.stable_count >= 3
        self.stable_pub.publish(Bool(data=bool(ready)))
        if self.anchor is None or not self.map_ok:
            return
        x, y, h = relative_propagation(*self.anchor, od)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = 'map'
        out.child_frame_id = 'cmn_predicted_base'
        out.pose.pose.position.x = x
        out.pose.pose.position.y = y
        quat(h, out.pose.pose.orientation)
        # Conservative placeholder covariance; NOT a statistically calibrated estimate.
        out.pose.covariance[0] = out.pose.covariance[7] = 25.0
        out.pose.covariance[35] = 3.0
        self.odom_pub.publish(out)

        marker = self.make_marker(x, y, h, 'cmn_predicted', 1,
                                  (0.2, 0.9, 0.3) if ready else (1., 0.7, 0.0))
        marker.header.stamp = msg.header.stamp
        self.marker_pub.publish(marker)

        transform = TransformStamped()
        transform.header = out.header
        transform.child_frame_id = 'cmn_predicted_base'
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        quat(h, transform.transform.rotation)
        self.tf.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    n = CmnNavigationBridge()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
