#!/usr/bin/env python3
"""Simulation-only, fail-closed gate for localization-driven proposed cmd_vel.

Deliberately disabled by default. Even when explicitly enabled, it requires
an accurate estimated pose relative to simulator /odom before forwarding any
movement. It does NOT use GT to compute navigation position or path.
"""
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))


def difference(a, b):
    return math.atan2(math.sin(a-b), math.cos(a-b))


class LocalizationMotionGate(Node):
    def __init__(self):
        super().__init__('cmn_simulation_motion_gate')
        self.declare_parameter('enable_motion', False)
        self.armed = bool(self.get_parameter('enable_motion').value)
        self.stable = False
        self.goal_set = False
        self.gt = None
        self.est = None
        self.cmd = None
        self.gt_time = self.est_time = self.cmd_time = 0.0
        self.last_block_log = 0.0
        self.create_subscription(Twist, '/localization/proposed_cmd_vel', self.on_cmd, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 10)
        self.create_subscription(Bool, '/localization/pose_stable', self.on_stable, 10)
        self.create_subscription(Odometry, '/odom', self.on_gt, 10)
        self.create_subscription(Odometry, '/localization/nav_odom', self.on_est, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10) if self.armed else None
        self.create_timer(0.1, self.tick)
        self.get_logger().warn('SIMULATION-ONLY motion gate: ' +
                               ('ARMED but GT error + stability guard active' if self.armed
                                else 'DISARMED: publishes NOTHING on /cmd_vel'))

    def on_goal(self, msg):
        if msg.header.frame_id == 'map':
            self.goal_set = True
            self.get_logger().info('New explicit /goal_pose received; movement authorization still gated.')

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_time = time.monotonic()

    def on_stable(self, msg):
        self.stable = bool(msg.data)

    def on_gt(self, msg):
        self.gt = msg
        self.gt_time = time.monotonic()

    def on_est(self, msg):
        self.est = msg
        self.est_time = time.monotonic()

    def tick(self):
        if not self.armed:
            return
        now = time.monotonic()
        reason = None
        simulator_publishers = self.get_publishers_info_by_topic('/odom')
        simulator_only = any(pub.node_name == 'habitat_bridge' for pub in simulator_publishers)
        if not simulator_only:
            reason = 'not receiving odom from expected Habitat bridge; simulation gate disabled'
        elif not self.goal_set:
            reason = 'no explicit map-frame goal received in this session'
        elif not self.stable:
            reason = 'localization not stable'
        elif self.gt is None or self.est is None or self.cmd is None:
            reason = 'missing pose or motion command'
        elif now-self.gt_time > 0.5 or now-self.est_time > 0.5 or now-self.cmd_time > 0.4:
            reason = 'stale pose or cmd_vel'
        else:
            gp = self.gt.pose.pose.position
            ep = self.est.pose.pose.position
            e = math.hypot(gp.x-ep.x, gp.y-ep.y)
            a = abs(math.degrees(difference(
                yaw(self.gt.pose.pose.orientation), yaw(self.est.pose.pose.orientation))))
            if e > 0.35 or a > 15.0:
                reason = f'GT guard blocked ({e:.2f}m / {a:.0f}deg)' 
        if reason is not None:
            self.pub.publish(Twist())
            if now-self.last_block_log > 3.0:
                self.get_logger().warn(reason)
                self.last_block_log = now
            return
        safe = Twist()
        safe.linear.x = max(-0.12, min(0.12, float(self.cmd.linear.x)))
        safe.angular.z = max(-0.3, min(0.3, float(self.cmd.angular.z)))
        self.pub.publish(safe)


def main(args=None):
    rclpy.init(args=args)
    n = LocalizationMotionGate()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if n.pub is not None:
            for _ in range(3):
                n.pub.publish(Twist())
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
