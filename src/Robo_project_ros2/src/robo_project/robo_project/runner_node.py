#!/usr/bin/env python3
import json
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HABITAT_IP = "127.0.0.1"
HABITAT_PORT = 5005


class CmdVelUdpSender(Node):
    def __init__(self):
        super().__init__("cmd_vel_udp_sender")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sub = self.create_subscription(Twist, "/cmd_vel", self.cb, 10)
        self.get_logger().info(f"Sending /cmd_vel to {HABITAT_IP}:{HABITAT_PORT}")

    def cb(self, msg):
        data = {
            "linear_x": float(msg.linear.x),
            "linear_y": float(msg.linear.y),
            "linear_z": float(msg.linear.z),
            "angular_x": float(msg.angular.x),
            "angular_y": float(msg.angular.y),
            "angular_z": float(msg.angular.z),
        }
        self.sock.sendto(json.dumps(data).encode("utf-8"), (HABITAT_IP, HABITAT_PORT))


def main():
    rclpy.init()
    node = CmdVelUdpSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
def main(args=None):
    rclpy.init(args=args)
    node = RunnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # stop robot before shutdown
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        node.cmd_vel_pub.publish(stop_msg)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
