#!/usr/bin/env python3
"""
Deadman gate for the PS4 joystick.

ps4_ros2 publishes a velocity on EVERY joystick update (zeros when nothing is touched), so twist_mux
would always let the joystick win over the autonomous driving. This node forwards ps4_ros2's commands
only while the enable button (default L1 = buttons[4]) is held; on release it sends one zero command.
twist_mux then gives the motors back to the autonomy (/cmd_vel) after its timeout.

  in:  /joy (sensor_msgs/Joy), in_topic (default /cmd_vel_joy_raw, from ps4_ros2)
  out: out_topic (default /cmd_vel_joy, to twist_mux)
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy


class JoyGate(Node):
    def __init__(self):
        super().__init__("joy_gate")
        self.button = int(self.declare_parameter("enable_button", 4).value)
        in_topic = str(self.declare_parameter("in_topic", "/cmd_vel_joy_raw").value)
        out_topic = str(self.declare_parameter("out_topic", "/cmd_vel_joy").value)
        self.held = False
        self.pub = self.create_publisher(Twist, out_topic, 10)
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.create_subscription(Twist, in_topic, self.on_cmd, 10)
        self.get_logger().info(f"Joystick drives only while button {self.button} (L1) is held: "
                               f"{in_topic} -> {out_topic}")

    def on_joy(self, msg: Joy):
        held = len(msg.buttons) > self.button and msg.buttons[self.button] == 1
        if self.held and not held:
            self.pub.publish(Twist())          # released: stop once, then autonomy takes over
            self.get_logger().info("Joystick released - autonomy may drive again")
        elif held and not self.held:
            self.get_logger().info("Joystick active (L1 held) - overrides autonomy")
        self.held = held

    def on_cmd(self, msg: Twist):
        if self.held:
            self.pub.publish(msg)


def main():
    rclpy.init()
    node = JoyGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
