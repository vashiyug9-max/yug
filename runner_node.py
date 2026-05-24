#!/usr/bin/env python3
import random
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class AutomatedRunnerNode(Node):
    def __init__(self):
        super().__init__("automated_runner_node")
        
        # FIXED: We are now natively publishing to the ROS 2 /cmd_vel topic!
        # No more messy UDP sockets or IP Addresses!
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # State management for our random wanderer
        self.state = "MOVE_FORWARD"  
        self.state_duration_ticks = 0
        self.max_forward_ticks = random.randint(20, 40) 
        
        self.timer = self.create_timer(0.1, self.automated_loop)
        
        self.get_logger().info("Automated Random Explorer Started! Publishing Twist messages directly to /cmd_vel")

    def automated_loop(self):
        msg = Twist()
        self.state_duration_ticks += 1

        if self.state == "MOVE_FORWARD":
            msg.linear.x = 0.25
            msg.angular.z = 0.0
            
            if self.state_duration_ticks >= self.max_forward_ticks:
                self.get_logger().info("Changing state: Time to turn dynamically.")
                self.switch_to_spin()

        elif self.state == "SPIN_AWAY":
            msg.linear.x = 0.0
            msg.angular.z = self.spin_direction * 0.5
            
            if self.state_duration_ticks >= 15:
                self.get_logger().info("Finished turning. Resuming forward exploration.")
                self.state = "MOVE_FORWARD"
                self.state_duration_ticks = 0
                self.max_forward_ticks = random.randint(20, 50)

        # Send the Twist message natively through ROS 2
        self.cmd_vel_pub.publish(msg)

    def switch_to_spin(self):
        self.state = "SPIN_AWAY"
        self.state_duration_ticks = 0
        self.spin_direction = random.choice([1.0, -1.0])

    def stop_robot(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info("Sent emergency stop command to /cmd_vel.")

def main(args=None):
    rclpy.init(args=args)
    node = AutomatedRunnerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down exploration node...")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
