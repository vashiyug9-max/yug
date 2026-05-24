#!/usr/bin/env python3
import os
import json
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

class DataHarvesterNode(Node):
    def __init__(self):
        super().__init__("data_harvester_node")
        
        # 1. Create a dedicated folder for the ML team's dataset
        self.save_dir = os.path.expanduser("~/vinebot_ws/training_dataset")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Variables to store the latest camera frames
        self.latest_front = None
        self.latest_left = None
        self.latest_right = None
        
        # Distance tracking to prevent taking 1,000 photos in the same spot
        self.last_saved_x = None
        self.last_saved_y = None
        self.save_distance_threshold = 0.5  # Save a new snapshot every 0.5 meters
        self.sample_count = 0
        
        # FIXED: Listening to the actual topics defined in habitat_bridge_vinebot_2.py!
        self.create_subscription(Image, "/camera/front/image_raw", self.front_cb, 10)
        self.create_subscription(Image, "/camera/left/image_raw", self.left_cb, 10)
        self.create_subscription(Image, "/camera/right/image_raw", self.right_cb, 10)
        
        # Subscribe to the actual Odometry topic
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        
        self.get_logger().info(f"Data Harvester Online! Saving training data to: {self.save_dir}")

    def image_to_cv2(self, msg):
        """Converts raw ROS 2 Image bytes into an OpenCV array perfectly."""
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr

    def front_cb(self, msg): self.latest_front = self.image_to_cv2(msg)
    def left_cb(self, msg): self.latest_left = self.image_to_cv2(msg)
    def right_cb(self, msg): self.latest_right = self.image_to_cv2(msg)

    def odom_cb(self, msg):
        # Ensure all cameras have booted up before we start saving data
        if self.latest_front is None or self.latest_left is None or self.latest_right is None:
            return
            
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        
        # If it's the very first movement, trigger a baseline save
        if self.last_saved_x is None:
            self.last_saved_x = current_x
            self.last_saved_y = current_y
            self.save_dataset_sample(msg)
            return
            
        # Calculate geometric distance using Pythagoras theorem
        distance = math.sqrt((current_x - self.last_saved_x)**2 + (current_y - self.last_saved_y)**2)
        
        # If the robot has traveled 0.5m since the last photo, take another one!
        if distance >= self.save_distance_threshold:
            self.save_dataset_sample(msg)
            self.last_saved_x = current_x
            self.last_saved_y = current_y

    def save_dataset_sample(self, odom_msg):
        self.sample_count += 1
        prefix = f"{self.sample_count:05d}"
        
        # Save the 3 panoramic images
        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_front.jpg"), self.latest_front)
        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_left.jpg"), self.latest_left)
        cv2.imwrite(os.path.join(self.save_dir, f"{prefix}_right.jpg"), self.latest_right)
        
        # Save the exact coordinates for the Ground-Truth labels
        pose_data = {
            "sample_id": self.sample_count,
            "position": {
                "x": odom_msg.pose.pose.position.x,
                "y": odom_msg.pose.pose.position.y,
                "z": odom_msg.pose.pose.position.z
            },
            "yaw_quaternion": {
                "z": odom_msg.pose.pose.orientation.z,
                "w": odom_msg.pose.pose.orientation.w
            }
        }
        
        with open(os.path.join(self.save_dir, f"{prefix}_pose.json"), "w") as f:
            json.dump(pose_data, f, indent=4)
            
        self.get_logger().info(f"Snapshot #{self.sample_count} Saved! (Traveled 0.5m)")

def main(args=None):
    rclpy.init(args=args)
    node = DataHarvesterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down harvester...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
