#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import math
from robo_project.scripts.astar import Astar
from robo_project.scripts.basic_types import PosePixels

class ActionExecutive(Node):
    def __init__(self):
        super().__init__('action_executive')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        
        # RViz "2D Goal Pose" Listener
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        
        # 🟢 MAP PUBLISHER: Broadcasts the 2D layout to RViz
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 1)
        self.create_timer(1.0, self.publish_map)
        
        self.planner = Astar()
        
        try:
            raw_map = np.load('~/robo_project_ws/src/resources/inflated_binary_grid.npy')
            self.grid_map = raw_map.astype(int)
        except Exception as e:
            self.get_logger().error(f"Map Load Failed: {e}")
            self.grid_map = np.zeros((1024, 1024), dtype=int) 
            
        self.planner.map = self.grid_map
        
        desired_goal_r, desired_goal_c = 480, 512
        safe_goal_r, safe_goal_c = self.get_nearest_free_pixel(desired_goal_r, desired_goal_c)
        self.planner.goal_cell = PosePixels(safe_goal_r, safe_goal_c)
        
        self.get_logger().info("✅ BRAIN ONLINE! Waiting for RViz 2D Goal Pose clicks...")
        self.create_timer(0.2, self.timer_callback)
        self.current_action = "stop"

    def get_nearest_free_pixel(self, r, c, search_radius=40):
        if not (0 <= r < 1024 and 0 <= c < 1024):
            return 512, 512 
        if self.grid_map[r, c] == 0: 
            return r, c 
        for radius in range(1, search_radius):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) == radius:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 1024 and 0 <= nc < 1024:
                            if self.grid_map[nr, nc] == 0: 
                                return nr, nc
        return r, c 

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = 0.02  
        msg.info.width = 1527
        msg.info.height = 1024
        
        msg.info.origin.position.x = -15.0
        msg.info.origin.position.y = -10.0
        
        grid_data = np.where(self.grid_map == 1, 100, 0).astype(np.int8)
        msg.data = grid_data.flatten().tolist()
        
        self.map_pub.publish(msg)

    def goal_callback(self, msg):
        raw_row = int(512 + (msg.pose.position.y * 10))
        raw_col = int(512 + (msg.pose.position.x * 10))
        
        safe_r, safe_c = self.get_nearest_free_pixel(raw_row, raw_col)
        self.planner.goal_cell = PosePixels(safe_r, safe_c)
        
        self.current_action = "turn_left" 
        self.get_logger().info(f"📍 NEW MISSION RECEIVED FROM RVIZ: Driving to ({safe_r}, {safe_c})")

    def odom_callback(self, msg):
        raw_row = int(512 + (msg.pose.pose.position.y * 10))
        raw_col = int(512 + (msg.pose.pose.position.x * 10))
        
        row = max(0, min(1023, raw_row))
        col = max(0, min(1023, raw_col))
        
        if abs(row - self.planner.goal_cell.r) < 3 and abs(col - self.planner.goal_cell.c) < 3:
            if self.current_action != "stop":
                self.current_action = "stop"
                self.get_logger().info("🎯 GOAL REACHED! Standing by for next command.")
            return
            
        # --- FIXED: Pure, standard ROS 2 Yaw calculation ---
        yaw = math.atan2(msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) * 2
        
        start_r, start_c = self.get_nearest_free_pixel(row, col)
        path = self.planner.run_astar(PosePixels(start_r, start_c, yaw))
        
        if path and len(path) > 1:
            step_index = max(0, len(path) - 10)
            next_c, next_r = path[step_index].c, path[step_index].r
            
            target_yaw = math.atan2(next_r - start_r, next_c - start_c)
            angle_diff = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            
            if abs(angle_diff) < 0.4:
                self.current_action = "move_forward"
            else:
                # --- FIXED: Normal steering! If the target is Left (Positive), turn left! ---
                self.current_action = "turn_left" if angle_diff > 0 else "turn_right"
        else:
            self.current_action = "turn_left"

    def timer_callback(self):
        if self.planner.goal_cell is None:
            return

        cmd = Twist()
        if self.current_action == "move_forward": 
            cmd.linear.x = 0.4
        elif self.current_action == "turn_left": 
            cmd.angular.z = 0.5
        elif self.current_action == "turn_right": 
            cmd.angular.z = -0.5
            
        self.vel_pub.publish(cmd)

def main():
    rclpy.init()
    rclpy.spin(ActionExecutive())

if __name__ == "__main__": 
    main()
