#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import numpy as np

# Import your existing processor layout directly
from robo_project.scripts.map_handler import MapFrameManager

class StaticMapServerNode(Node):
    def __init__(self):
        super().__init__('static_map_server_node')
        
        # Initialize your existing processor framework
        # Defaulting to discrete space mapping layout matching your launches
        self.map_manager = MapFrameManager(use_discrete_state_space=True)
        
        # Publishers and Broadcasters
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        # Timer to latch map publishing data across network channels
        self.timer = self.create_timer(1.0, self.publish_map)
        
        # Broadcast standard static identity transforms connecting map coordinate spaces
        self.broadcast_static_transforms()
        self.get_logger().info("Static Map Server Node Initialized Successfully.")

    def broadcast_static_transforms(self):
        # Establish structural linking between global world 'map' and navigation 'odom'
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_static_broadcaster.sendTransform(t)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        
        # Build map metadata profile fields
        # Invert downscale tracking to derive resolution scaling mapping safely
        msg.info.resolution = float(self.map_manager.map_resolution_desired)
        msg.info.width = self.map_manager.map_with_border.shape[1]
        msg.info.height = self.map_manager.map_with_border.shape[0]
        
        # Align map array origin to match your custom pixel transform math bounds center
        msg.info.origin.position.x = float(- (msg.info.width // 2) * msg.info.resolution)
        msg.info.origin.position.y = float(- (msg.info.height // 2) * msg.info.resolution)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        # Translate internal map metrics cleanly:
        # Internal space mapping uses: 1.0 = Free, 0.0 = Occupied
        # Standard ROS occupancy grids map to: 0 = Free, 100 = Occupied, -1 = Unknown
        flat_grid = self.map_manager.map_with_border.flatten()
        ros_grid_data = np.where(flat_grid == 1.0, 0, 100).astype(np.int8)
        
        msg.data = ros_grid_data.tolist()
        self.map_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = StaticMapServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
