#!/usr/bin/env python3

"""
Main node for running the project. This should be run on the host PC while the locobot is connected.
Migrated from ROS 1 to ROS 2.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool

import yaml, sys, os
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from math import pi, atan2, asin
import numpy as np
import cv2
from time import strftime, time
from typing import Callable

from robo_project.scripts.cmn_interface import CoarseMapNavInterface, CmnConfig
from robo_project.scripts.basic_types import PoseMeters, PosePixels, rotate_image_to_north
import locobot_interface


class RunnerNode(Node):
    """
    Main ROS 2 node for the CMN project.
    All ROS 1 globals are now instance attributes on this node.
    """

    def __init__(self):
        super().__init__('runner_node')

        self.cv_bridge = CvBridge()
        self.cmn_interface: CoarseMapNavInterface = None

        # RealSense measurement buffers.
        self.most_recent_rgb_meas = None
        self.desired_meas_shape = None
        self.most_recent_depth_meas = None
        self.depth_proc_func: Callable = None

        # Odometry.
        self.first_odom: PoseMeters = None

        # Config / mode flags.
        self.run_modes = ["continuous", "discrete", "discrete_random"]
        self.run_mode = None
        self.use_ground_truth_map_to_generate_observations = False
        self.show_live_viz = False
        self.verbose = False
        self.use_lidar_as_ground_truth = False
        self.manual_goal_cell: PosePixels = None
        self.use_depth_pointcloud = False

        # Data saving.
        self.save_training_data: bool = False
        self.training_data_dirpath: str = None

        # Live flags.
        self.viz_paused = False
        self.pub_viz_images: bool = False

        # Read params from yaml.
        self.read_params()

        # Publishers.
        # ROS 2: create_publisher(MsgType, topic, qos)
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/locobot/mobile_base/commands/velocity", 1)
        self.sim_viz_pub = self.create_publisher(Image, "/cmn/viz/sim", 1)
        self.cmn_viz_pub = self.create_publisher(Image, "/cmn/viz/cmn", 1)

        # Parse launch arguments.
        # ROS 2: args are passed as ROS 2 parameters instead of sys.argv.
        # Declare and get parameters (set defaults matching original behavior).
        self.declare_parameter('run_mode', 'discrete')
        self.declare_parameter('use_sim', False)
        self.declare_parameter('use_viz', False)

        run_mode = self.get_parameter('run_mode').get_parameter_value().string_value
        use_sim = self.get_parameter('use_sim').get_parameter_value().bool_value
        use_viz = self.get_parameter('use_viz').get_parameter_value().bool_value

        if run_mode not in self.run_modes:
            self.get_logger().error("Invalid run_mode {:}. Shutting down.".format(run_mode))
            raise SystemExit("Invalid run_mode.")

        self.set_global_params(run_mode, use_sim, use_viz, self.cmd_vel_pub)

        # Subscribers.
        # ROS 2: create_subscription(MsgType, topic, callback, qos)
        #self.create_subscription(Image, self.meas_topic, self.get_RGB_image, 1)
        #self.create_subscription(Odometry, "/locobot/mobile_base/odom", self.get_odom, 1)
        #self.create_subscription(LaserScan, "/locobot/scan", self.get_lidar, 1)

        #if self.use_depth_pointcloud:
        #    self.create_subscription(
         #       PointCloud2, "/locobot/camera/depth/points", self.get_pointcloud_msg, 1)
        #else:
         #   self.create_subscription(
         #       Image, "/locobot/camera/depth/image_rect_raw", self.get_RS_depth_image, 1)

        # Timer — replaces rospy.Timer.
        # ROS 2: create_timer(period_seconds, callback)
        
        self.timer = self.create_timer(self.dt, self.timer_update_loop)

        self.get_logger().info("RunnerNode initialized in mode: {}".format(run_mode))
        # Subscribers.
# Store subscription objects to prevent garbage collection in ROS 2.


        self.rgb_sub = self.create_subscription(
        Image,
        self.meas_topic,
        self.get_RGB_image,
        1
        )

        self.odom_sub = self.create_subscription(
        Odometry,
        "/locobot/mobile_base/odom",
        self.get_odom,
        1
        )

        self.lidar_sub = self.create_subscription(
        LaserScan,
        "/locobot/scan",
        self.get_lidar,
        1
        )

        if self.use_depth_pointcloud:
            self.pointcloud_sub = self.create_subscription(
                PointCloud2,
            "/locobot/camera/depth/points",
            self.get_pointcloud_msg,
            1
        )
        else:
            self.depth_sub = self.create_subscription(
                Image,
                "/locobot/camera/depth/image_rect_raw",
                self.get_RS_depth_image,
            1
        )




    # ------------------------------------------------------------------ #
    #  Timer callback                                                      #
    # ------------------------------------------------------------------ #
    def timer_update_loop(self):
        self.get_logger().info("Timer loop active")
        """
        Main loop — called at every timer tick.
        """
        
        # Update visualization if enabled.
        if self.cmn_interface.visualizer is not None:
            sim_viz_img = None
            if self.use_ground_truth_map_to_generate_observations:
                sim_viz_img = self.cmn_interface.visualizer.get_updated_img()
            cmn_viz_img = None
            if (self.cmn_interface.cmn_node is not None and
                    self.cmn_interface.cmn_node.visualizer is not None):
                cmn_viz_img = self.cmn_interface.cmn_node.visualizer.get_updated_img()

            if self.pub_viz_images:
                if sim_viz_img is not None:
                    self.sim_viz_pub.publish(self.cv_bridge.cv2_to_imgmsg(sim_viz_img))
                if cmn_viz_img is not None:
                    self.cmn_viz_pub.publish(self.cv_bridge.cv2_to_imgmsg(cmn_viz_img))
            else:
                if sim_viz_img is not None:
                    cv2.imshow('viz image', sim_viz_img)
                if cmn_viz_img is not None:
                    cv2.imshow('cmn viz image', cmn_viz_img)
                key = cv2.waitKey(int(self.dt * 1000))
                if key == 113:  # q for quit.
                    cv2.destroyAllWindows()
                    # ROS 2: use rclpy.shutdown() instead of rospy.signal_shutdown()
                    self.get_logger().info("User pressed Q key. Shutting down.")
                    rclpy.shutdown()
                    return
                elif key == 32:  # spacebar — toggle pause.
                    self.viz_paused = not self.viz_paused

                if self.viz_paused:
                    return

        # Gather panoramic RGB measurement if needed.
        pano_rgb = None
        local_occ_depth = None
        if self.cmn_interface.last_pano_rgb is not None:
            pano_rgb = self.cmn_interface.last_pano_rgb
            local_occ_depth = self.cmn_interface.last_depth_local_occ
        elif self.use_ground_truth_map_to_generate_observations:
            pass  # Sim generates local occ internally.
        elif self.use_lidar_as_ground_truth:
            pass  # LiDAR local occ is used directly.
        else:
            pano_rgb, local_occ_depth = self.get_pano_meas()

        # Update CMN node's LiDAR occ grid for viz.
        if (not self.use_ground_truth_map_to_generate_observations and
                locobot_interface.g_lidar_local_occ_meas is not None):
            self.cmn_interface.cmn_node.visualizer.lidar_local_occ_meas = \
                locobot_interface.g_lidar_local_occ_meas

        # Run one iteration (continuous or discrete).
        try:
            self.cmn_interface.run(
                pano_rgb, self.dt,
                locobot_interface.g_lidar_local_occ_meas,
                local_occ_depth
            )
        except SystemExit as e:
            self.get_logger().info("Run ended: {}".format(str(e)))
            rclpy.shutdown()

    # ------------------------------------------------------------------ #
    #  Parameter / setup helpers                                           #
    # ------------------------------------------------------------------ #
    def read_params(self):
        """
        Read configuration params from the yaml.
        """
        # ROS 2: use ament_index to find the package share directory
        pkg_path = get_package_share_directory('robo_project')
        self.yaml_path = os.path.join(pkg_path, 'config/config.yaml')
        with open(self.yaml_path, 'r') as file:
            config = yaml.safe_load(file)
            self.verbose = config["verbose"]
            self.dt = config["dt"]
            self.enable_localization = config["particle_filter"]["enable"]
            self.enable_ml_model = not config["model"]["skip_loading"]
            self.discrete_assume_yaw_is_known = config["discrete_assume_yaw_is_known"]
            # Goal cell.
            if config["manually_set_goal_cell"]:
                self.manual_goal_cell = PosePixels(config["goal_row"], config["goal_col"])
            # LiDAR / depth params.
            self.use_lidar_as_ground_truth = config["lidar"]["use_lidar_as_ground_truth"]
            self.fuse_lidar_with_rgb = config["lidar"]["fuse_lidar_with_rgb"]
            self.use_depth_as_ground_truth = config["depth"]["use_depth_as_ground_truth"]
            if self.use_depth_as_ground_truth:
                self.use_depth_pointcloud = config["depth"]["use_pointcloud"]
                if self.use_depth_pointcloud:
                    self.depth_proc_func = locobot_interface.get_local_occ_from_pointcloud
                else:
                    self.depth_proc_func = locobot_interface.get_local_occ_from_depth
            locobot_interface.read_params()
            # Measurement topic / shape.
            self.meas_topic = config["measurements"]["topic"]
            self.desired_meas_shape = (
                config["measurements"]["height"],
                config["measurements"]["width"]
            )
            # Data saving.
            self.save_training_data = config["save_data_for_training"]
            if self.save_training_data:
                self.training_data_dirpath = config["training_data_dirpath"]
                if self.training_data_dirpath[0] != "/":
                    self.training_data_dirpath = os.path.join(pkg_path, self.training_data_dirpath)
                self.training_data_dirpath = os.path.join(
                    self.training_data_dirpath, strftime("%Y%m%d-%H%M%S"))
                os.makedirs(self.training_data_dirpath, exist_ok=True)

    def set_global_params(self, run_mode: str, use_sim: bool, use_viz: bool, cmd_vel_pub=None):
        """
        Set params specified by launch arguments and initialize CMN interface.
        """
        self.run_mode = run_mode
        self.use_ground_truth_map_to_generate_observations = use_sim
        self.show_live_viz = use_viz

        config = CmnConfig()
        config.run_mode = run_mode
        config.enable_sim = use_sim
        config.enable_viz = use_viz
        config.enable_ml_model = self.enable_ml_model
        config.enable_localization = self.enable_localization
        config.use_lidar_as_ground_truth = self.use_lidar_as_ground_truth and not use_sim
        config.fuse_lidar_with_rgb = (self.fuse_lidar_with_rgb and
                                      not self.use_lidar_as_ground_truth and
                                      not use_sim and
                                      self.enable_ml_model)
        config.use_depth_as_ground_truth = (self.use_depth_as_ground_truth and
                                             not self.use_lidar_as_ground_truth and
                                             not use_sim)
        config.assume_yaw_is_known = (self.discrete_assume_yaw_is_known and
                                       "discrete" in self.run_mode)
        if self.manual_goal_cell is not None:
            config.manually_set_goal_cell = True
            config.manual_goal_cell = self.manual_goal_cell

        self.cmn_interface = CoarseMapNavInterface(config, cmd_vel_pub)
        self.cmn_interface.save_training_data = self.save_training_data
        self.cmn_interface.training_data_dirpath = self.training_data_dirpath

    # ------------------------------------------------------------------ #
    #  Sensor callbacks                                                    #
    # ------------------------------------------------------------------ #
    def get_RGB_image(self, msg: Image):
        """
        Cache the most recent RGB image from the RealSense camera.
        """
        self.most_recent_rgb_meas = msg

    def get_odom(self, msg: Odometry):
        """
        Parse an odometry message and forward it to the CMN interface.
        """
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        roll = atan2(
            2.0 * (q.x * q.y + q.w * q.z),
            q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
        )
        odom_pose = PoseMeters(x, y, roll)

        if self.first_odom is None:
            self.first_odom = odom_pose
        else:
            odom_pose.make_relative(self.first_odom)

        self.cmn_interface.set_new_odom(odom_pose)

        if self.verbose:
            self.get_logger().info("Got odom: {:}".format(odom_pose))

    def get_lidar(self, msg: LaserScan):
        """
        Process a LiDAR scan and update the obstacle flag in the motion planner.
        """
        locobot_interface.get_local_occ_from_lidar(msg)
        self.cmn_interface.motion_planner.obstacle_in_front_of_robot = \
            locobot_interface.g_lidar_detects_robot_facing_wall

    def get_RS_depth_image(self, msg: Image):
        """
        Cache the most recent depth image from the RealSense camera.
        """
        self.most_recent_depth_meas = msg

    def get_pointcloud_msg(self, msg: PointCloud2):
        """
        Cache the most recent pointcloud message.
        """
        self.most_recent_depth_meas = msg

    # ------------------------------------------------------------------ #
    #  Measurement helpers                                                 #
    # ------------------------------------------------------------------ #
    def get_pano_meas(self):
        """
        Generate a panoramic RGB measurement by pivoting in-place four times.
        @return (pano_rgb, local_occ_depth) — depth occ is None if depth is disabled.
        """
        self.get_logger().info(
            "Attempting to generate a panoramic measurement by commanding four 90 degree pivots.")
        local_occ_meas = None

        pano_meas_front = self.pop_from_RGB_buffer()
        if self.use_depth_as_ground_truth:
            local_occ_east = self.depth_proc_func(self.pop_from_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action("turn_right")
        pano_meas_right = self.pop_from_RGB_buffer()
        if self.use_depth_as_ground_truth:
            local_occ_south = self.depth_proc_func(self.pop_from_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action("turn_right")
        pano_meas_back = self.pop_from_RGB_buffer()
        if self.use_depth_as_ground_truth:
            local_occ_west = self.depth_proc_func(self.pop_from_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action("turn_right")
        pano_meas_left = self.pop_from_RGB_buffer()
        if self.use_depth_as_ground_truth:
            local_occ_north = self.depth_proc_func(self.pop_from_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action("turn_right")

        pano_rgb = np.concatenate([
            pano_meas_front[:, :, 0:3],
            pano_meas_right[:, :, 0:3],
            pano_meas_back[:, :, 0:3],
            pano_meas_left[:, :, 0:3]
        ], axis=1)
        pano_rgb = cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR)

        if self.use_depth_as_ground_truth:
            rotated_local_occ_south = rotate_image_to_north(local_occ_south, 0)
            rotated_local_occ_west = rotate_image_to_north(local_occ_west, -np.pi / 2)
            rotated_local_occ_north = rotate_image_to_north(local_occ_north, np.pi)
            local_occ_meas = np.min(
                [local_occ_east, rotated_local_occ_south,
                 rotated_local_occ_west, rotated_local_occ_north], axis=0)

            div = 3
            one_third = local_occ_meas.shape[0] // div
            two_thirds = (div - 1) * local_occ_meas.shape[0] // div
            occ_thresh = 0.1

            def fill_if_occupied(block_slice):
                if 1 - np.mean(local_occ_meas[block_slice]) >= occ_thresh:
                    local_occ_meas[block_slice] = 0

            fill_if_occupied(np.s_[:one_third, :one_third])
            fill_if_occupied(np.s_[:one_third, two_thirds:])
            fill_if_occupied(np.s_[two_thirds:, two_thirds:])
            fill_if_occupied(np.s_[two_thirds:, :one_third])

        return pano_rgb, local_occ_meas

    def pop_from_RGB_buffer(self):
        """
        Block until a new RGB image is available, then return and clear it.
        """
        # ROS 2: busy-wait with rclpy.spin_once() instead of rospy.sleep()
        while self.most_recent_rgb_meas is None:
            self.get_logger().warn("Waiting on RGB measurement from RealSense!")
            rclpy.spin_once(self, timeout_sec=0.01)

        cv_img_meas = self.cv_bridge.imgmsg_to_cv2(
            self.most_recent_rgb_meas, desired_encoding='passthrough')
        self.most_recent_rgb_meas = None  # Consume the measurement.

        if self.verbose:
            self.get_logger().info("Trying to resize image from shape {:} to {:}".format(
                cv_img_meas.shape, self.desired_meas_shape))
        cv_img_meas = cv2.resize(cv_img_meas, self.desired_meas_shape)
        return cv_img_meas

    def pop_from_depth_buffer(self):
        """
        Block until a new depth measurement is available, then return and clear it.
        """
        self.most_recent_depth_meas = None
        while self.most_recent_depth_meas is None:
            self.get_logger().warn("Waiting on depth measurement from RealSense!")
            rclpy.spin_once(self, timeout_sec=0.5)
        return self.most_recent_depth_meas


def main(args=None):
    # ROS 2: rclpy.init() instead of rospy.init_node()
    rclpy.init(args=args)
    try:
        node = RunnerNode()
        # ROS 2: rclpy.spin() instead of rospy.spin()
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
