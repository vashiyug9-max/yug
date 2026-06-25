#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry

import yaml, os, time
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from math import atan2
import numpy as np
import cv2
from time import strftime
from typing import Callable

from robo_project.scripts.cmn_interface import CoarseMapNavInterface, CmnConfig
from robo_project.scripts.basic_types import PoseMeters, PosePixels, rotate_image_to_north
from robo_project import locobot_interface

TOPIC_CMD_VEL = '/cmd_vel'
TOPIC_ODOM    = '/odom'
TOPIC_CAMERA  = '/camera/front/image_raw'
TOPIC_LIDAR   = '/scan'

class RunnerNode(Node):
    def __init__(self):
        super().__init__('runner_node')

        self.cv_bridge = CvBridge()
        self.cmn_interface: CoarseMapNavInterface = None

        self.most_recent_rgb_meas = None
        self.most_recent_depth_meas = None
        self.desired_meas_shape = None
        self.depth_proc_func: Callable = None

        self.first_odom: PoseMeters = None
        self.ml_estimated_pose: PoseMeters = None          # ── [ML]

        self.run_modes = ['continuous', 'discrete', 'discrete_random']
        self.run_mode = None
        self.use_ground_truth_map_to_generate_observations = False
        self.verbose = False
        self.use_lidar_as_ground_truth = False
        self.manual_goal_cell: PosePixels = None
        self.use_depth_pointcloud = False
        self.save_training_data = False                    # ── [ML]
        self.training_data_dirpath = None                  # ── [ML]
        self.viz_paused = False
        self.pub_viz_images = False

        self.read_params()

        self.cmd_vel_pub = self.create_publisher(Twist, TOPIC_CMD_VEL, 1)
        self.sim_viz_pub = self.create_publisher(Image, '/cmn/viz/sim', 1)
        self.cmn_viz_pub = self.create_publisher(Image, '/cmn/viz/cmn', 1)

        self.declare_parameter('run_mode', 'discrete')
        self.declare_parameter('use_sim',  False)
        self.declare_parameter('use_viz',  False)

        run_mode = self.get_parameter('run_mode').get_parameter_value().string_value
        use_sim  = self.get_parameter('use_sim').get_parameter_value().bool_value
        use_viz  = self.get_parameter('use_viz').get_parameter_value().bool_value

        if run_mode not in self.run_modes:
            self.get_logger().error('Invalid run_mode: {}. Shutting down.'.format(run_mode))
            raise SystemExit('Invalid run_mode.')

        self.set_global_params(run_mode, use_sim, use_viz)

        self.cb_group_timer = MutuallyExclusiveCallbackGroup()
        self.cb_group_subs = ReentrantCallbackGroup()

        self.odom_sub  = self.create_subscription(Odometry, TOPIC_ODOM, self.get_odom, 10, callback_group=self.cb_group_subs)
        self.rgb_sub   = self.create_subscription(Image, TOPIC_CAMERA, self.get_rgb_image, 1, callback_group=self.cb_group_subs)
        self.lidar_sub = self.create_subscription(LaserScan, TOPIC_LIDAR, self.get_lidar, 1, callback_group=self.cb_group_subs)
        # ── [ML] pose estimate from the ML node, fused into odom in get_odom()
        self.ml_estimated_pose_sub = self.create_subscription(
            Point, '/ml_estimated_pose', self.get_ml_estimated_pose, 10, callback_group=self.cb_group_subs)

        self.timer = self.create_timer(self.dt, self.timer_update_loop, callback_group=self.cb_group_timer)

        self.get_logger().info('RunnerNode started — vinebot, mode: {}'.format(run_mode))

    def _publish_cmd_vel(self, fwd: float, ang: float):
        msg = Twist()
        msg.linear.x  = float(fwd)
        msg.angular.z = float(ang)
        self.cmd_vel_pub.publish(msg)

    def publish(self, twist_msg):
        # --- FIXED: Perfect 10-Tick Math Synchronization ---
        is_turn = abs(twist_msg.angular.z) > 0.01
        is_move = abs(twist_msg.linear.x) > 0.01

        if is_turn:
            self.get_logger().info("🤖 AI COMMAND: Perfect 90-degree turn.")
            cmd = Twist()
            import math
            cmd.angular.z = math.copysign(0.5, twist_msg.angular.z)
            steps = 10  # 10 ticks * 9.0 deg = 90 degrees
            for _ in range(steps):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        elif is_move:
            self.get_logger().info("🤖 AI COMMAND: Perfect grid step forward.")
            cmd = Twist()
            import math
            cmd.linear.x = math.copysign(0.5, twist_msg.linear.x)
            steps = 10 # 10 ticks * 0.05m = 0.5 meters
            for _ in range(steps):
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)

        # Slam the brakes and let the camera settle for the AI
        self.cmd_vel_pub.publish(Twist())
        time.sleep(0.5)

    def timer_update_loop(self):
        if self.cmn_interface is None: return

        if self.cmn_interface.visualizer is not None:
            sim_viz_img = None
            cmn_viz_img = None

            if self.use_ground_truth_map_to_generate_observations:
                sim_viz_img = self.cmn_interface.visualizer.get_updated_img()

            if (self.cmn_interface.cmn_node is not None and
                    self.cmn_interface.cmn_node.visualizer is not None):
                cmn_viz_img = self.cmn_interface.cmn_node.visualizer.get_updated_img()

            if self.pub_viz_images:
                if sim_viz_img is not None:
                    self.sim_viz_pub.publish(self.cv_bridge.cv2_to_imgmsg(sim_viz_img))
                if cmn_viz_img is not None:
                    self.cmn_viz_pub.publish(self.cv_bridge.cv2_to_imgmsg(cmn_viz_img))
            else:
                if sim_viz_img is not None: cv2.imshow('sim viz', sim_viz_img)
                if cmn_viz_img is not None: cv2.imshow('cmn viz', cmn_viz_img)
                key = cv2.waitKey(int(self.dt * 1000))
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    rclpy.shutdown()
                    return

        pano_rgb = None
        local_occ_depth = None

        if self.cmn_interface.last_pano_rgb is not None:
            pano_rgb = self.cmn_interface.last_pano_rgb
            local_occ_depth = self.cmn_interface.last_depth_local_occ
        elif self.use_ground_truth_map_to_generate_observations:
            pass
        elif self.use_lidar_as_ground_truth:
            pass
        else:
            pano_rgb, local_occ_depth = self.get_pano_meas()

        # ── [ML] forward lidar-derived local occupancy into the CMN visualizer
        if (locobot_interface.g_lidar_local_occ_meas is not None and
                self.cmn_interface.cmn_node is not None and
                self.cmn_interface.cmn_node.visualizer is not None):
            self.cmn_interface.cmn_node.visualizer.lidar_local_occ_meas = \
                locobot_interface.g_lidar_local_occ_meas

        try:
            self.cmn_interface.run(
                pano_rgb, self.dt,
                locobot_interface.g_lidar_local_occ_meas, local_occ_depth)
        except SystemExit as e:
            self.get_logger().info('Run ended: {}'.format(str(e)))
            rclpy.shutdown()

    def read_params(self):
        pkg_path = get_package_share_directory('robo_project')
        self.yaml_path = os.path.join(pkg_path, 'config/config.yaml')
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            self.verbose                      = config['verbose']
            self.dt                           = config['dt']
            self.enable_localization          = config['particle_filter']['enable']
            self.enable_ml_model              = not config['model']['skip_loading']
            self.discrete_assume_yaw_is_known = config['discrete_assume_yaw_is_known']
            if config.get('manually_set_goal_cell', False):
                self.manual_goal_cell = PosePixels(config['goal_row'], config['goal_col'])
            self.use_lidar_as_ground_truth = config['lidar']['use_lidar_as_ground_truth']
            self.fuse_lidar_with_rgb       = config['lidar']['fuse_lidar_with_rgb']
            self.use_depth_as_ground_truth = config['depth']['use_depth_as_ground_truth']
            if self.use_depth_as_ground_truth:
                self.use_depth_pointcloud = config['depth']['use_pointcloud']
                self.depth_proc_func = (locobot_interface.get_local_occ_from_pointcloud if self.use_depth_pointcloud else locobot_interface.get_local_occ_from_depth)
            locobot_interface.read_params()
            self.desired_meas_shape = (config['measurements']['height'], config['measurements']['width'])

            # ── [ML] training-data capture settings
            self.save_training_data = config.get('save_data_for_training', False)
            if self.save_training_data:
                dirpath = config.get('training_data_dirpath', 'data')
                if not dirpath.startswith('/'):
                    dirpath = os.path.join(pkg_path, dirpath)
                self.training_data_dirpath = os.path.join(dirpath, strftime('%Y%m%d-%H%M%S'))
                os.makedirs(self.training_data_dirpath, exist_ok=True)

    def set_global_params(self, run_mode: str, use_sim: bool = False, use_viz: bool = False):
        self.run_mode = run_mode
        self.use_ground_truth_map_to_generate_observations = use_sim
        config = CmnConfig()
        config.run_mode            = run_mode
        config.enable_sim          = use_sim
        config.enable_viz          = use_viz
        config.enable_ml_model     = self.enable_ml_model
        config.enable_localization = self.enable_localization
        config.use_lidar_as_ground_truth = (self.use_lidar_as_ground_truth and not use_sim)
        config.fuse_lidar_with_rgb = (self.fuse_lidar_with_rgb and not self.use_lidar_as_ground_truth and not use_sim and self.enable_ml_model)
        config.use_depth_as_ground_truth = (self.use_depth_as_ground_truth and not self.use_lidar_as_ground_truth and not use_sim)
        config.assume_yaw_is_known = (self.discrete_assume_yaw_is_known and 'discrete' in run_mode)
        if self.manual_goal_cell is not None:
            config.manually_set_goal_cell = True
            config.manual_goal_cell = self.manual_goal_cell
        self.cmn_interface = CoarseMapNavInterface(config, self)
        # ── [ML] wire training-data capture settings into the CMN interface
        self.cmn_interface.save_training_data = self.save_training_data
        self.cmn_interface.training_data_dirpath = self.training_data_dirpath

    def get_rgb_image(self, msg: Image): self.most_recent_rgb_meas = msg

    def get_lidar(self, msg: LaserScan):
        locobot_interface.get_local_occ_from_lidar(msg)
        # ── [ML] feed lidar-detected obstacle state to the motion planner
        if self.cmn_interface is not None:
            self.cmn_interface.motion_planner.obstacle_in_front_of_robot = \
                locobot_interface.g_lidar_detects_robot_facing_wall

    def get_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        odom_pose = PoseMeters(x, y, yaw)
        if self.first_odom is None:
            self.first_odom = odom_pose
        else:
            odom_pose.make_relative(self.first_odom)

        if self.cmn_interface is not None:
            # ── [ML] weighted fusion of raw odom with the ML-estimated pose
            if self.ml_estimated_pose is not None:
                alpha = 0.7
                fused_x = alpha * odom_pose.x + (1.0 - alpha) * self.ml_estimated_pose.x
                fused_y = alpha * odom_pose.y + (1.0 - alpha) * self.ml_estimated_pose.y
                fused_pose = PoseMeters(fused_x, fused_y, odom_pose.yaw)
                self.cmn_interface.set_new_odom(fused_pose)
                if self.verbose:
                    self.get_logger().info(f"Fused Pose: x={fused_x:.3f}, y={fused_y:.3f}")
            else:
                self.cmn_interface.set_new_odom(odom_pose)

        if self.verbose:
            self.get_logger().info('Odom: {}'.format(odom_pose))

    # ── [ML] receives the ML pose estimate, consumed by get_odom() above
    def get_ml_estimated_pose(self, msg: Point):
        self.ml_estimated_pose = PoseMeters(msg.x, msg.y, 0.0)
        if self.verbose:
            self.get_logger().info(f"ML Pose: x={msg.x:.3f}, y={msg.y:.3f}")

    def _turn_90_degrees_right(self):
        self.get_logger().info('Physically turning 90 degrees right...')
        cmd = Twist()
        cmd.angular.z = -0.5
        steps = 10 # 10 ticks * 9.0 deg = 90 degrees
        for _ in range(steps):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.1)
        self.cmd_vel_pub.publish(Twist()) # Stop
        time.sleep(0.5) # Wait for camera to settle

    def get_pano_meas(self):
        self.get_logger().info('Building panoramic measurement via four pivots.')
        local_occ_meas = None
        pano_front = self._pop_rgb_buffer()

        self._turn_90_degrees_right()
        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_right = self._pop_rgb_buffer()

        self._turn_90_degrees_right()
        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_back = self._pop_rgb_buffer()

        self._turn_90_degrees_right()
        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_left = self._pop_rgb_buffer()

        self._turn_90_degrees_right()
        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')

        pano_rgb = np.concatenate([pano_front[:, :, 0:3], pano_right[:, :, 0:3], pano_back[:, :, 0:3], pano_left[:, :, 0:3]], axis=1)
        pano_rgb = cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR)
        return pano_rgb, local_occ_meas

    def _pop_rgb_buffer(self):
        self.most_recent_rgb_meas = None
        while self.most_recent_rgb_meas is None: 
            time.sleep(0.05)
        img = self.cv_bridge.imgmsg_to_cv2(self.most_recent_rgb_meas, desired_encoding='passthrough')
        return cv2.resize(img, self.desired_meas_shape)

def main(args=None):
    rclpy.init(args=args)
    node = RunnerNode()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
