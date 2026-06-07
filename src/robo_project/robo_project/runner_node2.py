#!/usr/bin/env python3

"""
Main node for running the CMN project with the vinebot simulation.

Vinebot ROS topics (via ros_gz_bridge):
  subscribe:  /diff_cont/odom          [nav_msgs/msg/Odometry]
  subscribe:  /camera/image_raw        [sensor_msgs/msg/Image]
  subscribe:  /scan                    [sensor_msgs/msg/LaserScan]
  publish:    /diff_cont/cmd_vel       [geometry_msgs/msg/TwistStamped]  ← TwistStamped!

NOTE: The gz_bridge maps /diff_cont/cmd_vel (TwistStamped) → Ignition cmd_vel (Twist).
      Do NOT publish plain Twist or /diff_cont/cmd_vel_unstamped — that bypasses the bridge.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry

import yaml, os
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


# ── Vinebot topic names ───────────────────────────────────────────────────────
TOPIC_CMD_VEL = '/diff_cont/cmd_vel'       # TwistStamped → ros_gz_bridge → Ignition
TOPIC_ODOM    = '/diff_cont/odom'          # Odometry from diff_drive controller
TOPIC_CAMERA  = '/camera/image_raw'        # RGB from D455 (bridged from Ignition)
TOPIC_LIDAR   = '/scan'                    # LaserScan from Livox (bridged)
# ─────────────────────────────────────────────────────────────────────────────


class RunnerNode(Node):
    """
    Main ROS 2 node for the CMN project, wired to the vinebot simulation.
    """

    def __init__(self):
        super().__init__('runner_node')

        self.cv_bridge = CvBridge()
        self.cmn_interface: CoarseMapNavInterface = None

        # Measurement buffers.
        self.most_recent_rgb_meas = None
        self.most_recent_depth_meas = None
        self.desired_meas_shape = None
        self.depth_proc_func: Callable = None

        # Odometry state.
        self.first_odom: PoseMeters = None

        # Config flags.
        self.run_modes = ['continuous', 'discrete', 'discrete_random']
        self.run_mode = None
        self.use_ground_truth_map_to_generate_observations = False
        self.verbose = False
        self.use_lidar_as_ground_truth = False
        self.manual_goal_cell: PosePixels = None
        self.use_depth_pointcloud = False
        self.save_training_data = False
        self.training_data_dirpath = None
        self.viz_paused = False
        self.pub_viz_images = False

        self.read_params()

        # ── Publishers ────────────────────────────────────────────────────────
        # TwistStamped — required by the ros_gz_bridge config
        self.cmd_vel_pub = self.create_publisher(TwistStamped, TOPIC_CMD_VEL, 1)
        self.sim_viz_pub = self.create_publisher(Image, '/cmn/viz/sim', 1)
        self.cmn_viz_pub = self.create_publisher(Image, '/cmn/viz/cmn', 1)

        # ── Launch parameters ─────────────────────────────────────────────────
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

        # ── Subscribers ───────────────────────────────────────────────────────
        self.odom_sub  = self.create_subscription(
            Odometry,  TOPIC_ODOM,   self.get_odom,      10)
        self.rgb_sub   = self.create_subscription(
            Image,     TOPIC_CAMERA, self.get_rgb_image,  1)
        self.lidar_sub = self.create_subscription(
            LaserScan, TOPIC_LIDAR,  self.get_lidar,      1)

        # ── Main timer ────────────────────────────────────────────────────────
        self.timer = self.create_timer(self.dt, self.timer_update_loop)

        self.get_logger().info(
            'RunnerNode started — vinebot, mode: {}'.format(run_mode))

    # ── cmd_vel helper ────────────────────────────────────────────────────────
    def _publish_cmd_vel(self, fwd: float, ang: float):
        """
        Publish a TwistStamped velocity command to the vinebot.
        The ros_gz_bridge strips the header and forwards the twist to Ignition.
        """
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x  = float(fwd)
        msg.twist.linear.y  = 0.0
        msg.twist.linear.z  = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(ang)
        self.cmd_vel_pub.publish(msg)

    # ── Timer callback ────────────────────────────────────────────────────────
    def timer_update_loop(self):
        """Main loop called at every timer tick."""

        if self.cmn_interface is None:
            return

        # Visualization.
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
                    self.sim_viz_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(sim_viz_img))
                if cmn_viz_img is not None:
                    self.cmn_viz_pub.publish(
                        self.cv_bridge.cv2_to_imgmsg(cmn_viz_img))
            else:
                if sim_viz_img is not None:
                    cv2.imshow('sim viz', sim_viz_img)
                if cmn_viz_img is not None:
                    cv2.imshow('cmn viz', cmn_viz_img)
                key = cv2.waitKey(int(self.dt * 1000))
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    self.get_logger().info('User pressed Q. Shutting down.')
                    rclpy.shutdown()
                    return
                elif key == ord(' '):
                    self.viz_paused = not self.viz_paused
                if self.viz_paused:
                    return

        # Gather panoramic RGB.
        pano_rgb = None
        local_occ_depth = None

        if self.cmn_interface.last_pano_rgb is not None:
            pano_rgb = self.cmn_interface.last_pano_rgb
            local_occ_depth = self.cmn_interface.last_depth_local_occ
        elif self.use_ground_truth_map_to_generate_observations:
            # Sim generates observations internally — no camera needed
            self.get_logger().info("Using sim ground truth — skipping camera")
        elif self.use_lidar_as_ground_truth:
            # Lidar local occ passed directly to cmn_interface.run()
            self.get_logger().info("Using lidar ground truth")
        else:
            pano_rgb, local_occ_depth = self.get_pano_meas()

        # Forward lidar occ to CMN visualizer.
        if (locobot_interface.g_lidar_local_occ_meas is not None and
                self.cmn_interface.cmn_node is not None and
                self.cmn_interface.cmn_node.visualizer is not None):
            self.cmn_interface.cmn_node.visualizer.lidar_local_occ_meas = \
                locobot_interface.g_lidar_local_occ_meas

        try:
            self.cmn_interface.run(
                pano_rgb,
                self.dt,
                locobot_interface.g_lidar_local_occ_meas,
                local_occ_depth,
            )
        except SystemExit as e:
            self.get_logger().info('Run ended: {}'.format(str(e)))
            rclpy.shutdown()

    # ── Setup helpers ─────────────────────────────────────────────────────────
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
                self.manual_goal_cell = PosePixels(
                    config['goal_row'], config['goal_col'])

            self.use_lidar_as_ground_truth = config['lidar']['use_lidar_as_ground_truth']
            self.fuse_lidar_with_rgb       = config['lidar']['fuse_lidar_with_rgb']
            self.use_depth_as_ground_truth = config['depth']['use_depth_as_ground_truth']

            if self.use_depth_as_ground_truth:
                self.use_depth_pointcloud = config['depth']['use_pointcloud']
                self.depth_proc_func = (
                    locobot_interface.get_local_occ_from_pointcloud
                    if self.use_depth_pointcloud
                    else locobot_interface.get_local_occ_from_depth
                )

            locobot_interface.read_params()

            self.desired_meas_shape = (
                config['measurements']['height'],
                config['measurements']['width'],
            )

            self.save_training_data = config.get('save_data_for_training', False)
            if self.save_training_data:
                dirpath = config.get('training_data_dirpath', 'data')
                if not dirpath.startswith('/'):
                    dirpath = os.path.join(pkg_path, dirpath)
                self.training_data_dirpath = os.path.join(
                    dirpath, strftime('%Y%m%d-%H%M%S'))
                os.makedirs(self.training_data_dirpath, exist_ok=True)

    def set_global_params(self, run_mode: str, use_sim: bool = False,
                          use_viz: bool = False):
        self.run_mode = run_mode
        self.use_ground_truth_map_to_generate_observations = use_sim

        config = CmnConfig()
        config.run_mode            = run_mode
        config.enable_sim          = use_sim
        config.enable_viz          = use_viz
        config.enable_ml_model     = self.enable_ml_model
        config.enable_localization = self.enable_localization
        config.use_lidar_as_ground_truth = (
            self.use_lidar_as_ground_truth and not use_sim)
        config.fuse_lidar_with_rgb = (
            self.fuse_lidar_with_rgb
            and not self.use_lidar_as_ground_truth
            and not use_sim
            and self.enable_ml_model)
        config.use_depth_as_ground_truth = (
            self.use_depth_as_ground_truth
            and not self.use_lidar_as_ground_truth
            and not use_sim)
        config.assume_yaw_is_known = (
            self.discrete_assume_yaw_is_known and 'discrete' in run_mode)

        if self.manual_goal_cell is not None:
            config.manually_set_goal_cell = True
            config.manual_goal_cell = self.manual_goal_cell

        # Pass a wrapped publisher that sends TwistStamped
        self.cmn_interface = CoarseMapNavInterface(config, self._twisted_pub_wrapper())
        self.cmn_interface.save_training_data = self.save_training_data
        self.cmn_interface.training_data_dirpath = self.training_data_dirpath

    def _twisted_pub_wrapper(self):
        """
        Returns a fake publisher object that motion_planner.set_vel_pub() accepts.
        motion_planner calls pub.publish(Twist msg) — we intercept and upgrade
        to TwistStamped before sending to the vinebot bridge.
        """
        node = self

        class _TwistStampedAdapter:
            def publish(self, twist_msg):
                node._publish_cmd_vel(
                    twist_msg.linear.x,
                    twist_msg.angular.z
                )

        return _TwistStampedAdapter()

    # ── Sensor callbacks ──────────────────────────────────────────────────────
    def get_rgb_image(self, msg: Image):
        self.most_recent_rgb_meas = msg

    def get_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation

        # Correct yaw from quaternion
        yaw = atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        odom_pose = PoseMeters(x, y, yaw)

        if self.first_odom is None:
            self.first_odom = odom_pose
        else:
            odom_pose.make_relative(self.first_odom)

        if self.cmn_interface is not None:
            self.cmn_interface.set_new_odom(odom_pose)

        if self.verbose:
            self.get_logger().info('Odom: {}'.format(odom_pose))

    def get_lidar(self, msg: LaserScan):
        locobot_interface.get_local_occ_from_lidar(msg)
        if self.cmn_interface is not None:
            self.cmn_interface.motion_planner.obstacle_in_front_of_robot = \
                locobot_interface.g_lidar_detects_robot_facing_wall

    # ── Panoramic measurement ─────────────────────────────────────────────────
    def get_pano_meas(self):
        """
        Build panoramic RGB by pivoting the robot four times.
        Only used in discrete mode with the real robot responding to cmd_vel.
        """
        self.get_logger().info('Building panoramic measurement via four pivots.')
        local_occ_meas = None

        pano_front = self._pop_rgb_buffer()
        if self.use_depth_as_ground_truth:
            occ_east = self.depth_proc_func(self._pop_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_right = self._pop_rgb_buffer()
        if self.use_depth_as_ground_truth:
            occ_south = self.depth_proc_func(self._pop_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_back = self._pop_rgb_buffer()
        if self.use_depth_as_ground_truth:
            occ_west = self.depth_proc_func(self._pop_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')
        pano_left = self._pop_rgb_buffer()
        if self.use_depth_as_ground_truth:
            occ_north = self.depth_proc_func(self._pop_depth_buffer())

        self.cmn_interface.motion_planner.cmd_discrete_action('turn_right')

        pano_rgb = np.concatenate([
            pano_front[:, :, 0:3],
            pano_right[:, :, 0:3],
            pano_back[:, :, 0:3],
            pano_left[:, :, 0:3],
        ], axis=1)
        pano_rgb = cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR)

        if self.use_depth_as_ground_truth:
            rot_south = rotate_image_to_north(occ_south, 0)
            rot_west  = rotate_image_to_north(occ_west, -np.pi / 2)
            rot_north = rotate_image_to_north(occ_north,  np.pi)
            local_occ_meas = np.min(
                [occ_east, rot_south, rot_west, rot_north], axis=0)

        return pano_rgb, local_occ_meas

    def _pop_rgb_buffer(self):
        self.most_recent_rgb_meas = None
        timeout, elapsed, interval = 5.0, 0.0, 0.05
        while self.most_recent_rgb_meas is None:
            rclpy.spin_once(self, timeout_sec=interval)
            elapsed += interval
            if elapsed > timeout:
                self.get_logger().error(
                    'Timeout waiting for image on {}'.format(TOPIC_CAMERA))
                return np.zeros(
                    (self.desired_meas_shape[0], self.desired_meas_shape[1], 3),
                    dtype=np.uint8)
        img = self.cv_bridge.imgmsg_to_cv2(
            self.most_recent_rgb_meas, desired_encoding='passthrough')
        self.most_recent_rgb_meas = None
        return cv2.resize(img, self.desired_meas_shape)

    def _pop_depth_buffer(self):
        self.most_recent_depth_meas = None
        timeout, elapsed, interval = 5.0, 0.0, 0.05
        while self.most_recent_depth_meas is None:
            rclpy.spin_once(self, timeout_sec=interval)
            elapsed += interval
            if elapsed > timeout:
                self.get_logger().error('Timeout waiting for depth frame.')
                return None
        msg = self.most_recent_depth_meas
        self.most_recent_depth_meas = None
        return msg


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RunnerNode()
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
