#!/usr/bin/env python3

"""
Wrapper for the original CMN Habitat code from Chengguang Xu to work with my custom simulator or a physical robot.
Migrated from ROS 1 to ROS 2.
"""

import numpy as np
import cv2, os

# ROS 2: use rclpy logger instead of rospy
import rclpy.logging

from robo_project.scripts.basic_types import PoseMeters, PosePixels
from robo_project.scripts.map_handler import Simulator, MapFrameManager
from robo_project.scripts.motion_planner import DiscreteMotionPlanner, MotionPlanner
from robo_project.scripts.particle_filter import ParticleFilter
from robo_project.scripts.visualizer import Visualizer
from robo_project.scripts.cmn.cmn_ported import CoarseMapNavDiscrete

# Module-level logger
_logger = rclpy.logging.get_logger('cmn_interface')


class CmnConfig():
    # Flag to track whether we are using discrete or continuous state/action space.
    # Should be one of ["continuous", "discrete", "discrete_random"].
    run_mode: str = "continuous"
    enable_sim: bool = False
    enable_viz: bool = False
    # Debug flag: disable ML model loading/running (can't run on all machines).
    enable_ml_model: bool = False
    # Debug flag: disable localization, use ground truth pose for planning.
    enable_localization: bool = True
    use_lidar_as_ground_truth: bool = False
    # Fuse LiDAR local occ with RGB model prediction. Mutually exclusive with
    # use_lidar_as_ground_truth and enable_sim. Requires enable_ml_model.
    fuse_lidar_with_rgb: bool = False
    use_depth_as_ground_truth: bool = False
    # Original CMN does not estimate yaw — track it manually to uphold this assumption.
    # Setting False allows CMN to estimate between the four cardinal directions.
    assume_yaw_is_known: bool = True
    manually_set_goal_cell: bool = False
    manual_goal_cell: PosePixels = None  # Only used if manually_set_goal_cell is True.


class CoarseMapNavInterface():
    enable_sim: bool = False
    enable_viz: bool = False
    use_discrete_space: bool = False
    enable_localization: bool = True
    use_lidar_as_ground_truth: bool = False
    fuse_lidar_with_rgb: bool = False
    use_depth_as_ground_truth: bool = False
    assume_yaw_is_known: bool = True

    cmn_node: CoarseMapNavDiscrete = None
    map_frame_manager = None  # MapFrameManager or Simulator(MapFrameManager)
    particle_filter: ParticleFilter = None
    motion_planner = None  # MotionPlanner or DiscreteMotionPlanner(MotionPlanner)
    visualizer: Visualizer = None

    veh_pose_estimate_meters: PoseMeters = PoseMeters(0, 0, 0)
    last_pano_rgb: np.ndarray = None
    last_depth_local_occ: np.ndarray = None

    save_training_data: bool = False
    training_data_dirpath: str = None
    iteration: int = 0

    def __init__(self, config: CmnConfig, cmd_vel_pub):
        """
        Initialize all modules needed for this project.
        @param config - CmnConfig instance with relevant params/flags.
        @param cmd_vel_pub - ROS 2 publisher for Twist velocities.
        """
        self.enable_sim = config.enable_sim
        self.use_discrete_space = "discrete" in config.run_mode
        self.enable_viz = config.enable_viz
        self.enable_localization = config.enable_localization and config.enable_sim
        self.use_lidar_as_ground_truth = config.use_lidar_as_ground_truth
        self.fuse_lidar_with_rgb = config.fuse_lidar_with_rgb
        self.use_depth_as_ground_truth = config.use_depth_as_ground_truth
        self.assume_yaw_is_known = config.assume_yaw_is_known

        # Init the map manager / simulator.
        if self.enable_sim:
            self.map_frame_manager = Simulator(self.use_discrete_space)
        else:
            self.map_frame_manager = MapFrameManager(self.use_discrete_space)

        # Init the motion planner.
        if self.use_discrete_space:
            self.motion_planner = DiscreteMotionPlanner()
        else:
            self.motion_planner = MotionPlanner()
        self.motion_planner.set_vel_pub(cmd_vel_pub)
        self.motion_planner.set_map_frame_manager(self.map_frame_manager)
        # Discrete motion commands publish velocity commands and wait for completion —
        # this cannot run without a real robot (i.e., not in sim).
        self.motion_planner.wait_for_motion_to_complete = not self.enable_sim

        # Choose the first goal cell.
        if config.manually_set_goal_cell:
            self.motion_planner.set_goal_point(config.manual_goal_cell)
        else:
            self.motion_planner.set_goal_point_random()

        if not self.use_discrete_space:
            self.particle_filter = ParticleFilter()
            self.particle_filter.set_map_frame_manager(self.map_frame_manager)

        if self.use_discrete_space or not self.enable_sim:
            self.cmn_node = CoarseMapNavDiscrete(
                self.map_frame_manager,
                not config.enable_ml_model,
                "random" in config.run_mode,
                self.assume_yaw_is_known
            )
            self.cmn_node.set_goal_cell(self.motion_planner.goal_pos_px)
            self.cmn_node.enable_sim = self.enable_sim
            self.cmn_node.fuse_lidar_with_rgb = self.fuse_lidar_with_rgb
            self.cmn_node.visualizer.gt_is_depth = self.use_depth_as_ground_truth

        if self.enable_viz:
            self.visualizer = Visualizer()
            self.visualizer.set_map_frame_manager(self.map_frame_manager)
            self.visualizer.goal_cell = self.motion_planner.goal_pos_px

    def run(self, pano_rgb=None, dt: float = None, lidar_local_occ_meas=None, depth_local_occ_meas=None):
        """
        Run one iteration.
        @param pano_rgb - Numpy array of four color images concatenated horizontally (front, right, back, left).
        @param dt - Timer period in seconds. Used for particle filter propagation.
        @param lidar_local_occ_meas - Local occupancy measurement from LiDAR.
        @param depth_local_occ_meas - Local occupancy measurement from depth sensor.
        """
        current_local_map = None

        if self.enable_sim:
            current_local_map, rect = self.map_frame_manager.get_true_observation()
            if self.enable_viz:
                self.visualizer.set_observation(current_local_map, rect)
        elif self.use_lidar_as_ground_truth:
            current_local_map = lidar_local_occ_meas
        elif self.use_depth_as_ground_truth:
            current_local_map = depth_local_occ_meas

        if not self.use_discrete_space:
            if current_local_map is None:
                current_local_map = self.compute_observation_continuous(pano_rgb)
                if self.enable_viz:
                    self.visualizer.set_observation(current_local_map)
            self.run_particle_filter(current_local_map)
            self.command_motion_continuous(dt)

        else:
            if current_local_map is not None:
                # Ground-truth observation is robot-relative (facing east); rotate to global north for CMN.
                current_local_map = np.rot90(current_local_map, k=1)

            if self.assume_yaw_is_known:
                if self.enable_sim:
                    agent_yaw = self.map_frame_manager.veh_pose_true_px.yaw
                else:
                    agent_yaw = self.veh_pose_estimate_meters.yaw
            else:
                agent_yaw = None

            if pano_rgb is None and current_local_map is None:
                # ROS 2: use rclpy logger
                _logger.error("Need pano_rgb or ground truth observation to run CMN.")
                return

            self.cmn_node.predict_local_occupancy(pano_rgb, agent_yaw, current_local_map, lidar_local_occ_meas)

            plan_from_true_pose: bool = False
            if self.enable_sim and plan_from_true_pose:
                action = self.cmn_node.choose_next_action(agent_yaw, self.map_frame_manager.veh_pose_true_px)
            else:
                action = self.cmn_node.choose_next_action(agent_yaw)

            if action == "goal_reached":
                if self.motion_planner.move_goal_after_reaching:
                    _logger.warn("CMN: Goal reached, so choosing a new goal cell to continue the run.")
                    self.motion_planner.set_goal_point_random()
                    self.cmn_node.set_goal_cell(self.motion_planner.goal_pos_px)
                    if self.enable_viz:
                        self.visualizer.goal_cell = self.motion_planner.goal_pos_px
                else:
                    _logger.warn("CMN: Goal reached, so terminating run.")
                    # ROS 2: avoid exit() — raise an exception so the node can shut down cleanly.
                    raise SystemExit("Goal reached.")

            facing_a_wall: bool = self.cmn_node.is_facing_a_wall_in_pred_local_occ
            self.cmn_node.update_beliefs(action, agent_yaw, facing_a_wall)

            if pano_rgb is not None:
                width_each_img = pano_rgb.shape[1] // 4
                if action == "move_forward":
                    self.last_pano_rgb = None
                    self.last_depth_local_occ = None
                elif action == "turn_right":
                    self.last_pano_rgb = np.roll(pano_rgb, shift=-width_each_img, axis=1)
                    if depth_local_occ_meas is not None:
                        self.last_depth_local_occ = np.rot90(depth_local_occ_meas, k=1)
                elif action == "turn_left":
                    self.last_pano_rgb = np.roll(pano_rgb, shift=width_each_img, axis=1)
                    if depth_local_occ_meas is not None:
                        self.last_depth_local_occ = np.rot90(depth_local_occ_meas, k=-1)

            if self.enable_viz:
                if not self.enable_sim:
                    self.visualizer.set_observation(self.cmn_node.current_local_map)

            fwd, ang = self.motion_planner.cmd_discrete_action(action)

            if self.enable_sim:
                self.map_frame_manager.propagate_with_discrete_motion(action)

            _logger.warn("CMN: Just took action {:}".format(action))

            if self.cmn_node.agent_pose_estimate_px is not None:
                localization_result_px = self.cmn_node.agent_pose_estimate_px
                self.veh_pose_estimate_meters = self.map_frame_manager.transform_pose_px_to_m(localization_result_px)
                if self.enable_viz:
                    self.visualizer.veh_pose_estimate = localization_result_px
                    if self.enable_sim:
                        self.visualizer.veh_pose_true_px = self.map_frame_manager.veh_pose_true_px

        # Save training/eval data.
        if self.save_training_data:
            self.iteration += 1
            if pano_rgb is not None:
                cv2.imwrite(os.path.join(self.training_data_dirpath,
                                         "pano_rgb_{:03}.png".format(self.iteration)), pano_rgb)
            if current_local_map is not None:
                pass  # TODO: these are always blank.

    def set_new_odom(self, odom_pose: PoseMeters):
        """
        Update with a new odometry measurement from the robot.
        @param odom_pose - PoseMeters with x, y in meters and yaw in radians.
        """
        self.motion_planner.set_odom(odom_pose)

    def compute_observation_continuous(self, pano_rgb=None):
        """
        Use a panoramic RGB measurement to generate an observation grid.
        @param pano_rgb - Numpy array of four images concatenated horizontally (front, right, back, left).
        @return observation grid from the ML model.
        """
        map_obs = self.cmn_node.predict_local_occupancy(pano_rgb)
        self.cmn_node.current_local_map = map_obs
        return map_obs

    def run_particle_filter(self, current_local_map):
        """
        Run one iteration of the particle filter to update the estimated robot pose.
        @param current_local_map - New observation from ML model or ground truth sim.
        """
        if not self.enable_localization:
            self.veh_pose_estimate_meters = self.map_frame_manager.veh_pose_true_meters
            return

        self.veh_pose_estimate_meters = self.particle_filter.update_with_observation(current_local_map)
        if self.enable_viz:
            self.visualizer.particle_set = self.particle_filter.get_particle_set_px()
            self.visualizer.veh_pose_estimate = self.map_frame_manager.transform_pose_m_to_px(
                self.veh_pose_estimate_meters)
        self.particle_filter.resample()

    def command_motion_continuous(self, dt: float = None):
        """
        Choose and publish a continuous motion command, then propagate beliefs.
        @param dt - Timer period in seconds. Used for particle filter propagation.
        """
        fwd, ang = self.motion_planner.plan_path_to_goal(self.veh_pose_estimate_meters)
        if fwd is None and ang is None:
            _logger.info("Goal is reached, so ending the run loop.")
            # ROS 2: avoid exit() — raise so the calling node can shut down cleanly.
            raise SystemExit("Goal reached.")

        self.motion_planner.pub_velocity_cmd(fwd, ang)

        if self.enable_sim:
            self.map_frame_manager.propagate_with_vel(fwd, ang)

        if self.enable_localization:
            self.particle_filter.propagate_particles(fwd * dt, ang * dt)

        if self.enable_viz:
            self.visualizer.planned_path = self.motion_planner.path_px_reversed