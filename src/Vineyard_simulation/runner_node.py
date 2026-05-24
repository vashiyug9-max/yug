#!/usr/bin/env python3

"""
Simulation-only runner node for Vinebot.
Uses Habitat as the sensor backend and does not require any physical robot hardware.
"""

import os
import sys
import yaml
import cv2
import rclpy
import numpy as np

from time import strftime
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robo_project.scripts.cmn_interface import CoarseMapNavInterface, CmnConfig
from robo_project.scripts.basic_types import PosePixels
from habitat_interface import HabitatInterface

############ GLOBAL VARIABLES ###################
g_node = None
g_cv_bridge = CvBridge()

g_cmn_interface = None
g_habitat_interface = None

g_run_modes = ["continuous", "discrete", "discrete_random"]
g_run_mode = None
g_show_live_viz = False
g_verbose = False
g_manual_goal_cell = None

g_dt = 0.2
g_enable_localization = True
g_enable_ml_model = True
g_discrete_assume_yaw_is_known = False

g_save_training_data = False
g_training_data_dirpath = None

g_viz_paused = False
g_pub_viz_images = False
g_sim_viz_pub = None
g_cmn_viz_pub = None
#################################################


def read_params():
    """Read configuration params from the yaml."""

    possible_paths = []

    try:
        pkg_path = get_package_share_directory('robo_project')
        possible_paths.append(os.path.join(pkg_path, 'config', 'config.yaml'))
    except Exception:
        pkg_path = None

    possible_paths.append(os.path.expanduser('~/vinebot_ws/src/robo_project/config/config.yaml'))
    possible_paths.append(os.path.expanduser('~/vinebot_ws/src/robo_project/robo_project/config/config.yaml'))
    possible_paths.append(os.path.expanduser('~/vinebot_ws/src/Vineyard_simulation/config/config.yaml'))
    possible_paths.append(os.path.expanduser('~/vinebot_ws/src/Robo_project_ros2/src/robo_project/config/config.yaml'))

    global g_yaml_path
    g_yaml_path = None
    for path in possible_paths:
        if os.path.exists(path):
            g_yaml_path = path
            break

    if g_yaml_path is None:
        raise FileNotFoundError(
            "Could not find config.yaml. Checked:\n" + "\n".join(possible_paths)
        )

    with open(g_yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    global g_verbose, g_dt, g_enable_localization, g_enable_ml_model, g_discrete_assume_yaw_is_known
    g_verbose = config['verbose']
    g_dt = config['dt']
    g_enable_localization = config['particle_filter']['enable']
    g_enable_ml_model = not config['model']['skip_loading']
    g_discrete_assume_yaw_is_known = config['discrete_assume_yaw_is_known']

    if config['manually_set_goal_cell']:
        global g_manual_goal_cell
        g_manual_goal_cell = PosePixels(config['goal_row'], config['goal_col'])

    global g_use_lidar_as_ground_truth, g_fuse_lidar_with_rgb, g_use_depth_as_ground_truth
    g_use_lidar_as_ground_truth = config['lidar']['use_lidar_as_ground_truth']
    g_fuse_lidar_with_rgb = config['lidar']['fuse_lidar_with_rgb']
    g_use_depth_as_ground_truth = config['depth']['use_depth_as_ground_truth']

    global g_use_depth_pointcloud
    g_use_depth_pointcloud = False
    if g_use_depth_as_ground_truth:
        g_use_depth_pointcloud = config['depth']['use_pointcloud']

    global g_meas_topic, g_desired_meas_shape
    g_meas_topic = config['measurements']['topic']
    g_desired_meas_shape = (
        config['measurements']['height'],
        config['measurements']['width']
    )

    global g_save_training_data, g_training_data_dirpath
    g_save_training_data = config['save_data_for_training']
    if g_save_training_data:
        g_training_data_dirpath = config['training_data_dirpath']
        if g_training_data_dirpath[0] != '/':
            if pkg_path is None:
                pkg_path = os.path.expanduser('~/vinebot_ws/src/Robo_project_ros2/src/robo_project')
            g_training_data_dirpath = os.path.join(pkg_path, g_training_data_dirpath)

        g_training_data_dirpath = os.path.join(
            g_training_data_dirpath,
            strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(g_training_data_dirpath, exist_ok=True)

def set_global_params(run_mode: str, use_viz: bool):
    global g_run_mode, g_show_live_viz, g_cmn_interface, g_habitat_interface

    g_run_mode = run_mode
    g_show_live_viz = use_viz

    config = CmnConfig()
    config.run_mode = run_mode
    config.enable_sim = True
    config.enable_viz = use_viz
    config.enable_ml_model = g_enable_ml_model
    config.enable_localization = g_enable_localization
    config.use_lidar_as_ground_truth = False
    config.fuse_lidar_with_rgb = False
    config.use_depth_as_ground_truth = False
    config.assume_yaw_is_known = g_discrete_assume_yaw_is_known and "discrete" in g_run_mode

    if g_manual_goal_cell is not None:
        config.manually_set_goal_cell = True
        config.manual_goal_cell = g_manual_goal_cell

    g_cmn_interface = CoarseMapNavInterface(config, cmd_vel_pub=None)
    g_cmn_interface.save_training_data = g_save_training_data
    g_cmn_interface.training_data_dirpath = g_training_data_dirpath

    g_habitat_interface = HabitatInterface()
    g_node.get_logger().info("Habitat interface initialized for Vinebot simulation.")


def get_pano_meas():
    pano_rgb = g_habitat_interface.get_pano_rgb()
    local_occ_meas = None
    return pano_rgb, local_occ_meas


def apply_action_to_habitat():
    if g_cmn_interface is None or g_habitat_interface is None:
        return

    if not hasattr(g_cmn_interface, "motion_planner"):
        return

    planner = g_cmn_interface.motion_planner

    if hasattr(planner, "last_discrete_action"):
        action = planner.last_discrete_action
        if action == "move_forward":
            g_habitat_interface.act("move_forward")
        elif action == "turn_left":
            g_habitat_interface.act("turn_left")
        elif action == "turn_right":
            g_habitat_interface.act("turn_right")


def timer_update_loop():
    global g_viz_paused

    if g_cmn_interface.visualizer is not None:
        sim_viz_img = g_cmn_interface.visualizer.get_updated_img()
        cmn_viz_img = None

        if g_cmn_interface.cmn_node is not None and g_cmn_interface.cmn_node.visualizer is not None:
            cmn_viz_img = g_cmn_interface.cmn_node.visualizer.get_updated_img()

        if g_pub_viz_images:
            if sim_viz_img is not None:
                g_sim_viz_pub.publish(g_cv_bridge.cv2_to_imgmsg(sim_viz_img))
            if cmn_viz_img is not None:
                g_cmn_viz_pub.publish(g_cv_bridge.cv2_to_imgmsg(cmn_viz_img))
        else:
            if sim_viz_img is not None:
                cv2.imshow("viz image", sim_viz_img)
            if cmn_viz_img is not None:
                cv2.imshow("cmn viz image", cmn_viz_img)

            key = cv2.waitKey(int(g_dt * 1000))
            if key == ord('q'):
                cv2.destroyAllWindows()
                g_node.get_logger().info("User pressed Q key. Shutting down.")
                rclpy.shutdown()
                sys.exit()
            elif key == ord(' '):
                g_viz_paused = not g_viz_paused

            if g_viz_paused:
                return

    pano_rgb, local_occ_depth = get_pano_meas()

    g_cmn_interface.run(
        pano_rgb,
        g_dt,
        None,
        local_occ_depth
    )

    apply_action_to_habitat()


def main(args=None):
    rclpy.init(args=args)

    global g_node
    g_node = rclpy.create_node('vinebot_runner_node')

    read_params()

    if len(sys.argv) > 2:
        run_mode = sys.argv[1]
        use_viz = sys.argv[2].lower() == "true"
    else:
        print("Usage: python3 runner_node.py <run_mode> <use_viz>")
        print("Example: python3 runner_node.py discrete true")
        sys.exit()

    if run_mode not in g_run_modes:
        g_node.get_logger().error(f"Invalid run_mode {run_mode}. Exiting.")
        sys.exit()

    set_global_params(run_mode, use_viz)

    global g_sim_viz_pub, g_cmn_viz_pub
    g_sim_viz_pub = g_node.create_publisher(Image, "/cmn/viz/sim", 1)
    g_cmn_viz_pub = g_node.create_publisher(Image, "/cmn/viz/cmn", 1)

    g_node.create_timer(g_dt, timer_update_loop)

    executor = MultiThreadedExecutor()
    executor.add_node(g_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        g_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
