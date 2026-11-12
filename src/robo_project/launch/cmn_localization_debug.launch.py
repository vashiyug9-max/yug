"""CMN localization debug pipeline. Does NOT start planner, runner, or cmd_vel."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fusion = Node(
        package='robo_project',
        executable='fusion_obstacle_node',
        name='fusion_obstacle_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_fusion')),
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'rig': LaunchConfiguration('rig'),
            'lidar_map': False,
        }],
    )

    localizer = Node(
        package='robo_project',
        executable='live_global_pose_node',
        name='step361_live_localization',
        output='screen',
    )

    evaluator = Node(
        package='robo_project',
        executable='localization_error_monitor_node',
        name='step364_motion_accuracy',
        output='screen',
        condition=IfCondition(LaunchConfiguration('evaluate_gt')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_fusion', default_value='false',
            description='Set true only if Fusion is NOT already running'),
        DeclareLaunchArgument(
            'evaluate_gt', default_value='true',
            description='GT evaluation ONLY in simulation; set false on robot'),
        DeclareLaunchArgument('model_path', default_value=''),
        DeclareLaunchArgument('rig', default_value=''),
        fusion,
        localizer,
        evaluator,
    ])
