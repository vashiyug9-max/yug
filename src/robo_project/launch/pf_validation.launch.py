#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("dataset_path", default_value=[EnvironmentVariable("HOME"), "/robo_project_ws/dataset/cmn_dataset_delton"]),
            DeclareLaunchArgument("trajectory_id", default_value="trajectory_01"),
            DeclareLaunchArgument("publish_period", default_value="0.2"),
            DeclareLaunchArgument("odom_topic", default_value="/odom"),
            DeclareLaunchArgument("observation_topic", default_value="/local_occupancy_gt"),
            Node(
                package="robo_project",
                executable="dataset_gt_replay_node",
                output="screen",
                parameters=[
                    {
                        "dataset_path": LaunchConfiguration("dataset_path"),
                        "trajectory_id": LaunchConfiguration("trajectory_id"),
                        "publish_period": LaunchConfiguration("publish_period"),
                    }
                ],
            ),
            Node(
                package="robo_project",
                executable="particle_filter_validation_node",
                output="screen",
                parameters=[
                    {
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "observation_topic": LaunchConfiguration("observation_topic"),
                    }
                ],
            ),
        ]
    )
