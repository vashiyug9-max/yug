#!/usr/bin/env python3

"""
ROS 2 launch file for the CMN project.
Replaces the ROS 1 .launch XML file.

Usage:
  ros2 launch robo_project run.launch.py run_mode:=discrete use_sim:=true use_viz:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Declare launch arguments (replaces <arg> tags in ROS 1 .launch) ──
    run_mode_arg = DeclareLaunchArgument(
        'run_mode',
        default_value='discrete',
        description='Run mode: one of [continuous, discrete, discrete_random]'
    )
    use_sim_arg = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use the simulator to generate ground truth observations'
    )
    use_viz_arg = DeclareLaunchArgument(
        'use_viz',
        default_value='false',
        description='Show the live visualization window'
    )

    # ── Runner node ───────────────────────────────────────────────────────
    runner_node = Node(
        package='robo_project',
        executable='runner_node',
        name='runner_node',
        output='screen',
        emulate_tty=True,   # Ensures coloured log output in the terminal.
        parameters=[{
            # These map to self.declare_parameter() calls in RunnerNode.__init__()
            'run_mode': LaunchConfiguration('run_mode'),
            'use_sim':  LaunchConfiguration('use_sim'),
            'use_viz':  LaunchConfiguration('use_viz'),
        }]
    )

    return LaunchDescription([
        run_mode_arg,
        use_sim_arg,
        use_viz_arg,
        runner_node,
    ])