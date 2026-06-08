import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    # Robot localization EKF node for sensor fusion (Wheel Odometry + IMU)
    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        parameters=[{
            'use_sim_time': True,
            'frequency': 30.0,
            'sensor_timeout': 0.1,
            'two_d_mode': False,
            'transform_time_offset': 0.0,
            'transform_timeout': 0.0,
            'print_diagnostics': True,
            'debug': False,
            
            # Wheel odometry settings
            'odom0': '/diff_cont/odom',
            'odom0_config': [True, True, False,      # x, y, z (no z for wheeled robot)
                           False, False, True,       # roll, pitch, yaw (only yaw)
                           True, True, False,        # vx, vy, vz (no vz)  
                           False, False, True,       # vroll, vpitch, vyaw (only vyaw)
                           False, False, False],     # ax, ay, az
            'odom0_queue_size': 5,
            'odom0_differential': False,
            'odom0_relative': True,
            
            # IMU settings
            'imu0': '/imu',  # Change to your IMU topic name
            'imu0_config': [False, False, False,    # x, y, z (don't use position)
                          True, True, True,         # roll, pitch, yaw (orientation)
                          False, False, False,      # vx, vy, vz (don't use velocities)
                          True, True, True,         # vroll, vpitch, vyaw (angular velocities)
                          True, True, True],        # ax, ay, az (linear accelerations)
            'imu0_queue_size': 5,
            'imu0_differential': False,
            'imu0_relative': True,
            'imu0_remove_gravitational_acceleration': True,
            
            # Frame settings
            'map_frame': 'map',
            'odom_frame': 'odom',
            'base_link_frame': 'base_footprint',
            'world_frame': 'odom',

            'process_noise_covariance': [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
                                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.015]
        }],
        output='screen'
    )

    # RTAB-Map node using fused odometry from robot_localization
    rtab = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        parameters=[{
            'use_sim_time': True,
            'subscribe_depth': False,
            'subscribe_rgb': False,
            'subscribe_scan_cloud': True,
            'subscribe_odom': True,
            'frame_id': 'base_footprint',
            'map_frame_id': 'map',
            'odom_frame_id': 'odom',
            'wait_for_transform': 10.0,
            'Reg/Strategy': '1',  # Use ICP registration
            'RGBD/NeighborLinkRefining': 'true',
            'Grid/FromDepth': 'false',
            'Grid/Sensor': '0'
        }],
        remappings=[
            ('scan_cloud', '/points'),
            ('odom', '/odometry/filtered')  # Use EKF fused odometry instead of raw wheel odom
        ],
        output='screen'
    )
   
    
    return LaunchDescription([
        robot_localization_node,
        rtab,
    ])
