"""End-to-end CMN navigation in preview mode; /cmd_vel blocked by default.

Workflow: Habitat sensors -> Fusion local grid -> existing global localizer ->
relative-odom pose bridge -> existing motion_planner + runner_node.
Only the optional SIMULATION-ONLY gate may command /cmd_vel; it is disabled by
default and requires a fresh user-selected goal + small GT error if enabled.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arg = lambda k, default, description='': DeclareLaunchArgument(
        k, default_value=default, description=description)

    simulator = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('start_habitat')),
        cmd=[
            LaunchConfiguration('habitat_python'),
            '-m', 'robo_project.sim.habitat_bridge', '--ros-args',
            '-p', 'gpu_device_id:=-1',
            '-p', 'show_window:=true',
            '-p', 'publish_lidar:=true',
            '-p', 'lidar_model:=mid360',
        ],
        name='cmn_habitat_cpu', output='screen',
    )

    fusion = Node(
        package='robo_project', executable='fusion_obstacle_node',
        name='fusion_obstacle_node', output='screen',
        condition=IfCondition(LaunchConfiguration('start_fusion')),
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'rig': LaunchConfiguration('rig'),
            # World obstacle map must not be projected with GT in final CMN.
            'lidar_map': False,
        }],
    )

    localizer = Node(
        package='robo_project', executable='live_global_pose_node',
        name='step361_live_localization', output='screen')

    evaluator = Node(
        package='robo_project', executable='localization_error_monitor_node',
        name='step364_motion_accuracy', output='screen',
        condition=IfCondition(LaunchConfiguration('evaluate_gt')))

    bridge = Node(
        package='robo_project', executable='localization_navigation_bridge',
        name='cmn_navigation_bridge', output='screen')

    # Scope remaps to THESE nodes only. Habitat/Fusion still receive the real
    # relative /odom; planner and runner see ONLY localization-derived nav_odom.
    # The Fusion WORLD map is disconnected: its coordinates currently depend on
    # simulator /odom, which is not a legitimate localization global map frame.
    planner = Node(
        package='robo_project', executable='motion_planner',
        name='motion_planner', output='screen',
        remappings=[
            ('/odom', '/localization/nav_odom'),
            ('/fusion/obstacle_map', '/localization/disabled_world_obstacle_map'),
        ])

    runner = Node(
        package='robo_project', executable='runner_node',
        name='runner_node', output='screen',
        remappings=[
            ('/odom', '/localization/nav_odom'),
            ('/cmd_vel', '/localization/proposed_cmd_vel'),
            ('/fusion/obstacle_map', '/localization/disabled_world_obstacle_map'),
        ])

    gate = Node(
        package='robo_project', executable='localization_motion_gate',
        name='cmn_simulation_motion_gate', output='screen',
        parameters=[{'enable_motion': ParameterValue(
            LaunchConfiguration('enable_motion'), value_type=bool)}])

    return LaunchDescription([
        arg('start_habitat', 'false', 'Start CPU-GL Habitat from ROS-capable conda Python'),
        arg('habitat_python', '/home/jd1152/miniconda3/envs/habitat_ros310_np1/bin/python'),
        arg('start_fusion', 'false', 'Start Fusion ML (only if not already running)'),
        arg('model_path', '/home/jd1152/robotics_project/robo_project_ws/runs/fusion_mid360_e3.0/best.pt'),
        arg('rig', '/home/jd1152/robotics_project/robo_project_ws/src/robo_project/config/vinebot_rig.yaml'),
        arg('evaluate_gt', 'true', 'Simulation-only GT reporting, never localization input'),
        arg('enable_motion', 'false',
            'Simulation ONLY; guarded by GT error <0.35m, yaw <15deg + fresh explicit goal'),
        simulator, fusion, localizer, evaluator, bridge, planner, runner, gate,
    ])
