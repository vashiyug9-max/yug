"""
Real VineBot bringup: motors (Roboteq via ros2_control), PS4 joystick and who-drives logic.

  ros2 launch vinebot_bringup real_robot.launch.py                      # motors + joystick
  ros2 launch vinebot_bringup real_robot.launch.py joystick:=false      # motors only
  ros2 launch vinebot_bringup real_robot.launch.py use_mock_hardware:=true   # no robot attached

What runs
  controller_manager   their roboteq package unchanged: hardware plugin + diff_drive_controller
                       (config roboteq/config/control.yaml), robot model = roboteq differential-robot
                       (it carries the ros2_control part). Odometry -> /odom (+ TF odom -> base_footprint).
  robot_state_publisher  OUR vinebot_description (base_footprint at the track centre, sensors where
                       config/vinebot_rig.yaml has them) - the only robot model for TF.
  twist_mux            config/twist_mux.yaml: joystick (L1 held) > autonomy (/cmd_vel); /emergency
                       (PS4 O button) = stop. Output -> /roboteq_controller/cmd_vel_unstamped.
  joy_node, ps4_ros2, joy_gate   the joystick (ps4_ros2 unchanged, its output goes through joy_gate).
Our autonomous nodes (runner, planner, ML) need no change: they keep publishing /cmd_vel, reading /odom.
confirm_moves:=true (default): /cmd_vel reaches the motors only after "y" in motion_confirm
  (ros2 run vinebot_bringup motion_confirm, own terminal) - once per new goal.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = [
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0", description="Roboteq serial port"),
        DeclareLaunchArgument("baud_rate", default_value="115200", description="Roboteq baud rate"),
        DeclareLaunchArgument("use_mock_hardware", default_value="false",
                              description="true: no Roboteq attached, commands are mirrored to states"),
        DeclareLaunchArgument("joystick", default_value="true", description="start joy_node + ps4_ros2 + joy_gate"),
        DeclareLaunchArgument("enable_button", default_value="4", description="deadman button index (4 = L1)"),
        DeclareLaunchArgument("odom_topic", default_value="/odom",
                              description="where the Roboteq wheel odometry goes: /odom (no localization) or "
                                          "/wheel_odom (with localization/sketch_pf_node, which then provides "
                                          "/localization/odom for runner / planner / ML)"),
        DeclareLaunchArgument("confirm_moves", default_value="true",
                              description="true: autonomy reaches the motors only through motion_confirm "
                                          "(run it in its own terminal, it asks Move? [y/N] per goal)"),
    ]
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    use_mock = LaunchConfiguration("use_mock_hardware")
    joystick = LaunchConfiguration("joystick")

    roboteq_share = FindPackageShare("roboteq")
    xacro = PathJoinSubstitution([FindExecutable(name="xacro")])

    # Model for the controller_manager only (contains the Roboteq ros2_control hardware block).
    control_description = {"robot_description": ParameterValue(Command([
        xacro, " ", PathJoinSubstitution([roboteq_share, "urdf", "differential-robot.urdf.xacro"]),
        " use_mock_hardware:=", use_mock,
        " roboteq_params_file:=", PathJoinSubstitution([roboteq_share, "config", "roboteq.yaml"]),
        " serial_port:=", serial_port, " baud_rate:=", baud_rate]), value_type=str)}
    controllers = PathJoinSubstitution([roboteq_share, "config", "control.yaml"])

    # Model for TF / RViz: ours.
    tf_description = {"robot_description": ParameterValue(Command([
        xacro, " ", PathJoinSubstitution([FindPackageShare("vinebot_description"), "urdf", "vinebot.xacro"])]),
        value_type=str)}

    control_node = Node(
        package="controller_manager", executable="ros2_control_node", output="screen",
        parameters=[control_description, controllers],
        # all controllers run in this process: their odometry goes to /odom for our nodes
        remappings=[("/roboteq_controller/odom", LaunchConfiguration("odom_topic"))],
    )
    robot_controller = Node(
        package="controller_manager", executable="spawner",
        arguments=["roboteq_controller", "--param-file", controllers],
    )
    joint_state_broadcaster = Node(
        package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"],
    )
    after_controller = RegisterEventHandler(OnProcessExit(target_action=robot_controller,
                                                          on_exit=[joint_state_broadcaster]))

    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[tf_description],
    )
    # our model's track wheels are not driven by the Roboteq joints: publish them at 0 for TF
    joint_state_publisher = Node(
        package="joint_state_publisher", executable="joint_state_publisher", parameters=[tf_description],
    )

    # autonomy input of twist_mux: /cmd_vel_confirmed (via motion_confirm) or /cmd_vel directly
    nav_topic = PythonExpression(["'/cmd_vel_confirmed' if '", LaunchConfiguration("confirm_moves"),
                                  "' == 'true' else '/cmd_vel'"])
    twist_mux = Node(
        package="twist_mux", executable="twist_mux", output="screen",
        parameters=[PathJoinSubstitution([FindPackageShare("vinebot_bringup"), "config", "twist_mux.yaml"]),
                    {"topics.navigation.topic": nav_topic}],
        remappings=[("cmd_vel_out", "/roboteq_controller/cmd_vel_unstamped")],
    )

    joy = Node(package="joy", executable="joy_node", condition=IfCondition(joystick))
    ps4 = Node(package="ps4_ros2", executable="ps4_ros2", output="screen", condition=IfCondition(joystick),
               parameters=[{"pub_topic": "/cmd_vel_joy_raw"}])
    gate = Node(package="vinebot_bringup", executable="joy_gate", output="screen", condition=IfCondition(joystick),
                parameters=[{"enable_button": ParameterValue(LaunchConfiguration("enable_button"), value_type=int)}])

    return LaunchDescription(args + [control_node, robot_controller, after_controller, robot_state_publisher,
                                     joint_state_publisher, twist_mux, joy, ps4, gate])
