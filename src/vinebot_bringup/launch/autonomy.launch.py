"""
Real VineBot autonomy: localization on the hand-drawn map + ML obstacle map + A* planner + path follower.
Start it AFTER real_robot.launch.py (motors / joystick / twist_mux) and the sensor drivers.

  ros2 launch vinebot_bringup real_robot.launch.py odom_topic:=/wheel_odom          # terminal 1
  ros2 run vinebot_bringup motion_confirm                                            # terminal 1b
  ros2 launch vinebot_bringup autonomy.launch.py init_x:=... init_y:=... init_yaw:=...   # this file

What runs (our nodes unchanged, connected by remaps / parameters - the same as the tested simulation setup
"run_regression.sh ... loc_imu"):
  odometry (use_ekf:=true)  /wheel_odom (distance) + /imu/data (turn rate) -> /odometry/filtered
                      default: vinebot_bringup wheel_imu_odom (wheel distance + gyro turn, learns the gyro bias at
                      standstill; imu_yaw_sign:=-1 if the IMU is upside down). odometry:=ekf: robot_localization
                      ekf_node (config/ekf.yaml) - in the simulation test 15 deg mean heading error vs 4 deg (no bias)
  sketch_pf_node      odometry + /fusion/local_grid + /map -> /localization/odom (pose on the sketch)
  runner_node         publishes /map (sketch), follows /planned_path -> /cmd_vel (-> motion_confirm -> motors)
  fusion_obstacle_node  4 cameras + front depth + Livox -> ML local grid + obstacle map
                        (sensor stamps matched within sync_tolerance; Livox points moved to base_footprint
                        with the rig mounting; forget_s fades old obstacles)
  motion_planner      A* on sketch + ML obstacles (run with the python that has numba: python3 -m ...)
Start pose: init_x / init_y / init_yaw on the sketch map, or "2D Pose Estimate" in RViz (/initialpose).
"""
from launch import LaunchDescription
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("init_x", default_value="0.0", description="start pose on the sketch map (m)"),
        DeclareLaunchArgument("init_y", default_value="0.0"),
        DeclareLaunchArgument("init_yaw", default_value="0.0", description="rad, 0 = +x of the map"),
        DeclareLaunchArgument("use_ekf", default_value="true",
                              description="true: wheels + IMU (needs ros-humble-robot-localization and /imu/data); "
                                          "false: wheel odometry only (larger heading errors)"),
        DeclareLaunchArgument("imu_topic", default_value="/imu/data"),
        DeclareLaunchArgument("imu_yaw_sign", default_value="1.0",
                              description="built-in odometry only: -1.0 if turning left gives a negative gyro z"),
        DeclareLaunchArgument("odometry", default_value="builtin",
                              description="builtin (vinebot_bringup wheel_imu_odom, learns the gyro bias - tested best) | "
                                          "ekf (robot_localization) | auto (ekf if installed)"),
        DeclareLaunchArgument("front_rgb", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("front_depth", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("left_rgb", default_value="/camera/left/image_raw"),
        DeclareLaunchArgument("right_rgb", default_value="/camera/right/image_raw"),
        DeclareLaunchArgument("back_rgb", default_value="/camera/back/image_raw"),
        DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("sync_tolerance", default_value="0.1", description="s, sensor stamp matching"),
        DeclareLaunchArgument("python", default_value="python3",
                              description="interpreter for motion_planner (needs numba, e.g. the conda env)"),
    ]
    use_ekf = L("use_ekf")
    odom_in = PythonExpression(["'/odometry/filtered' if '", use_ekf, "' == 'true' else '/wheel_odom'"])
    # filter settings matched to the odometry quality (tested on seeded replays, see sketch_pf_node.py):
    # wheels + IMU = node defaults; wheel odometry only = larger motion noise + sketch heading corrections
    sel = lambda imu, wheel: PythonExpression([f"{imu} if '", use_ekf, f"' == 'true' else {wheel}"])
    rot_k, rot_m = sel(0.02, 0.15), sel(0.005, 0.05)
    n_t, n_r = sel(0.05, 0.08), sel(0.03, 0.10)
    to_loc = ("/odom", "/localization/odom")

    try:
        get_package_share_directory("robot_localization")
        have_rl = "true"
    except PackageNotFoundError:
        have_rl = "false"
    mode = L("odometry")
    use_rl = PythonExpression(["'", use_ekf, "' == 'true' and ('", mode, "' == 'ekf' or ('", mode,
                               "' == 'auto' and ", "'" + have_rl + "'", " == 'true'))"])
    use_builtin = PythonExpression(["'", use_ekf, "' == 'true' and not (", use_rl, ")"])
    ekf = Node(package="robot_localization", executable="ekf_node", name="ekf_filter_node", output="screen",
               condition=IfCondition(use_rl),
               parameters=[PathJoinSubstitution([FindPackageShare("vinebot_bringup"), "config", "ekf.yaml"])],
               remappings=[("/imu/data", L("imu_topic"))])
    builtin = Node(package="vinebot_bringup", executable="wheel_imu_odom", output="screen",
                   condition=IfCondition(use_builtin),
                   parameters=[{"imu_topic": L("imu_topic"),
                                "imu_yaw_sign": ParameterValue(L("imu_yaw_sign"), value_type=float)}])
    note = LogInfo(condition=IfCondition(use_builtin),
                   msg="Odometry: built-in wheels + IMU fusion (wheel_imu_odom)"
                       + ("" if have_rl == "true" else " - robot_localization is not installed"))
    pf = Node(package="robo_project", executable="sketch_pf_node", output="screen",
              parameters=[{"odom_topic": ParameterValue(odom_in, value_type=str),
                           "init_x": ParameterValue(L("init_x"), value_type=float),
                           "init_y": ParameterValue(L("init_y"), value_type=float),
                           "init_yaw": ParameterValue(L("init_yaw"), value_type=float),
                           "corr_rot_per_rad": ParameterValue(rot_k, value_type=float),
                           "corr_rot_per_m": ParameterValue(rot_m, value_type=float),
                           "noise_trans_per_m": ParameterValue(n_t, value_type=float),
                           "noise_rot_per_rad": ParameterValue(n_r, value_type=float)}])
    runner = Node(package="robo_project", executable="runner_node", output="screen", remappings=[to_loc])
    fusion = Node(package="robo_project", executable="fusion_obstacle_node", output="screen",
                  parameters=[{"sync_tolerance": ParameterValue(L("sync_tolerance"), value_type=float),
                               "forget_s": 20.0}],
                  remappings=[to_loc,
                              ("/camera/front/image_raw", L("front_rgb")),
                              ("/camera/front/depth/image_raw", L("front_depth")),
                              ("/camera/left/image_raw", L("left_rgb")),
                              ("/camera/right/image_raw", L("right_rgb")),
                              ("/camera/back/image_raw", L("back_rgb")),
                              ("/livox/lidar", L("lidar_topic"))])
    planner = ExecuteProcess(
        cmd=[L("python"), "-m", "robo_project.planning.motion_planner", "--ros-args",
             "-r", "/odom:=/localization/odom"],
        output="screen")
    return LaunchDescription(args + [note, ekf, builtin, pf, runner, fusion, planner])
