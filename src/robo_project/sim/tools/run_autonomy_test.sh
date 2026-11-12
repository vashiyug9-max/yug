#!/bin/bash
# End-to-end test of the REAL-ROBOT autonomy launch (vinebot_bringup/autonomy.launch.py) with the simulation
# as the robot: Habitat provides cameras / depth / LiDAR, wheel_odom_sim.py stands in for the Roboteq
# encoders (/wheel_odom), and autonomy.launch.py starts localization + ML + planner + runner exactly as on
# the robot (topic remaps, sync_tolerance, forget_s, goal_clearance, python for the planner).
#   src/robo_project/sim/tools/run_autonomy_test.sh OUT_DIR ["extra autonomy.launch.py args", e.g. odometry:=builtin]
# Then the 6-goal sequence (judged on the localized pose), collision analysis and localization error.
D=$1; XARGS=$2
[ -z "$D" ] && { echo "usage: $0 OUT_DIR"; exit 1; }
D=$(realpath -m "$D")
WS=$(cd "$(dirname "$(realpath "$0")")/../../../.." && pwd)
cd "$WS" && eval "$(conda shell.bash hook)" && conda activate habitat && source install/setup.bash
rm -rf "$D"; mkdir -p "$D"
T=src/robo_project/sim/tools
PAT='robo_project\.(sim|planning)\.|lib/robo_project/|lib/robot_localization/|lib/vinebot_bringup/|goal_sequencer|wheel_odom_sim|imu_sim|ros2 (run|bag|launch)'
stopall() {
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_autonomy_test" | awk '{print $1}')
  [ -n "$P" ] && kill -INT $P 2>/dev/null; sleep 5
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_autonomy_test" | awk '{print $1}')
  [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 1
}
stopall
(python3 -m robo_project.sim.habitat_bridge --ros-args -p show_window:=false > $D/bridge.log 2>&1 &)
timeout 120 bash -c "until grep -q 'fully initialized' $D/bridge.log 2>/dev/null; do sleep 1; done"
# the real chain: track encoders (3% distance, 8% turn error, noise) + IMU gyro (1% scale, bias, noise);
# autonomy.launch.py fuses them (default wheel_imu_odom; pass "odometry:=ekf" for robot_localization)
(python3 $T/wheel_odom_sim.py > $D/wheel.log 2>&1 &)
(python3 $T/imu_sim.py > $D/imu.log 2>&1 &)
(ros2 bag record -o $D/bag /odom /tf /tf_static /cmd_vel /map /planned_path /goal_pose /fusion/local_grid \
   /fusion/obstacle_map /livox/lidar /localization/odom /wheel_odom /odometry/filtered /imu/data > $D/bag.log 2>&1 &)
(ros2 launch vinebot_bringup autonomy.launch.py use_ekf:=true init_x:=-4.10 init_y:=4.90 init_yaw:=0.0 \
   front_rgb:=/camera/front/image_raw front_depth:=/camera/front/depth/image_raw \
   left_rgb:=/camera/left/image_raw right_rgb:=/camera/right/image_raw back_rgb:=/camera/back/image_raw \
   python:=$(which python3) $XARGS > $D/autonomy.log 2>&1 &)
timeout 120 bash -c "until grep -q 'Path published' $D/autonomy.log 2>/dev/null; do sleep 1; done" \
  || echo "WARNING: no path within 120 s"
echo "=== nodes:"; grep -oE "\[(sketch_pf|runner_node|fusion_obstacle_node|motion_planner|python3)[^]]*\]" $D/autonomy.log | sort -u | tr '\n' ' '; echo
grep -m1 "Fusion model" $D/autonomy.log | sed 's/.*Fusion model /  model: /; s/ on .*//'
grep -m1 -E "Odometry: (built-in|distance)|ekf_node-[0-9]+\]: process started" $D/autonomy.log | sed 's/.*Odometry/  odometry/; s/.*ekf_node.*/  odometry: robot_localization EKF/'
grep -m1 "PF started" $D/autonomy.log | sed 's/.*PF started/  PF started/'
echo "=== 6 goals (localized pose)"
timeout 1100 python3 $T/goal_sequencer.py --ros-args -r /odom:=/localization/odom > $D/seq.log 2>&1
sed -n '/SUMMARY/,/DONE/p' $D/seq.log | grep goal
echo "  NO PATH: $(grep -c 'NO PATH' $D/autonomy.log)  stalls: $(grep -c 'Stalled' $D/autonomy.log)  sim contacts: $(grep -c 'blocked by the navmesh' $D/bridge.log)"
stopall
python3 $T/analyze_collisions.py $D/bag src/environment/IMR_lab8.navmesh $D/analysis 2>&1 | grep -A6 '^2\. BODY'
echo "=== driving smoothness: stops on the way"; /usr/bin/python3 $T/stop_analysis.py $D/bag 2>&1 | grep -E "^ALL"
/usr/bin/python3 $T/loc_error.py $D/bag $D/analysis 2>&1 | grep -vE "^\["
echo "all stopped"
