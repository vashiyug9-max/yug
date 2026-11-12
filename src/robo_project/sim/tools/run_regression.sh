#!/bin/bash
# Full headless regression test of the simulation stack (no RViz, no camera window).
#
#   src/robo_project/sim/tools/run_regression.sh OUT_DIR ["extra bridge args"] [loc]
#   e.g. src/robo_project/sim/tools/run_regression.sh results/test_tracked "-p drive_model:=tracked"
#        src/robo_project/sim/tools/run_regression.sh results/test_loc "" loc        (wheel odometry only)
#        src/robo_project/sim/tools/run_regression.sh results/test_loc "" loc_imu    (wheels + IMU heading)
#   loc: localization test - simulated wheel odometry (sim/tools/wheel_odom_sim.py) + sketch particle filter
#        (localization/sketch_pf_node.py); runner, planner, ML and the goal check use /localization/odom (what
#        the robot believes) instead of the true /odom. Contacts / body overlap are judged with the TRUE pose,
#        and loc_error.py reports the estimate's error against the true pose.
#
# A: the 6-goal sequence (goal_sequencer.py), recorded to OUT_DIR/bag, then analysed:
#    analyze_collisions.py (contacts / body overlap) and eval_model.py (ML accuracy vs ground truth)
# B: the former freeze spot: start (-4.29, 2.12), goal (-3.84, -0.78) (goal inside an obstacle)
# Stop everything else first (the script also kills leftover nodes). Takes ~10 min.
D=$1; BARGS=$2; LOC=$3
[ -z "$D" ] && { echo "usage: $0 OUT_DIR [\"extra bridge args\"] [loc]"; exit 1; }
D=$(realpath -m "$D")
WS=$(cd "$(dirname "$(realpath "$0")")/../../../.." && pwd)      # workspace root (this file: WS/src/robo_project/sim/tools)
cd "$WS" && eval "$(conda shell.bash hook)" && conda activate habitat && source install/setup.bash
rm -rf "$D"; mkdir -p "$D"
REMAP=""; FUSION_LOC=""; PLAN_LOC=""; WHEEL_ARGS=""; PF_ARGS=""
# loc / loc_imu: estimated pose; the ML obstacle map forgets old evidence (forget_s) so pose corrections
# leave no ghosts. loc = wheel odometry only (tracks over-report turns by 8%); loc_imu = wheels + IMU heading
# (1% turn error, like an EKF with the BNO055) and the filter then hardly corrects the heading from the sketch.
if [ "$LOC" = "loc" ] || [ "$LOC" = "loc_imu" ]; then
  REMAP="--ros-args -r /odom:=/localization/odom"; FUSION_LOC="-p forget_s:=20.0"
fi
if [ "$LOC" = "loc_imu" ]; then
  WHEEL_ARGS="--ros-args -p rot_scale:=1.01 -p rot_noise:=0.01"     # PF defaults = wheels + IMU
fi
if [ "$LOC" = "loc" ]; then
  PF_ARGS="-p noise_trans_per_m:=0.08 -p noise_rot_per_rad:=0.10 -p corr_rot_per_rad:=0.15 -p corr_rot_per_m:=0.05"
fi
T=src/robo_project/sim/tools
PAT='robo_project\.(sim|planning)\.|lib/robo_project/|lib/vinebot_bringup/|lib/robot_localization/|goal_sequencer|one_goal|wheel_odom_sim|imu_sim|ros2 (run|bag)'
stopall() {
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_regression" | awk '{print $1}')
  [ -n "$P" ] && kill -INT $P 2>/dev/null; sleep 4
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_regression" | awk '{print $1}')
  [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 1
}
start() {   # $1 = run name, $2 = start pose "x y yaw" (for the PF), rest = more bridge args
  local nm=$1; local pose=($2); shift 2
  (python3 -m robo_project.sim.habitat_bridge --ros-args -p show_window:=false $BARGS "$@" > $D/${nm}_bridge.log 2>&1 &)
  timeout 120 bash -c "until grep -q 'fully initialized' $D/${nm}_bridge.log 2>/dev/null; do sleep 1; done"
  if [ -n "$REMAP" ]; then
    for try in 1 2; do
      (python3 $T/wheel_odom_sim.py $WHEEL_ARGS > $D/${nm}_wheel.log 2>&1 &)
      (ros2 run robo_project sketch_pf_node --ros-args -p init_x:=${pose[0]} -p init_y:=${pose[1]} -p init_yaw:=${pose[2]} $PF_ARGS > $D/${nm}_pf.log 2>&1 &)
      # the localization chain must be alive (ROS discovery right after killed processes is sometimes lost)
      timeout 30 bash -c "until grep -q 'PF started' $D/${nm}_pf.log 2>/dev/null; do sleep 1; done" && break
      echo "  WARNING: particle filter got no wheel odometry within 30 s - restarting it (try $try)"
      P=$(ps -eo pid,args | grep -E "wheel_odom_sim|sketch_pf_node" | grep -vE "grep|bash -c" | awk '{print $1}')
      [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 2
    done
  fi
  (ros2 run robo_project runner_node $REMAP > $D/${nm}_runner.log 2>&1 &)
  timeout 60 bash -c "until grep -q 'MAP BROADCAST' $D/${nm}_runner.log 2>/dev/null; do sleep 1; done"
  (ros2 run robo_project fusion_obstacle_node $REMAP $FUSION_LOC > $D/${nm}_fusion.log 2>&1 &)
  timeout 90 bash -c "until grep -q 'aligned with /map' $D/${nm}_fusion.log 2>/dev/null; do sleep 1; done"
}
counts() {
  local nm=$1
  echo "  runner stalls: $(grep -c 'Stalled' $D/${nm}_runner.log)  NO PATH: $(grep -c 'NO PATH' $D/${nm}_planner.log)" \
       " stuck: $(grep -ci 'stuck' $D/${nm}_planner.log)  sim contacts: $(grep -c 'blocked by the navmesh' $D/${nm}_bridge.log)" \
       " goal shifted: $(grep -c 'Driving to' $D/${nm}_planner.log)"
  grep -m1 "Fusion model" $D/${nm}_fusion.log | sed 's/.*Fusion model /  model: /; s/ on .*//'
  grep -m1 -i "trained for another" $D/${nm}_fusion.log | sed 's/^/  WARNING /'
}
stopall
echo "=== A: 6 goals  (bridge args: ${BARGS:-none}${LOC:+, localization})"
start A "-4.10 4.90 0.0"
(ros2 bag record -o $D/bag /odom /tf /tf_static /cmd_vel /map /planned_path /goal_pose /fusion/local_grid /fusion/obstacle_map /livox/lidar /localization/odom /wheel_odom > $D/A_bag.log 2>&1 &)
sleep 2
(python3 -m robo_project.planning.motion_planner $REMAP $PLAN_LOC > $D/A_planner.log 2>&1 &)
timeout 1100 python3 $T/goal_sequencer.py $REMAP > $D/A_seq.log 2>&1
sed -n '/SUMMARY/,/DONE/p' $D/A_seq.log | grep goal
counts A
grep "Driving to" $D/A_planner.log | sed 's/.*Goal/  shifted: Goal/'
stopall
echo "=== B: former freeze spot (-4.29,2.12) -> goal (-3.84,-0.78)"
start B "-4.29 2.12 -0.82" -p start_x:=-4.29 -p start_y:=2.12 -p start_yaw:=-0.82
(python3 -m robo_project.planning.motion_planner $REMAP $PLAN_LOC > $D/B_planner.log 2>&1 &)
timeout 200 python3 $T/one_goal.py -3.84 -0.78 150 $REMAP 2>&1 | grep robot
counts B
stopall
echo "=== analysis (A)"
python3 $T/analyze_collisions.py $D/bag src/environment/IMR_lab8.navmesh $D/analysis 2>&1 | grep -A10 '^1\. SIM' | head -11
python3 src/robo_project/ml/tools/eval_model.py $D/bag src/environment/IMR_lab8.navmesh src/robo_project $D/analysis 2>&1 \
  | grep -E "cell3|FusionOccNet|LiDAR only" | head -4
echo "=== driving smoothness (A): stops on the way"; /usr/bin/python3 $T/stop_analysis.py $D/bag 2>&1 | grep -E "^ALL|^goal"
[ -n "$REMAP" ] && /usr/bin/python3 $T/loc_error.py $D/bag $D/analysis 2>&1 | grep -vE "^\[" 
echo "all stopped"
