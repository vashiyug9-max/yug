#!/bin/bash
# Full headless regression test of the simulation stack (no RViz, no camera window).
#
#   src/robo_project/sim/tools/run_regression.sh OUT_DIR ["extra bridge args"]
#   e.g. src/robo_project/sim/tools/run_regression.sh results/test_tracked "-p drive_model:=tracked"
#
# A: the 6-goal sequence (goal_sequencer.py), recorded to OUT_DIR/bag, then analysed:
#    analyze_collisions.py (contacts / body overlap) and eval_model.py (ML accuracy vs ground truth)
# B: the former freeze spot: start (-4.29, 2.12), goal (-3.84, -0.78) (goal inside an obstacle)
# Stop everything else first (the script also kills leftover nodes). Takes ~10 min.
D=$1; BARGS=$2
[ -z "$D" ] && { echo "usage: $0 OUT_DIR [\"extra bridge args\"]"; exit 1; }
cd ~/robo_project_ws && eval "$(conda shell.bash hook)" && conda activate habitat && source install/setup.bash
D=$(realpath -m "$D"); rm -rf "$D"; mkdir -p "$D"
T=src/robo_project/sim/tools
PAT='robo_project\.(sim|planning)\.|lib/robo_project/|goal_sequencer|one_goal|ros2 (run|bag)'
stopall() {
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_regression" | awk '{print $1}')
  [ -n "$P" ] && kill -INT $P 2>/dev/null; sleep 4
  P=$(ps -eo pid,args | grep -E "$PAT" | grep -vE "grep|bash -c|run_regression" | awk '{print $1}')
  [ -n "$P" ] && kill -9 $P 2>/dev/null; sleep 1
}
start() {   # $1 = run name, rest = more bridge args
  local nm=$1; shift
  (python3 -m robo_project.sim.habitat_bridge --ros-args -p show_window:=false $BARGS "$@" > $D/${nm}_bridge.log 2>&1 &)
  timeout 120 bash -c "until grep -q 'fully initialized' $D/${nm}_bridge.log 2>/dev/null; do sleep 1; done"
  (ros2 run robo_project runner_node > $D/${nm}_runner.log 2>&1 &)
  timeout 60 bash -c "until grep -q 'MAP BROADCAST' $D/${nm}_runner.log 2>/dev/null; do sleep 1; done"
  (ros2 run robo_project fusion_obstacle_node > $D/${nm}_fusion.log 2>&1 &)
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
echo "=== A: 6 goals  (bridge args: ${BARGS:-none})"
start A
(ros2 bag record -o $D/bag /odom /tf /tf_static /cmd_vel /map /planned_path /goal_pose /fusion/local_grid /fusion/obstacle_map /livox/lidar > $D/A_bag.log 2>&1 &)
sleep 2
(python3 -m robo_project.planning.motion_planner > $D/A_planner.log 2>&1 &)
timeout 1100 python3 $T/goal_sequencer.py > $D/A_seq.log 2>&1
sed -n '/SUMMARY/,/DONE/p' $D/A_seq.log | grep goal
counts A
grep "Driving to" $D/A_planner.log | sed 's/.*Goal/  shifted: Goal/'
stopall
echo "=== B: former freeze spot (-4.29,2.12) -> goal (-3.84,-0.78)"
start B -p start_x:=-4.29 -p start_y:=2.12 -p start_yaw:=-0.82
(python3 -m robo_project.planning.motion_planner > $D/B_planner.log 2>&1 &)
timeout 200 python3 $T/one_goal.py -3.84 -0.78 150 2>&1 | grep robot
counts B
stopall
echo "=== analysis (A)"
python3 $T/analyze_collisions.py $D/bag src/environment/IMR_lab8.navmesh $D/analysis 2>&1 | grep -A10 '^1\. SIM' | head -11
python3 src/robo_project/ml/tools/eval_model.py $D/bag src/environment/IMR_lab8.navmesh src/robo_project $D/analysis 2>&1 \
  | grep -E "cell3|FusionOccNet|LiDAR only" | head -4
echo "all stopped"
