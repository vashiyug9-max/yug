# CMN Localization — Team Debug Integration

## Status (2026-11-10)

This integration deliberately preserves the current **incorrect global localization estimates**. It is for team debugging, **not navigation**. No changes are made to the trained Fusion checkpoint or the particle-filter matching algorithm.

### Data flow

1. Habitat 4 RGB cameras + front depth + Livox LiDAR
2. Existing Fusion ML node -> `/fusion/local_grid` (`nav_msgs/OccupancyGrid`, 64x64, 3.0m extent)
3. Packaged STEP 361 global localizer + hand-drawn sketch -> `/localization/provisional_pose` (`geometry_msgs/PoseStamped`, `cmn_sketch_map` frame)
4. STEP 364 evaluator, simulation only, observes `/odom` as reference and writes CSV under `localization_debug_manual/`

No `odom` or GT subscription is present in the localizer: evaluation is a separate process.

### Current known issues

- STEP 364: GT moved ~0.540m; selected pose jumped ~9.79m.
- STEP 365: similarity at GT coordinates and GT yaw was ~0.0685 versus ~0.6316 at the predicted pose.
- Global search only considers four cardinal headings, so continuous yaw tracking is **not** implemented.
- `cmn_sketch_map` and `odom` coordinate frames have not been registered; direct numerical errors assume alignment.
- Similarity scores are **not probabilities**.
- `runner_node` and `motion_planner` in the inspected code still consume `/odom`; do not assume they use CMN localization.

### Launch with already-running Habitat + Fusion (use this now)

```bash
cd /home/jd1152/robotics_project/robo_project_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 launch robo_project cmn_localization_debug.launch.py start_fusion:=false evaluate_gt:=true
```

**Stop the old manually-running STEP 361 and STEP 364 nodes first** (Ctrl+C). Avoid duplicate publishers and monitors.

### Fresh session, after starting Habitat Bridge separately

```bash
cd /home/jd1152/robotics_project/robo_project_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
ros2 launch robo_project cmn_localization_debug.launch.py \
  start_fusion:=true evaluate_gt:=true \
  model_path:="$PWD/runs/fusion_mid360_e3.0/best.pt" \
  rig:="$PWD/src/robo_project/config/vinebot_rig.yaml"
```

Do not start a second standalone Fusion ML process in this mode. Before every session, verify that exactly one publisher exists on `/fusion/local_grid` and `/localization/provisional_pose`.

### Topic checks

```bash
ros2 topic info /fusion/local_grid
ros2 topic info /localization/provisional_pose
ros2 topic echo /localization/provisional_pose --once
ros2 node info /step361_live_localization
```

### Safety and responsibilities

- Do not connect provisional pose to `/cmd_vel`, navigation, or an A* controller.
- Do not publish a fake TF transform from `cmn_sketch_map` to `odom` or `map`.
- GT `/odom` is **evaluation reference only**, not absolute localization input.
- The evaluator relies on current working directory being the workspace root; launch from there so the CSV directory exists.
- For real robot: `evaluate_gt:=false`, and validate sources, frames, freshness, and safety gates before any motion.

### Next investigation

STEP 366: compare eight local-grid orientations, then map frame alignment, local map/ML mismatch, continuous yaw and sequential multi-hypothesis PF. Keep the existing `.pt` checkpoint unchanged for baseline comparison.
