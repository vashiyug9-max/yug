# CMN end-to-end integration preview — 10 November 2026

This is a follow-on to the already-installed `install_cmn_debug_integration.sh`.
It keeps the existing `fusion_mid360_e3.0/best.pt`, corrected experimental
Global Search implementation (STEP 349/361), sketch map, existing
`runner_node`, and existing `motion_planner` **unchanged**.

## End-to-end flow

```
Habitat 4 RGB + front depth + Mid360 LiDAR  -->  fusion_obstacle_node
                     64x64 /fusion/local_grid
                             |
Existing CMN Global Search + sketch_map_imr_lab8.jpeg
              /localization/provisional_pose
                             |
          localization_navigation_bridge
     (only RELATIVE odom motion since last ML pose)
                   /localization/nav_odom
                             |
             +---------------+---------------+
             |                               |
    existing motion_planner             existing runner_node
       (odom remapped)                  (odom remapped)
             |                               |
           /planned_path ------> /localization/proposed_cmd_vel
                                             |
                                simulation-only GT motion gate
                                             |
                                   /cmd_vel (DISABLED by default)
```

Visuals:
- Existing hand-drawn map: `/map` (RViz Fixed Frame = `map`)
- Estimated robot (green if stable, yellow otherwise): `/localization/robot_marker`
- Raw unfiltered global-search estimate (red): `/localization/raw_pose_marker`
- Selected goal (blue): `/localization/goal_marker`
- A* path: `/planned_path`
- The RobotModel/`base_footprint` TF still shows simulator odom: **use marker for ML estimated pose.**

## Preconditions

The installed STEP 367/368A ROS 2 package works with ordinary colcon build,
not symlink install (`package_dir={package_name:'.'}`). This installer adds
standalone Python modules and a launch file, plus package registrations.

Before launch, stop **old** STEP 361 scripts, STEP 367 debug launch, old
runner_node, old motion_planner, and map server to avoid duplicates. Also stop
existing Habitat/Fusion if launching them from this new launch.

### One-launch full simulation — use only after stopping older duplicates

```bash
cd /home/jd1152/robotics_project/robo_project_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
ros2 launch robo_project cmn_full_navigation_preview.launch.py \
  start_habitat:=true start_fusion:=true evaluate_gt:=true enable_motion:=false
```

If Habitat and Fusion are already running, use `start_habitat:=false
start_fusion:=false` (but stop old localization and navigation nodes).

### Verify that planning consumes localization, not GT odom

```bash
ros2 node info /motion_planner
ros2 node info /runner_node
ros2 topic info /localization/nav_odom -v
ros2 topic info /localization/proposed_cmd_vel -v
ros2 topic info /cmd_vel -v
ros2 topic echo /localization/nav_odom --once
```

Both `/motion_planner` and `/runner_node` must subscribe to
`/localization/nav_odom` and **not** `/odom`. The runner publishes to
`/localization/proposed_cmd_vel`, not `/cmd_vel`.

### RViz

```bash
rviz2 -d src/vinebot_description/config/display_vinebot.rviz
```

Set **Fixed Frame: `map`**. Add displays: **Map** topic `/map`, **Marker**
`/localization/robot_marker`, **Marker** `/localization/raw_pose_marker`,
**Marker** `/localization/goal_marker`, and **Path** `/planned_path`.
Use **2D Goal Pose** to publish a goal on `/goal_pose` in `map` frame.
You can also set it through ROS CLI (replace numbers with a *free* map cell):

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, pose: {position: {x: -3.0, y: 3.0}, orientation: {w: 1.0}}}'
```

The goal may be shifted by the existing planner for clearance. The blue marker
represents the effective goal (`/motion_planner_goal`).

## Movement safety

By default `enable_motion:=false`: there is **no** publisher from this new
gate on the real `/cmd_vel`. The existing runner computes a path and proposed
commands, but cannot move Habitat through this pipeline. This intentionally
lets the team debug with known 1.8–9.3m position discrepancies and large yaw
errors. To test simulated movement once localization has been aligned and
validated, there is an explicit simulation-only `enable_motion:=true` mode. It
requires a fresh explicitly selected goal; at least 3 stable global matches;
fresh command, reference and estimated odom; and position discrepancy <0.35m
and yaw discrepancy <15deg against simulation `/odom`, and caps speeds at
0.12m/s and 0.3rad/s. Errors seen in STEP 364/365 will BLOCK motion.

**Do not use the GT gate on a real robot**, where absolute reference is not
available; use an independently validated real-robot safety system. The GT
safety gate does not supply the navigation pose. The localizer does not use
absolute odom; the bridge uses only incremental delta motion. Stop autonomous
motion if localization jumps or timestamps become stale.

## Existing limitations, still unresolved

1. Global Search is only 0/90/180/-90 degrees. Actual yaw can be arbitrary.
2. Global Search jumps between distant hypotheses; continuity is currently an
   outlier-rejection preview with relative odometry, **not a full PF**.
3. Sketch `map` and simulator `/odom` have not been accurately registered; GT
   error comparisons assume alignment.
4. Existing Fusion *world* obstacle map integrates using simulator odom and
   cannot be used for map-aligned navigation as-is. This launch deliberately
   disables planner/runner subscriptions to `/fusion/obstacle_map`. For final
   robot, project local ML obstacles using the validated localized pose.
5. Planner may find an invalid physical path because the hand-drawn map is
   coarse. Planning-only preview is not a safety assurance.
6. If localization is wrong, a path from the wrong initial pose will be wrong.

Next: STEP 366 orientation/frame audit; continuous-yaw and multi-hypothesis
PF; ML-grid projection in map frame using localized pose; validate goal driving.
