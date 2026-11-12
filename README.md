# VineBot – Habitat simulation with ML obstacle avoidance (ROS 2 Humble)

Habitat simulation (IMR Lab 8) of the VineBot with its **measured sensor rig** and a **realistic
Livox Mid-360** model built from a real recording. A hand-drawn sketch is the map; an A* planner and a
path follower drive the robot; a camera + LiDAR fusion model (FusionOccNet) predicts the occupancy around
the robot. What the model and the LiDAR see overrides the sketch (the drawing can be ~1 m off).

**How to build, run, test, record, analyse and retrain:**
[`src/robo_project/terminal commands`](src/robo_project/terminal%20commands)

## Robot and sensors (`src/robo_project/config/vinebot_rig.yaml`)
- Robot 1.15 m x 0.60 m, rubber tracks (skid steering).
- Front RealSense D455 (RGB-D) at 0.31 m, 0.18 m behind the front edge; left / right / back RGB cameras
  at 0.40 m in the middle of their side.
- Livox Mid-360 at the **front**, on the D455 post, 0.45 m high (CAD: 0.395 m ahead of the centre),
  mounted turned 90 deg. Its real beam pattern (measured from a 174 s recording with
  `ml/tools/real_lidar_check.py`) looks only 4-7 deg down, and the robot's deck blocks low beams to the
  rear: **it does not see obstacles lower than ~30-40 cm within 2 m**. The simulation reproduces this
  (`sim/lidar_sim.py`), so the model learns to use the cameras / depth for low obstacles.

## Obstacle detection (live runs in the simulation vs ground truth, `ml/tools/eval_model.py`)
| | IoU | missed obstacles | false alarms |
|---|---|---|---|
| **FusionOccNet `runs/fusion_mid360_e3.0`** (realistic LiDAR, 3 m grid) | **92.2%** | 4.8% | 0.9% |
| LiDAR alone (realistic Mid-360) | 69.5% | 25.0% | 1.5% |

On the realistic-LiDAR validation set the new model reaches 93.5% IoU; the previous model
(`fusion_real_rig_e3.0`, trained with a flat simulated LiDAR) drops to 80.4% there.

## Navigation test (`sim/tools/run_regression.sh`, `sim/tools/run_autonomy_test.sh`; results in `results/f_*`)
Six goals around the lab (incl. the narrow goal-2 alcove), judged against the TRUE simulator pose.
| run | goals | contacts | body overlap | closest | stops on the way | localization error |
|---|---|---|---|---|---|---|
| true pose (`results/f_normal`) | 6/6 | 0 | 0 | 4.0 cm | 5 | – |
| localization, wheels + IMU (`results/f_loc_imu`) | 6/6 | 0 | 0 | 5.6 cm | 6 | 0.67 m mean |
| **real-robot launch** with the sim as robot (`results/f_auto_builtin`) | 6/6 | 0 | 0 | 2.0 cm | 4 | 0.58 m mean |
| real-robot launch, robot_localization EKF (`results/f_auto_ekf`) | 6/6 | 0 | 0 | 4.4 cm | 16 | 0.59 m mean |

Path follower (`ros/runner_node.py`): corners up to 75 deg are driven (40-75 deg as a slow tight curve),
sharper turns on the spot. Before turning on the spot it computes the area the body sweeps on the LIVE ML
grid (`/fusion/local_grid`, robot frame - right even when the localized pose is off); blocked -> the other
way round or a searched multi-point turn. Obstacle within 12 cm of the front bumper -> no forward driving.
Remaining stops: turning around when the next goal is behind (tracked robot), the narrow goal-2 passage.
`sim/tools/stop_analysis.py BAG --detail` lists every stop and its cause.

## Localization on the hand-drawn sketch (`localization/sketch_pf_node.py`)
Particle filter: odometry + ML local grid matched to the sketch -> `/localization/odom`; runner, planner and
ML node use it by a remap only (`-r /odom:=/localization/odom`). 4 cm on the true floor plan, ~0.6 m on the
sketch: the sketch itself is drawn up to ~0.8 m off, differently in different areas (a global
shift/rotation/scale correction only removes ~0.1 m). An accurate map (e.g. LiDAR SLAM) would remove it.
Wheel odometry alone gives ~25 deg heading errors on tracks: an IMU is required.

## Real robot (`src/vinebot_bringup`, + `serial`, `ps4_ros2`, `roboteq_controller_ros2` from Vinebot_ros2)
- `real_robot.launch.py`: Roboteq ros2_control, PS4 joystick (drive only while L1 is held), twist_mux
  (joystick > autonomy, `/emergency` lock), our robot model for TF.
- `motion_confirm`: autonomy moves only after "y" for every new goal.
- `autonomy.launch.py`: odometry (`wheel_imu_odom`: wheel distance + IMU gyro, learns the gyro bias at
  standstill; `odometry:=ekf` = robot_localization) + localization + ML + planner + path follower.
- Drivers: `install_robot_drivers.sh` (RealSense, BNO055, IMU tools, USB cameras, Livox SDK2 +
  livox_ros_driver2 in `~/drivers_ws`). Build Livox with the system python (conda breaks it).
- Livox: `/livox/lidar` (rviz_MID360_launch.py, PointCloud2) is moved to `base_footprint` by
  `fusion_obstacle_node` with the rig (x 0.395, height 0.45, yaw 90); keep the driver extrinsic at 0.
- Not tested on hardware yet. Check first: motor directions, IMU sign (`imu_yaw_sign`), camera topics and
  fields of view, start pose. Biggest risk: the ML model has only seen simulated images - record a
  joystick drive and check / fine-tune it on real data. Commands: "REAL ROBOT" in the terminal commands.

## Layout – one folder per team task (`src/robo_project/`)
| Folder | Contents |
|---|---|
| `sim/` | `habitat_bridge.py` (Habitat sim: cameras, depth, LiDAR, odometry, drive model), `lidar_sim.py` (Mid-360 model), `tools/` (regression test, goal replay, collision analysis, rig display) |
| `ml/` | `fusion/` (model, `fusion_obstacle_node`, `collect_data.py`, `fusion_data.py`, `train_fusion.py`), `pose_estimation/`, `tools/` (model evaluation, real LiDAR check) |
| `localization/` | `sketch_pf_node.py` (tracking particle filter on the sketch, used) + earlier global-search nodes (kept for reference) |
| `cmn/` | Coarse Map Navigation (CMN): interface, ported core, `topo_map`, its own planner / pure pursuit / visualizer |
| `planning/` | `motion_planner.py` (A* node), `astar.py`, `fast_astar.py` (numba A*, also with per-cell costs) |
| `ros/` | glue nodes: `runner_node.py` (path follower + map), `map_server_node.py`, `pose_topic_bridge.py` |
| `common/` | shared helpers: `map_handler.py` (sketch -> map), `basic_types.py`, `rig.py`, `rotated_rect_crop.py` |
| `config/`, `launch/` | settings, maps, sensor rig (`vinebot_rig.yaml`; `vinebot_rig_training_1m.yaml` = old placeholder rig) |

Other packages: `src/vinebot_bringup` (real robot launch, joystick gate, move confirmation, wheels+IMU
odometry), `src/serial`, `src/ps4_ros2`, `src/roboteq_controller_ros2`, `src/vinebot_description` (URDF, meshes, RViz configs incl. a 3D follow view,
Gazebo/RTAB-Map launch), `src/vineyard_world` (Gazebo vineyard world). Data: `src/environment` (Habitat
scene + navmesh), `src/resources/sketch_map_imr_lab8.jpeg` (the hand-drawn map). `results/`: test reports.
`runs/*/`: training reports and example predictions.

## Not included (too large for this repo) – share separately, e.g. via Drive
- **Fusion model `runs/fusion_mid360_e3.0/best.pt` (72 MB)** – or retrain it (see below).
  Fallbacks: `runs/fusion_real_rig_e3.0/best.pt`, `runs/fusion_full_e1.2/best.pt`
- ML pose estimator weights `src/robo_project/ml/pose_estimation/*.pth` (2 x 43 MB)
- Training data `data/fusion_mid360` (2.4 GB, regenerate with `collect_data.py`)
- Real LiDAR recording (LVX2, 500 MB), Gazebo vineyard meshes / textures, Denmark scene

## Quick start
```bash
conda activate habitat
cd ~/robo_project_ws
colcon build \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
python3 -m robo_project.sim.habitat_bridge          # then the other terminals, see "terminal commands"
```
Run the bridge and the planner with the conda Python (`python3 -m ...`), not `ros2 run`:
the system Python has no habitat_sim / numba.

## Retraining after a rig change
1. `collect_data.py` (conda env `habitat`): 40k samples with the rig in `vinebot_rig.yaml` (~40/s).
2. `train_fusion.py` (conda env `fusion_train` with CUDA torch, run with `PYTHONNOUSERSITE=1`):
   fine-tune from the previous model; ~5 min per epoch on an RTX 3050.
3. `fusion_obstacle_node` picks the newest model and warns if a model does not match the rig.

Exact commands: "retrain the ML model" in the terminal commands file.
