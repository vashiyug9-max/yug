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

## Navigation test (`sim/tools/run_regression.sh`, results in `results/final_*`)
| drive model | 6 goals | contacts | body overlap | stalls / no path | former freeze spot |
|---|---|---|---|---|---|
| ideal (default) | 6/6 | 0 | 0.0% | 0 / 0 | 12 s, 2.9 m |
| tracked (`-p drive_model:=tracked`) | 6/6 | 0 | 0.0% | 0 / 0 | 14 s, 2.8 m |

Goals closer than 0.70 m to an obstacle are moved (max 0.5 m) to where the whole body fits;
RViz "Goal" shows where the robot will park. The tracked model (acceleration limits, 75% turn
efficiency) uses estimated values: measure them on the robot.

## Connecting the real robot
- `/livox/lidar` in the sensor frame is moved to `base_footprint` by `fusion_obstacle_node` with the rig
  (x 0.395, height 0.45, yaw 90). If the livox_ros_driver2 config already applies the 90 deg extrinsic,
  set `lidar.yaw_deg: 0`. `ml/tools/real_lidar_check.py` checks the yaw from a recording.
- The robot's own LiDAR returns (deck, e-stop) are removed (`common/rig.py: drop_self_hits`).
- Open points: the measured LiDAR low edge varies -3.6 to -7.4 deg around the circle (mount tilted
  ~2 deg?); FOVs and topics of the left / right / back cameras are not measured yet.

## Layout – one folder per team task (`src/robo_project/`)
| Folder | Contents |
|---|---|
| `sim/` | `habitat_bridge.py` (Habitat sim: cameras, depth, LiDAR, odometry, drive model), `lidar_sim.py` (Mid-360 model), `tools/` (regression test, goal replay, collision analysis, rig display) |
| `ml/` | `fusion/` (model, `fusion_obstacle_node`, `collect_data.py`, `fusion_data.py`, `train_fusion.py`), `pose_estimation/`, `tools/` (model evaluation, real LiDAR check) |
| `localization/` | particle filter + validation / dataset replay nodes |
| `cmn/` | Coarse Map Navigation (CMN): interface, ported core, `topo_map`, its own planner / pure pursuit / visualizer |
| `planning/` | `motion_planner.py` (A* node), `astar.py`, `fast_astar.py` (numba A*, also with per-cell costs) |
| `ros/` | glue nodes: `runner_node.py` (path follower + map), `map_server_node.py`, `pose_topic_bridge.py` |
| `common/` | shared helpers: `map_handler.py` (sketch -> map), `basic_types.py`, `rig.py`, `rotated_rect_crop.py` |
| `config/`, `launch/` | settings, maps, sensor rig (`vinebot_rig.yaml`; `vinebot_rig_training_1m.yaml` = old placeholder rig) |

Other packages: `src/vinebot_description` (URDF, meshes, RViz configs incl. a 3D follow view,
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
colcon build --packages-select robo_project vinebot_description \
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
