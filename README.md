# VineBot – Habitat simulation with ML obstacle avoidance (ROS 2 Humble)

Habitat simulation (IMR Lab 8) of the VineBot. A hand-drawn sketch is the map; an A* planner and a
path follower drive the robot; a camera + LiDAR fusion model (FusionOccNet) predicts the local
occupancy around the robot. What the model sees overrides the sketch (the drawing can be ~1 m off):
obstacles it sees are avoided, free space it sees is used even where a wall is drawn.

**How to build and run:** [`src/robo_project/terminal commands`](src/robo_project/terminal%20commands)
(one terminal per step, plus recording and analysis).

## Layout – one folder per team task (`src/robo_project/`)
| Folder | Contents |
|---|---|
| `sim/` | `habitat_bridge.py` (Habitat sim: cameras, depth, simulated LiDAR, odometry), `tools/` (goal replay, collision analysis) |
| `ml/` | `fusion/` (fusion model + `fusion_obstacle_node`), `pose_estimation/` (ML pose estimator + nodes), `tools/eval_model.py` |
| `localization/` | particle filter + validation / dataset replay nodes |
| `cmn/` | Coarse Map Navigation (CMN): interface, ported core, `topo_map`, its own planner / pure pursuit / visualizer |
| `planning/` | `motion_planner.py` (A* node), `astar.py`, `fast_astar.py` (numba A*, also with per-cell costs) |
| `ros/` | glue nodes: `runner_node.py` (path follower + map), `map_server_node.py`, `pose_topic_bridge.py` |
| `common/` | shared helpers: `map_handler.py` (sketch -> map), `basic_types.py`, `rig.py`, `rotated_rect_crop.py` |
| `config/`, `launch/` | settings, maps, sensor rig (`vinebot_rig.yaml`), launch files |

Other packages: `src/vinebot_description` (URDF, meshes, RViz config, Gazebo/RTAB-Map launch),
`src/vineyard_world` (Gazebo vineyard world). Data: `src/environment` (Habitat scene + navmesh),
`src/resources/sketch_map_imr_lab8.jpeg` (the hand-drawn map). `results/`: reports of recorded test runs.

## Not included (too large for this repo)
- Fusion model checkpoint: put it at `runs/fusion_full_e1.2/best.pt` (72 MB)
- ML pose estimator weights `src/robo_project/ml/pose_estimation/*.pth` (2 x 43 MB)
- Gazebo vineyard meshes and textures (`src/vineyard_world/mesh`, `materials/textures`)
- Denmark scene (training only)

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
