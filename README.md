# VineBot – Habitat simulation with ML obstacle avoidance (ROS 2 Humble)

Habitat (IMR Lab 8) simulation of the VineBot with an A* planner, a path follower and a
camera + LiDAR fusion model (FusionOccNet) that predicts a local occupancy grid. The model's
predictions are fused into an obstacle map that the planner avoids.

Start-up commands for every terminal are in [`src/robo_project/terminal commands`](src/robo_project/terminal%20commands).

## Packages
- `src/robo_project` – Habitat bridge, `runner_node` (path follower), `motion_planner` (A*),
  `fusion_obstacle_node` (ML model -> `/fusion/obstacle_map`), model code and sensor rig
- `src/vinebot_description` – URDF, meshes, RViz config
- `src/vineyard_world` – Gazebo world (meshes/textures not included)
- `tools/` – collision analysis, model evaluation and goal-replay scripts for recorded runs

## Not included (too large for this repo)
- Fusion model checkpoint: put it at `runs/fusion_full_e1.2/best.pt` (72 MB)
- Gazebo vineyard meshes and textures (`src/vineyard_world/mesh`, `materials`)
- Other model files (`*.pt`, `*.pth`) and the Denmark scene

## Build
```bash
conda activate habitat
colcon build --packages-select robo_project vinebot_description \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```
Run `motion_planner` with the conda Python (see the terminal commands file) so the
numba-compiled A* is used.
