# Vineyard Simulation Environment for Autonomous Robot Testing

A modular, procedural Gazebo Sim (gz-sim) world generator for agricultural robotics research. The toolchain takes real or synthetic terrain data, vine plant meshes, and ground textures as inputs and assembles a complete SDF world file ready for robot deployment. The resulting environment supports SLAM, autonomous navigation, path planning, and any other robotics algorithm that requires a realistic outdoor scene.

---

## Overview

Manually authoring Gazebo world files for varied agricultural environments is time-consuming and error-prone. This project separates the environment into independently configurable layers — terrain, texture, and plant layout — so new vineyard configurations can be generated in seconds by changing a few parameters.

**Pipeline summary:**

```
Vineyard Field Analysis
│
├── Texture & Friction Selection ──► Texture PNG
│
├── Real-world terrain data? ──YES──► DEM to PNG conversion ──► Grayscale PNG ──┐
│                            └─NO──► Heightmap PNG Generator ──► Grayscale PNG ──┤
│                                                                                 │
├── Modeling vine plants in Blender ──► Multiple vine plant .obj meshes ──────────┤
│                                                                                 ▼
└─────────────────────────────────────────────────────────────────────────► SDF Generator
                                                                                 │
                                                                                 ▼
                                                                       vineyard_world.sdf
                                                                                 │
Robot URDF ──► Process URDF ──► robot_launch.py ◄────────────────────────────────┘
                                       │
                              launch_sim.launch.py
                                       │
                    ┌──────────────────┼──────────────────┐
               Teleop Twist      ROS-Gazebo Bridge    ROS 2 Control
               Keyboard
```

---

## Repository Structure

```
vineyard_ws/
└── src/
    ├── vineyard_world/
    │   ├── materials/
    │   │   ├── png/            # Heightmap PNGs (real DEM or generated)
    │   │   └── textures/       # Ground texture PNGs (e.g. dirt4.png)
    │   ├── mesh/               # Vine plant meshes (Tree_1.obj … Tree_N.obj)
    │   └── world/              # Output SDF world files
    └── vinebot_description/
        ├── launch/
        │   ├── robot_launch.py
        │   └── launch_sim.launch.py
        └── config/
            ├── my_controller.yaml
            └── gz_bridge.yaml

heighmap_generator.py           # Step 1: Generate synthetic terrain heightmap
sdf_generator.py                # Step 2: Assemble the SDF world file
```

---

## Assets

### Terrain — Heightmap PNG

The terrain is encoded as a **512×512 grayscale PNG** where pixel intensity maps linearly to elevation. Two sources are supported:

- **Real-world DEM data** — export your field's elevation data and convert to a normalised grayscale PNG (e.g. with GDAL or QGIS). This produces terrain that exactly matches a real vineyard's topography.
- **Synthetic generation** — `heighmap_generator.py` produces a smooth arc-slope profile with Gaussian noise, suitable for controlled experiments where a specific terrain shape is needed without real field data.

Both terrain types render correctly in Gazebo Sim with the `ogre2` render engine. The grayscale PNG, ground texture, and resulting heightmap in Gazebo look as follows:

- **(a) Grayscale PNG** — elevation encoded as pixel intensity (0 = lowest, 255 = highest)
- **(b) Texture PNG** — tiling dirt/soil surface material applied over the terrain geometry
- **(c) Generated heightmap** — the assembled terrain as rendered in Gazebo Sim

### Ground Texture

Any tiling PNG can be used as the ground material. The texture path and tiling scale are configured in `sdf_generator.py`. The included `dirt4.png` gives a realistic vineyard soil appearance with visible tilled row structure.

### Vine Plant Meshes

Four vine plant variants were modelled in Blender and exported as `.obj` files (`Tree_1.obj` – `Tree_4.obj`). Each model represents a different vine growth form:

| Variant | Description |
|---------|-------------|
| **Plant 1** | Upright with a dense, rounded canopy |
| **Plant 2** | Arching primary branch with lateral spread |
| **Plant 3** | Low, horizontal spreading growth |
| **Plant 4** | Tall and narrow with an open canopy |

During world generation, each plant slot is assigned a randomly selected mesh, producing natural visual variety across the rows without manual placement.

---

## Scripts

### `heighmap_generator.py` — Synthetic Terrain Generation

Generates a 512×512 grayscale PNG heightmap using a circular arc function to create a gentle vineyard slope, with smoothed Gaussian noise for natural surface variation.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width`, `height` | 512, 512 | Output image resolution (pixels) |
| `x0,y0` / `x1,y1` | (0,10) / (50,0) | Arc start and end points |
| `xc, yc` | (50, 130) | Arc centre point |
| Noise `scale` | 10.0 | Surface roughness amplitude |
| Noise `sigma` | 3 | Gaussian smoothing radius |
| `output_dir` | `src/vineyard_world/materials/png` | Save location |

```bash
python heighmap_generator.py
```

A matplotlib preview is displayed before saving. The output `vineyard_arc_slope_heightmap_bumpy_smooth_512.png` is used directly by `sdf_generator.py`.

---

### `sdf_generator.py` — SDF World Assembly

Reads the heightmap PNG and assembles a complete Gazebo SDF world including terrain, lighting, physics, ground plane, and individually placed vine plant models. Each plant samples its Z position from the heightmap so it sits correctly on the slope regardless of terrain variation. Mesh selection is randomised per plant for visual variety.

**Key parameters (`CONFIG` section at top of file):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HEIGHTMAP_PNG` | `...bumpy_smooth_512.png` | Path to input heightmap PNG |
| `HEIGHTMAP_SIZE_M` | `(50.0, 50.0)` | Real-world terrain extent in metres |
| `HEIGHTMAP_MAX_Z` | `12.0` | Maximum elevation difference in metres |
| `PLANT_MESH_BASE_PATH` | `.../mesh` | Directory containing `Tree_N.obj` files |
| `PLANT_MESH_COUNT` | `4` | Number of mesh variants to randomise over |
| `PLANT_SCALE` | `(1, 1, 1)` | Uniform scale applied to all plant meshes |
| `PLANT_ROLL` | `1.57 rad` | Roll offset applied to every plant |
| `PLANTS_PER_ROW` | `12` | Number of plants per vineyard row |
| `NUM_ROWS` | `5` | Number of vineyard rows |
| `PLANT_SPACING` | `3.0 m` | Along-row spacing between plants |
| `ROW_SPACING` | `10.0 m` | Cross-row spacing between rows |
| `PHYSICS_ENGINE` | `"ode"` | Physics backend: `"ode"` or `"bullet"` |

```bash
# Default output path
python sdf_generator.py

# Custom output path
python sdf_generator.py --out /path/to/my_world.sdf

# Reproducible random mesh assignment
python sdf_generator.py --seed 42
```

Console output reports the total number of trees placed and how many times each mesh variant was used.

---

### `launch_sim.launch.py` — Simulation Launch

ROS 2 launch file that brings up the full simulation stack in the correct order, using `RegisterEventHandler` / `OnProcessExit` to enforce the startup sequence and prevent controller race conditions.

**Startup sequence:**

1. **Robot State Publisher** (`robot_launch.py`) — processes the URDF and publishes `/robot_description` with `use_sim_time: true`.
2. **Gazebo Sim** — launches with the `ogre2` render engine and loads the configured SDF world.
3. **Spawn entity** — spawns `vinebot` from `/robot_description` at the configured initial pose.
4. **Differential drive controller** (`diff_cont`) — spawned after the entity exists, loaded from `my_controller.yaml`.
5. **Joint state broadcaster** (`joint_broad`) — spawned after the drive controller is active.
6. **ROS-Gazebo bridge** — bridges Gazebo topics to ROS 2 via `gz_bridge.yaml` (commented out by default; enable by uncommenting the relevant block).

```bash
# Launch with default world
ros2 launch vinebot_description launch_sim.launch.py

# Launch with a custom world file
ros2 launch vinebot_description launch_sim.launch.py world:=/path/to/custom.sdf
```

---

## Quick Start

**1. Install Python dependencies**

```bash
pip install Pillow numpy scipy matplotlib
```

**2. Generate the heightmap**

```bash
python heighmap_generator.py
```

**3. Generate the SDF world file**

```bash
python sdf_generator.py
```

**4. Build the ROS 2 workspace**

```bash
cd vineyard_ws
colcon build --symlink-install
source install/setup.bash
```

**5. Launch the simulation**

```bash
ros2 launch vinebot_description launch_sim.launch.py
```

---

## Dependencies

### Python

```bash
pip install Pillow numpy scipy matplotlib
```

### ROS 2 / Gazebo

| Package | Purpose |
|---------|---------|
| `ros_gz_sim` | Gazebo Sim ROS 2 integration and entity spawning |
| `ros_gz_bridge` | Topic bridging between ROS 2 and Gazebo |
| `controller_manager` | ROS 2 Control framework |
| `ament_index_python` | ROS 2 package path resolution |

Tested with **ROS 2 Humble** and **Gazebo Sim 7**. The `ogre2` render engine is required for correct heightmap texture rendering.

---

## Customisation Guide

**Swap the terrain** — replace `HEIGHTMAP_PNG` with any grayscale PNG and update `HEIGHTMAP_SIZE_M` and `HEIGHTMAP_MAX_Z` to match the real-world scale of your field.

**Use real DEM data** — convert your field's elevation raster to a normalised 8-bit grayscale PNG using GDAL (`gdal_translate -ot Byte -scale`) or QGIS, then point `HEIGHTMAP_PNG` at it and set the size/elevation parameters accordingly.

**Add more plant variants** — place additional `.obj` files in the mesh directory following the `Tree_N.obj` naming convention and increment `PLANT_MESH_COUNT`.

**Change the vineyard layout** — adjust `PLANTS_PER_ROW`, `NUM_ROWS`, `PLANT_SPACING`, and `ROW_SPACING` to match the geometry of the target field.

**Change the ground texture** — update the `diffuse` and `normal` texture paths inside `build_heightmap_geom()` in `sdf_generator.py` to point to a different tiling PNG.

---

## Robotics Applications

The generated environment is compatible with standard ROS 2 robotics workflows:

- **SLAM** — the structured row layout and varied terrain provide good feature density for LiDAR or visual SLAM algorithms.
- **Autonomous navigation** — the diff-drive controller is pre-configured and compatible with Nav2.
- **Coverage path planning** — the regular row/column plant grid provides a well-defined waypoint structure for between-row traversal research.
- **Teleoperation and manual testing** — connect `teleop_twist_keyboard` or a joystick node directly to the running `diff_cont` controller.

---

## License

This project was developed as part of a Master's thesis. Please contact the author before reuse or redistribution.
