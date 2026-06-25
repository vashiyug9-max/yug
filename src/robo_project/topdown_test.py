import habitat_sim
import numpy as np
import cv2
from habitat.utils.visualizations import maps
from scipy.ndimage import binary_dilation

SCENE = "~/robo_project_ws/src/environment/Denmark.glb"

ORIGINAL_MAP_PATH = "~/robo_project_ws/src/resouces/topdown_grid_map.png"
INFLATED_MAP_PATH = "~/robo_project_ws/src/resouces/topdown_inflated_grid_map.png"
DIFFERENCE_MAP_PATH = "~/robo_project_ws/src/resouces/topdown_difference_map.png"

BINARY_GRID_PATH = "~/robo_project_ws/src/resouces/binary_grid.npy"
INFLATED_BINARY_GRID_PATH = "~/robo_project_ws/src/resouces/inflated_binary_grid.npy"

MAP_RESOLUTION = 1024
INFLATION_PIXELS = 25

def main():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)

    if not sim.pathfinder.is_loaded:
        print("Navmesh/pathfinder is not loaded.")
        sim.close()
        return

    nav_point = sim.pathfinder.get_random_navigable_point()
    height = float(nav_point[1])

    top_down_map = maps.get_topdown_map(
        sim.pathfinder,
        height=height,
        map_resolution=MAP_RESOLUTION,
        draw_border=True
    )

    # Habitat values:
    # 0 = occupied
    # 1 = free
    # 2 = border

    recolor_map = np.array(
        [
            [120, 0, 140],   # occupied -> purple
            [30, 180, 180],  # free -> cyan
            [0, 0, 0]        # border -> black
        ],
        dtype=np.uint8
    )

    original_color_map = recolor_map[top_down_map]
    cv2.imwrite(ORIGINAL_MAP_PATH, cv2.cvtColor(original_color_map, cv2.COLOR_RGB2BGR))

    # 0 = free, 1 = obstacle
    binary_grid = np.where(top_down_map == 1, 0, 1).astype(np.uint8)
    np.save(BINARY_GRID_PATH, binary_grid)

    obstacle_mask = binary_grid == 1
    inflated_obstacle_mask = binary_dilation(
        obstacle_mask,
        iterations=INFLATION_PIXELS
    )
    inflated_grid = np.where(inflated_obstacle_mask, 1, 0).astype(np.uint8)
    np.save(INFLATED_BINARY_GRID_PATH, inflated_grid)

    inflated_color_map = np.zeros((inflated_grid.shape[0], inflated_grid.shape[1], 3), dtype=np.uint8)
    inflated_color_map[inflated_grid == 0] = [30, 180, 180]
    inflated_color_map[inflated_grid == 1] = [220, 0, 220]

    cv2.imwrite(INFLATED_MAP_PATH, cv2.cvtColor(inflated_color_map, cv2.COLOR_RGB2BGR))

    difference_map = np.zeros((binary_grid.shape[0], binary_grid.shape[1], 3), dtype=np.uint8)
    difference_map[binary_grid == 0] = [30, 180, 180]
    difference_map[binary_grid == 1] = [120, 0, 140]

    new_blocked_area = (inflated_grid == 1) & (binary_grid == 0)
    difference_map[new_blocked_area] = [255, 0, 0]

    cv2.imwrite(DIFFERENCE_MAP_PATH, cv2.cvtColor(difference_map, cv2.COLOR_RGB2BGR))

    print("Saved original map:", ORIGINAL_MAP_PATH)
    print("Saved inflated map:", INFLATED_MAP_PATH)
    print("Saved difference map:", DIFFERENCE_MAP_PATH)
    print("Saved binary grid:", BINARY_GRID_PATH)
    print("Saved inflated binary grid:", INFLATED_BINARY_GRID_PATH)

    sim.close()

if __name__ == "__main__":
    main()
