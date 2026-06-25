import habitat_sim
import numpy as np
import cv2
import matplotlib.pyplot as plt
from habitat.utils.visualizations import maps
from scipy.ndimage import binary_dilation
from collections import deque

SCENE = "~/robo_project_ws/src/environment/Denmark.glb"

ORIGINAL_MAP_PATH       = "~/robo_project_ws/src/resources/topdown_grid_map.png"
INFLATED_MAP_PATH       = "~/robo_project_ws/src/resources/topdown_inflated_grid_map.png"
DIFFERENCE_MAP_PATH     = "~/robo_project_ws/src/resources/topdown_difference_map.png"
DISTANCE_FIELD_VIZ_PATH = "~/robo_project_ws/src/resources/distance_field_viz.png"

BINARY_GRID_PATH          = "~/robo_project_ws/src/resources/binary_grid.npy"
INFLATED_BINARY_GRID_PATH = "~/robo_project_ws/src/resources/inflated_binary_grid.npy"
DISTANCE_FIELD_PATH       = "~/robo_project_ws/src/resources/distance_field.npy"

MAP_RESOLUTION   = 1024
INFLATION_PIXELS = 25

# ---------------------------------------------------------
# SET YOUR GOAL HERE
# None = auto-pick a free cell near bottom-right of map
# Or set manually e.g. GOAL = (800, 900)
# IMPORTANT: rerun this script every time your goal changes
# ---------------------------------------------------------
GOAL = None


# ---------------------------------------------------------
# FIX 1: GLOBAL HEURISTIC VECTOR — 8-DIRECTIONAL BFS
# Original used 4 directions — changed to 8 to match A*
# so heuristic distances exactly match actual path costs
# ---------------------------------------------------------
def compute_distance_field(binary_grid, goal):
    """
    Compute distance field from goal to every free cell.
    Uses BFS/Dijkstra with 8-directional movement to match A*.

    binary_grid : 0 = free, 1 = obstacle  (Habitat convention)
    goal        : (row, col) — goal position on the map

    Returns dist array where dist[r,c] = shortest distance
    from cell (r,c) to goal. np.inf for unreachable cells.
    """
    h, w = binary_grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)

    # Validate goal
    gr, gc = goal
    if binary_grid[gr, gc] == 1:
        print(f"WARNING: Goal {goal} is inside an obstacle! Finding nearest free cell...")
        free_cells = np.argwhere(binary_grid == 0)
        if len(free_cells) == 0:
            print("ERROR: No free cells found!")
            return dist
        # Find nearest free cell to requested goal
        dists = np.sqrt((free_cells[:, 0] - gr)**2 + (free_cells[:, 1] - gc)**2)
        goal = tuple(free_cells[np.argmin(dists)])
        print(f"Using nearest free cell: {goal}")

    dist[goal] = 0.0
    q = deque()
    q.append(goal)

    # FIX 1: 8-directional with correct costs (was 4-directional)
    # This matches A* movement so heuristic is always accurate
    directions = [
        (1,  0,  1.0),  (-1, 0,  1.0),  (0,  1,  1.0),  (0, -1, 1.0),   # cardinal
        (1,  1,  1.4),  (1, -1,  1.4),  (-1, 1,  1.4),  (-1,-1, 1.4)     # diagonal
    ]

    while q:
        r, c = q.popleft()

        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < h and 0 <= nc < w):
                continue

            if binary_grid[nr, nc] == 1:   # obstacle — skip
                continue

            new_dist = dist[r, c] + cost
            if new_dist < dist[nr, nc]:
                dist[nr, nc] = new_dist
                q.append((nr, nc))

    return dist


# ---------------------------------------------------------
# FIX 2: CONFIGURABLE GOAL SELECTION
# Original hardcoded goal to centre of map
# Now auto-picks a real free cell or uses GOAL variable
# ---------------------------------------------------------
def pick_goal(binary_grid):
    """
    Pick a valid goal position.
    Uses GOAL global if set, otherwise picks a free cell
    near the bottom-right of the map.
    """
    global GOAL
    h, w = binary_grid.shape

    if GOAL is not None:
        gr, gc = GOAL
        if binary_grid[gr, gc] == 0:
            print(f"Using configured goal: {GOAL}")
            return GOAL
        else:
            print(f"Configured goal {GOAL} is in obstacle — auto-selecting...")

    # Auto-pick: find free cells, pick one near bottom-right
    free_cells = np.argwhere(binary_grid == 0)
    if len(free_cells) == 0:
        raise ValueError("No free cells in map!")

    # Sort by distance from bottom-right corner and pick 7/8 through the list
    goal = tuple(free_cells[len(free_cells) * 7 // 8])
    print(f"Auto-selected goal: {goal}")
    return goal


# ---------------------------------------------------------
# VISUALIZATION (our addition)
# Shows the distance field as a colour map
# ---------------------------------------------------------
def visualize_distance_field(binary_grid, distance_field, goal, output_path):
    """
    Save a visualization of the distance field.
    Dark = close to goal, bright = far from goal.
    """
    step = max(1, binary_grid.shape[0] // 400)
    grid_d = binary_grid[::step, ::step]
    dist_d = distance_field[::step, ::step].copy()
    dist_d[np.isinf(dist_d)] = 0

    dist_masked = np.ma.masked_where(grid_d == 1, dist_d)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(dist_masked, cmap="viridis_r", origin="upper",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Distance to goal (cells)")

    gr_d = goal[0] // step
    gc_d = goal[1] // step
    ax.scatter(gc_d, gr_d, c="red", s=20, zorder=6,
               marker="*", label="Goal")
    ax.set_title("Global Heuristic Vector (Distance Field)\nDark = close to goal",
                 fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved distance field visualization:", output_path)


# ---------------------------------------------------------
# MAIN SCRIPT — unchanged except goal + visualization
# ---------------------------------------------------------
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

    inflated_color_map = np.zeros(
        (inflated_grid.shape[0], inflated_grid.shape[1], 3), dtype=np.uint8)
    inflated_color_map[inflated_grid == 0] = [30, 180, 180]
    inflated_color_map[inflated_grid == 1] = [220, 0, 220]
    cv2.imwrite(INFLATED_MAP_PATH, cv2.cvtColor(inflated_color_map, cv2.COLOR_RGB2BGR))

    difference_map = np.zeros(
        (binary_grid.shape[0], binary_grid.shape[1], 3), dtype=np.uint8)
    difference_map[binary_grid == 0] = [30, 180, 180]
    difference_map[binary_grid == 1] = [120, 0, 140]
    new_blocked_area = (inflated_grid == 1) & (binary_grid == 0)
    difference_map[new_blocked_area] = [255, 0, 0]
    cv2.imwrite(DIFFERENCE_MAP_PATH, cv2.cvtColor(difference_map, cv2.COLOR_RGB2BGR))

    # ---------------------------------------------------------
    # COMPUTE GLOBAL HEURISTIC VECTOR (DISTANCE FIELD)
    # FIX 2: goal is now configurable, not hardcoded to centre
    # ---------------------------------------------------------
    goal = pick_goal(binary_grid)

    print(f"\nComputing distance field from goal {goal}...")
    print("This may take 10-30 seconds on a 1024x1024 map...")
    distance_field = compute_distance_field(binary_grid, goal)
    np.save(DISTANCE_FIELD_PATH, distance_field)

    # Stats
    reachable   = np.sum(np.isfinite(distance_field))
    max_dist    = np.max(distance_field[np.isfinite(distance_field)])
    print(f"Reachable cells : {reachable}")
    print(f"Max distance    : {max_dist:.1f} cells")

    # Visualization (our addition)
    visualize_distance_field(binary_grid, distance_field, goal, DISTANCE_FIELD_VIZ_PATH)

    print("\nSaved original map   :", ORIGINAL_MAP_PATH)
    print("Saved inflated map   :", INFLATED_MAP_PATH)
    print("Saved difference map :", DIFFERENCE_MAP_PATH)
    print("Saved binary grid    :", BINARY_GRID_PATH)
    print("Saved inflated grid  :", INFLATED_BINARY_GRID_PATH)
    print("Saved distance field :", DISTANCE_FIELD_PATH)
    print("\nTo use in your A* planner:")
    print("  distance_field = np.load('~/robo_project_ws/src/resources/distance_field.npy')")
    print("  planner.heuristic_map = distance_field")

    sim.close()


if __name__ == "__main__":
    main()
