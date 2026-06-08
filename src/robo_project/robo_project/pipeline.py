#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

from astar import Astar, Cell, PosePixels

MAP_FILE = "curve_map.npy"
OUTPUT   = "pipeline_result.png"

START = (16, 11)   # (row, col) — where robot starts
GOAL  = (5, 14)  # (row, col) — where robot needs to go

def load_map(fname):
    raw  = np.load(fname)
    raw  = np.array(raw, dtype=float)
    if raw.max() > 1:
        raw = raw / raw.max()
    grid = np.round(raw).astype(int)
    print(f"Map loaded     : {grid.shape[0]} x {grid.shape[1]}")
    print(f"Free cells     : {np.sum(grid==1)}")
    print(f"Obstacles      : {np.sum(grid==0)}")
    return grid

def validate(grid, pos, name):
    H, W = grid.shape
    r, c = pos
    if not (0 <= r < H and 0 <= c < W):
        raise ValueError(
            f"{name} {pos} is outside map bounds! "
            f"Valid range: row 0-{H-1}, col 0-{W-1}"
        )
    if grid[r, c] == 0:
        free    = np.argwhere(grid == 1)
        dists   = np.sqrt((free[:,0]-r)**2 + (free[:,1]-c)**2)
        snapped = tuple(free[np.argmin(dists)])
        print(f"WARNING: {name} {pos} is inside a wall!")
        print(f"  Snapped to nearest free cell: {snapped}")
        return snapped
    return pos

def compute_heuristic_vector(grid, goal):
    print("\n--- Heuristic Vector ---")
    H, W = grid.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    dist[goal] = 0.0
    q = deque([goal])

    directions = [
        (1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0),
        (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)
    ]
    while q:
        r, c = q.popleft()
        for dr, dc, cost in directions:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            if grid[nr, nc] == 0:                  continue
            nd = dist[r,c] + cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                q.append((nr, nc))

    reachable = np.sum(np.isfinite(dist))
    print(f"Reachable cells: {reachable} / {np.sum(grid==1)}")
    print(f"Max distance   : {np.max(dist[np.isfinite(dist)]):.1f} cells")
    return dist

def visualize(grid, start, goal, raw_path, smooth_path, heuristic_map):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    h_show   = heuristic_map.copy()
    h_show[np.isinf(h_show)] = 0
    h_masked = np.ma.masked_where(grid == 0, h_show)
    im = ax.imshow(h_masked, cmap="viridis_r", origin="upper",
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Distance to goal")
    ax.scatter(goal[1],  goal[0],  c="red",  s=200, marker="*",
               zorder=6, label="Goal")
    ax.scatter(start[1], start[0], c="lime", s=200, marker="*",
               zorder=6, label="Start")
    ax.set_title("Heuristic Vector\nDistance to goal from every cell",
                 fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.imshow(grid, cmap="gray", origin="upper", interpolation="nearest")
    px = [p.c for p in raw_path]
    py = [p.r for p in raw_path]
    ax.plot(px, py, "b-o", markersize=4, linewidth=1.5,
            label=f"Raw A* ({len(raw_path)} pts)")
    ax.scatter(start[1], start[0], c="lime", s=200, marker="*",
               zorder=6, label="Start")
    ax.scatter(goal[1],  goal[0],  c="red",  s=200, marker="*",
               zorder=6, label="Goal")
    ax.set_title(f"Raw A* Path\n{len(raw_path)} waypoints", fontsize=11)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.imshow(grid, cmap="gray", origin="upper", interpolation="nearest")
    ax.plot(px, py, "b-", linewidth=1, alpha=0.3, label="Raw (faded)")
    sx = [p.c for p in smooth_path]
    sy = [p.r for p in smooth_path]
    ax.plot(sx, sy, color="orangered", linewidth=2.5,
            label=f"Smooth ({len(smooth_path)} pts)")
    ax.scatter(sx, sy, c="orangered", s=60, zorder=5)
    ax.scatter(start[1], start[0], c="lime", s=200, marker="*",
               zorder=6, label="Start")
    ax.scatter(goal[1],  goal[0],  c="red",  s=200, marker="*",
               zorder=6, label="Goal")
    ax.set_title(f"Smooth Path\n{len(raw_path)} → {len(smooth_path)} waypoints",
                 fontsize=11, color="green")
    ax.legend(fontsize=9)

    plt.suptitle(
        f"Planning Pipeline  |  Start: {start}  →  Goal: {goal}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT}")

def main():
    print("=" * 50)
    print("PLANNING PIPELINE  (using astar.py)")
    print("=" * 50)
    print(f"Map  : {MAP_FILE}")
    print(f"START: {START}")
    print(f"GOAL : {GOAL}")
    print("=" * 50)
    grid = load_map(MAP_FILE)
    print("\n--- Validating positions ---")
    start = validate(grid, START, "START")
    goal  = validate(grid, GOAL,  "GOAL")

    if start == goal:
        print("ERROR: START and GOAL are the same cell!")
        return
    heuristic_map = compute_heuristic_vector(grid, goal)
    if np.isinf(heuristic_map[start]):
        print(f"WARNING: START {start} cannot reach GOAL — snapping...")
        fr    = np.argwhere(np.isfinite(heuristic_map) & (grid==1))
        dists = np.sqrt((fr[:,0]-start[0])**2 + (fr[:,1]-start[1])**2)
        start = tuple(fr[np.argmin(dists)])
        print(f"  Snapped to: {start}")
    print("\n--- A* Planner (from astar.py) ---")
    planner = Astar()
    planner.map               = grid
    planner.goal_cell         = PosePixels(goal[0], goal[1])
    planner.heuristic_map     = heuristic_map
    planner.include_diagonals = True
    planner.verbose           = False

    start_px = PosePixels(start[0], start[1])
    goal_px  = PosePixels(goal[0],  goal[1])
    raw_path_reversed = planner.run_astar(start_px, goal_px)

    if raw_path_reversed is None:
        print("No path found!")
        return
    raw_path = raw_path_reversed[::-1]
    smooth_path_reversed = planner.smooth_path(raw_path_reversed)
    smooth_path = smooth_path_reversed[::-1]

    print(f"Raw path    : {len(raw_path)} waypoints")
    print(f"Smooth path : {len(smooth_path)} waypoints")
    try:
        action = planner.get_smooth_next_action(start_px)
        print(f"Next action : {action}")
    except Exception:
        print("Next action : (available in ROS2 mode)")
    visualize(grid, start, goal, raw_path, smooth_path, heuristic_map)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Start       : {start}")
    print(f"Goal        : {goal}")
    print(f"Raw path    : {len(raw_path)} waypoints")
    print(f"Smooth path : {len(smooth_path)} waypoints")
    print(f"Status      : COMPLETE ✓")

if __name__ == "__main__":
    main()
