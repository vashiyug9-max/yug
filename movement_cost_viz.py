#!/usr/bin/env python3
"""
Movement Cost Visualization
Shows 2 paths: Cardinal raw, Cardinal merged
Diagonal panel removed.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from heapq import heappush, heappop

from pipeline import load_map, validate, compute_heuristic_vector, merge_straight, MAP_FILE, START, GOAL

OUTPUT = "movement_cost_visualization.png"


def run_astar(grid, start, goal, heuristic_map, allow_diagonal=False):
    H, W  = grid.shape
    moves = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0)]

    counter   = 0
    heap      = []
    heappush(heap, (0.0, counter, start))
    g_scores  = {start: 0.0}
    came_from = {}
    closed    = set()

    while heap:
        _, _, cur = heappop(heap)
        if cur in closed: continue
        closed.add(cur)
        if cur == goal:
            path = []
            c2   = cur
            while c2 is not None:
                path.append(c2)
                c2 = came_from.get(c2)
            return path[::-1], g_scores[goal]

        cr, cc = cur
        for dr, dc, cost in moves:
            nr, nc = cr+dr, cc+dc
            nb = (nr, nc)
            if not (0<=nr<H and 0<=nc<W): continue
            if grid[nr,nc] == 0:           continue
            if nb in closed: continue
            ng = g_scores[cur] + cost
            if ng >= g_scores.get(nb, 1e9): continue
            g_scores[nb]  = ng
            came_from[nb] = cur
            h = float(heuristic_map[nr,nc])
            if np.isinf(h): h = math.hypot(goal[0]-nr, goal[1]-nc)
            counter += 1
            heappush(heap, (ng+h, counter, nb))

    return None, None


def main():
    print("=" * 50)
    print("MOVEMENT COST VISUALIZATION")
    print(f"MAP_FILE : {MAP_FILE}")
    print(f"START    : {START}")
    print(f"GOAL     : {GOAL}")
    print("=" * 50)

    grid  = load_map(MAP_FILE)
    start = validate(grid, START, "START")
    goal  = validate(grid, GOAL,  "GOAL")

    heuristic_map = compute_heuristic_vector(grid, goal)

    if np.isinf(heuristic_map[start]):
        fr    = np.argwhere(np.isfinite(heuristic_map) & (grid==1))
        dists = np.sqrt((fr[:,0]-start[0])**2+(fr[:,1]-start[1])**2)
        start = tuple(fr[np.argmin(dists)])
        print(f"Start snapped to: {start}")

    # Run A* cardinal only
    path_cardinal, cost_cardinal = run_astar(grid, start, goal, heuristic_map, allow_diagonal=False)

    # Merge straight segments
    from astar import PosePixels
    cardinal_as_pose = [PosePixels(p[0], p[1]) for p in path_cardinal]
    merged_poses     = merge_straight(cardinal_as_pose)
    path_merged      = [(int(p.r), int(p.c)) for p in merged_poses]
    cost_merged      = cost_cardinal

    print(f"\nCardinal raw    : {len(path_cardinal)} pts, cost={cost_cardinal:.2f}")
    print(f"Cardinal merged : {len(path_merged)} pts, cost={cost_merged:.2f}")

    # Only 2 panels now
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#0f0f23')

    titles = [
        f"Cardinal Raw\ncost={cost_cardinal:.1f}  {len(path_cardinal)} pts",
        f"Cardinal Merged (our pipeline)\ncost={cost_merged:.1f}  {len(path_merged)} pts",
    ]
    paths  = [path_cardinal, path_merged]
    colors = ['#e74c3c', '#f39c12']

    for idx, (ax, path, color, title) in enumerate(zip(axes, paths, colors, titles)):
        ax.set_facecolor('#1a1a2e')
        ax.imshow(grid, cmap='gray', origin='upper',
                  interpolation='nearest', alpha=0.6)

        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, color=color, linewidth=2.5,
                label=f'{len(path)} waypoints', zorder=4)
        ax.scatter(px, py, color=color, s=25, zorder=5)

        # Draw arrows for merged path
        if idx == 1:
            for i in range(len(path)-1):
                p1 = path[i]; p2 = path[i+1]
                ax.annotate("", xy=(p2[1], p2[0]), xytext=(p1[1], p1[0]),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))

        ax.scatter(start[1], start[0], c='lime', s=200,
                   marker='*', zorder=6, label='Start')
        ax.scatter(goal[1],  goal[0],  c='red',  s=200,
                   marker='*', zorder=6, label='Goal')
        ax.set_title(title, fontsize=11, color='white', pad=8)
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(
        f"A* Movement Cost Comparison  |  Map: {MAP_FILE}\n"
        f"Cardinal movement only (cost=1.0)",
        fontsize=12, fontweight='bold', color='white'
    )
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, facecolor='#0f0f23')
    plt.close()
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()

