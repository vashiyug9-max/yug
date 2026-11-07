"""
Compiled (numba) version of Astar.run_astar from astar.py: same neighbours, move costs, corner
rule, heuristic (Chebyshev with diagonals) and tie-breaking, so it returns the same path - only
~100x faster (the pure-Python search took up to ~9 s on the 768 x 1024 map, during which the robot
kept following the old path into a newly seen obstacle).

run_astar(map, start_px, goal_px, include_diagonals=True) -> list of PosePixels, goal -> start
(the same reversed order as Astar.run_astar), or None.
Falls back to the pure-Python Astar if numba is not installed.
"""
import heapq

import numpy as np

from robo_project.common.basic_types import PosePixels

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                           # pragma: no cover
    HAVE_NUMBA = False


if HAVE_NUMBA:
    @njit(cache=True)
    def _astar(free, sr, sc, gr, gc, diag):
        H, W = free.shape
        nbr_r = np.array([0, 0, -1, 1, -1, -1, 1, 1])
        nbr_c = np.array([-1, 1, 0, 0, -1, 1, -1, 1])
        nbr_w = np.array([1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4])
        n_nbr = 8 if diag else 4
        g = np.full((H, W), np.inf)
        closed = np.zeros((H, W), np.bool_)
        parent = np.full((H, W), -1, np.int64)
        g[sr, sc] = 0.0
        heap = [(0.0, 0, sr, sc)]
        counter = 0
        while len(heap) > 0:
            _, _, r, c = heapq.heappop(heap)
            if closed[r, c]:
                continue
            closed[r, c] = True
            if r == gr and c == gc:
                n = 0
                k = r * W + c
                while k >= 0:
                    n += 1
                    k = parent[k // W, k % W]
                out = np.empty((n, 2), np.int64)
                k = r * W + c
                i = 0
                while k >= 0:
                    out[i, 0] = k // W
                    out[i, 1] = k % W
                    i += 1
                    k = parent[k // W, k % W]
                return out
            for j in range(n_nbr):
                dr, dc = nbr_r[j], nbr_c[j]
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= H or nc >= W:
                    continue
                if free[nr, nc] == 0 and free[r, c] != 0:
                    continue
                if dr != 0 and dc != 0:
                    if free[r + dr, c] == 0 or free[r, c + dc] == 0:
                        continue
                if closed[nr, nc]:
                    continue
                new_g = g[r, c] + nbr_w[j]
                if new_g >= g[nr, nc]:
                    continue
                g[nr, nc] = new_g
                parent[nr, nc] = r * W + c
                if diag:
                    h = float(max(abs(gr - nr), abs(gc - nc)))
                else:
                    h = np.hypot(gr - nr, gc - nc)
                counter += 1
                heapq.heappush(heap, (new_g + h, counter, nr, nc))
        return np.empty((0, 2), np.int64)


if HAVE_NUMBA:
    @njit(cache=True)
    def _los(free, r0, c0, r1, c1):
        """Astar._line_of_sight: samples every 0.1 px, checks the floor/ceil neighbours."""
        H, W = free.shape
        dist = np.hypot(r1 - r0, c1 - c0)
        if dist == 0:
            return True
        steps = int(dist / 0.1) + 1
        for i in range(steps + 1):
            t = i / steps
            r = r0 + t * (r1 - r0)
            c = c0 + t * (c1 - c0)
            for ri in (int(np.floor(r)), int(np.ceil(r))):
                for ci in (int(np.floor(c)), int(np.ceil(c))):
                    if not (0 <= ri < H and 0 <= ci < W):
                        return False
                    if free[ri, ci] == 0:
                        return False
        return True

    @njit(cache=True)
    def _smooth(free, pts_fwd):
        """Astar.smooth_path on a start -> goal point array; returns indices of the kept points."""
        n = len(pts_fwd)
        keep = [0]
        i = 0
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if _los(free, float(pts_fwd[i, 0]), float(pts_fwd[i, 1]), float(pts_fwd[j, 0]), float(pts_fwd[j, 1])):
                    break
                j -= 1
            keep.append(j)
            i = j
        return np.array(keep)


def smooth_path(free_map: np.ndarray, path_reversed: list) -> list:
    """Same result as Astar.smooth_path (input and output goal -> start)."""
    if path_reversed is None or len(path_reversed) < 3:
        return path_reversed
    if not HAVE_NUMBA:
        from robo_project.planning.astar import Astar
        a = Astar()
        a.map = free_map
        return a.smooth_path(path_reversed)
    fwd = path_reversed[::-1]
    pts = np.array([[p.r, p.c] for p in fwd], dtype=np.float64)
    idx = _smooth(np.ascontiguousarray(free_map, dtype=np.uint8), pts)
    return [fwd[k] for k in idx][::-1]


def run_astar(free_map: np.ndarray, start_px: PosePixels, goal_px: PosePixels, include_diagonals: bool = True):
    sr, sc = int(start_px.r), int(start_px.c)
    gr, gc = int(goal_px.r), int(goal_px.c)
    H, W = free_map.shape
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if free_map[gr, gc] == 0:
        return None
    if not HAVE_NUMBA:
        from robo_project.planning.astar import Astar
        a = Astar()
        a.map = free_map
        a.include_diagonals = include_diagonals
        return a.run_astar(start_px, goal_px)
    out = _astar(np.ascontiguousarray(free_map, dtype=np.uint8), sr, sc, gr, gc, bool(include_diagonals))
    if len(out) == 0:
        return None
    return [PosePixels(int(r), int(c)) for r, c in out]


# ── A* on a cost map (used when the drawing has no route: walls the robot has not seen yet
#    are passable at a high cost) ──────────────────────────────────────────────────────────
if HAVE_NUMBA:
    @njit(cache=True)
    def _astar_cost(cost, sr, sc, gr, gc):
        """Same search as _astar (8 neighbours, 1.0 / 1.4 steps, no corner cutting, Chebyshev
        heuristic), but each step costs step * cost[cell]; cost = inf is impassable, min cost 1."""
        H, W = cost.shape
        nbr_r = np.array([0, 0, -1, 1, -1, -1, 1, 1])
        nbr_c = np.array([-1, 1, 0, 0, -1, 1, -1, 1])
        nbr_w = np.array([1.0, 1.0, 1.0, 1.0, 1.4, 1.4, 1.4, 1.4])
        g = np.full((H, W), np.inf)
        closed = np.zeros((H, W), np.bool_)
        parent = np.full((H, W), -1, np.int64)
        g[sr, sc] = 0.0
        heap = [(0.0, 0, sr, sc)]
        counter = 0
        while len(heap) > 0:
            _, _, r, c = heapq.heappop(heap)
            if closed[r, c]:
                continue
            closed[r, c] = True
            if r == gr and c == gc:
                n = 0
                k = r * W + c
                while k >= 0:
                    n += 1
                    k = parent[k // W, k % W]
                out = np.empty((n, 2), np.int64)
                k = r * W + c
                i = 0
                while k >= 0:
                    out[i, 0] = k // W
                    out[i, 1] = k % W
                    i += 1
                    k = parent[k // W, k % W]
                return out
            for j in range(8):
                dr, dc = nbr_r[j], nbr_c[j]
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= H or nc >= W:
                    continue
                if np.isinf(cost[nr, nc]):
                    continue
                if dr != 0 and dc != 0:
                    if np.isinf(cost[r + dr, c]) or np.isinf(cost[r, c + dc]):
                        continue
                if closed[nr, nc]:
                    continue
                new_g = g[r, c] + nbr_w[j] * cost[nr, nc]
                if new_g >= g[nr, nc]:
                    continue
                g[nr, nc] = new_g
                parent[nr, nc] = r * W + c
                h = float(max(abs(gr - nr), abs(gc - nc)))
                counter += 1
                heapq.heappush(heap, (new_g + h, counter, nr, nc))
        return np.empty((0, 2), np.int64)


def run_astar_cost(cost_map: np.ndarray, start_px: PosePixels, goal_px: PosePixels):
    """A* on a float cost map (>= 1 per cell, inf = blocked). Path goal -> start, or None.
    Needs numba (no pure-Python fallback: returns None without it)."""
    if not HAVE_NUMBA:
        return None
    sr, sc, gr, gc = int(start_px.r), int(start_px.c), int(goal_px.r), int(goal_px.c)
    H, W = cost_map.shape
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W) or np.isinf(cost_map[gr, gc]):
        return None
    out = _astar_cost(np.ascontiguousarray(cost_map, dtype=np.float64), sr, sc, gr, gc)
    if len(out) == 0:
        return None
    return [PosePixels(int(r), int(c)) for r, c in out]
