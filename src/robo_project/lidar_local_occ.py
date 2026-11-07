"""
Range sensor (LiDAR, depth camera) -> egocentric local occupancy grid for CMN.
No ROS imports: shared by the ROS nodes, the simulators, data collection and training, so every
stage computes the geometric grids in exactly the same way.

The grid covers a square of side `extent` metres centred on the robot. Robot forward (+x of the
base frame) is the top row, robot left (+y) is the left column, 1 = free, 0 = occupied,
0.5 = unknown (outside the sensor's field of view). That is the same convention as the paper's
learned predictor and the Task CG ground truth.

Free space is found per bearing (polar sweep): along each bearing, everything closer than the
nearest obstacle return is free, everything beyond it is occupied. Behind a wall is treated as
occupied rather than unknown, because coarse maps mark the outside of rooms as occupied too.
"""

import numpy as np


class RangeLocalOcc:
    def __init__(self, grid_size: int = 64, extent: float = 1.2, z_min: float = 0.10, z_max: float = 1.50,
                 max_range: float = 15.0, footprint=(-0.05, 0.05, 0.05), n_bearings: int = 360,
                 robot_half_width: float = 0.5, fov: float = 2 * np.pi, max_gap: int = 2,
                 min_range: float = 0.15):
        """
        @param extent    side of the local grid in metres. Should be 3 x coarse cell size (3 x MPP),
                         because CMN compares the grid with a 3x3 patch of the coarse map.
        @param z_min, z_max  height band (metres above the ground / base_footprint) counted as obstacles.
                         Points below z_min are ground, above z_max are ceiling/overhangs.
        @param max_range bearings without an obstacle return are free up to this range.
        @param footprint (x_min, x_max, y_half) box in the base frame: points inside are the robot's own
                         body and are ignored. VineBot is large, so set this to its real outline.
        @param robot_half_width  half width of the corridor used for the front clearance.
        @param fov       horizontal field of view centred on +x (2*pi for a 360 deg LiDAR, ~1.5 rad for
                         the D455 depth camera). Cells outside it are unknown (0.5).
        @param max_gap   empty bearing bins (1 deg each by default) between returns that are filled
                         from their neighbours.
        @param min_range points closer than this to the sensor (3D) are ignored: self-returns from the
                         sensor housing / mount (Mid-360 blind zone ~0.1 m; seen in the real recordings).
        """
        self.n = grid_size
        self.extent = extent
        self.z_min, self.z_max = z_min, z_max
        self.max_range = max_range
        self.footprint = footprint
        self.n_bearings = n_bearings
        self.half_width = robot_half_width
        self.fov = fov
        self.max_gap = max_gap
        self.min_range = min_range
        # Cell centres in the base frame (x forward, y left); row 0 = front, col 0 = left.
        res = extent / grid_size
        coords = (np.arange(grid_size) + 0.5) * res - extent / 2
        self.cell_x = -coords[:, None] * np.ones((1, grid_size))   # rows: +x at the top
        self.cell_y = -coords[None, :] * np.ones((grid_size, 1))   # cols: +y on the left

    def obstacle_points(self, pts_base: np.ndarray, sensor_xy=(0.0, 0.0), sensor_z: float = None) -> np.ndarray:
        """
        Filter Nx3 points (base frame) to the obstacle height band, outside the robot footprint and
        outside the sensor's blind zone (min_range around the sensor; vertical offset used if sensor_z given).
        """
        x, y, z = pts_base[:, 0], pts_base[:, 1], pts_base[:, 2]
        x0, x1, yh = self.footprint
        body = (x >= x0) & (x <= x1) & (np.abs(y) <= yh)
        dz = 0.0 if sensor_z is None else z - sensor_z
        near = (x - sensor_xy[0]) ** 2 + (y - sensor_xy[1]) ** 2 + dz ** 2 < self.min_range ** 2
        keep = (z >= self.z_min) & (z <= self.z_max) & ~body & ~near & np.isfinite(x) & np.isfinite(y)
        return pts_base[keep]

    def compute(self, pts_base: np.ndarray, sensor_xy=(0.0, 0.0), sensor_z: float = None):
        """
        @param pts_base  Nx3 points in the robot base frame (x forward, y left, z up from the ground).
        @param sensor_xy sensor position in the base frame (rays are cast from here).
        @param sensor_z  sensor height in the base frame (for the 3D blind-zone filter; optional).
        @return (grid, front_clearance): grid n x n float32 in [0, 1] (1 = free, 0.5 = unknown), and the
                distance to the nearest obstacle in front of the robot within its width (inf if none).
        """
        obs = self.obstacle_points(pts_base, sensor_xy, sensor_z)
        sx, sy = sensor_xy
        nearest = np.full(self.n_bearings, np.inf, dtype=np.float64)
        if len(obs):
            ang = np.arctan2(obs[:, 1] - sy, obs[:, 0] - sx)
            rng = np.hypot(obs[:, 0] - sx, obs[:, 1] - sy)
            b = ((ang + np.pi) / (2 * np.pi) * self.n_bearings).astype(int) % self.n_bearings
            np.minimum.at(nearest, b, rng)
            # Sparse sampling (few LiDAR rays, sub-sampled depth pixels) leaves single empty bins
            # between returns; fill gaps of up to max_gap bins from the neighbours so free space
            # does not leak through walls.
            filled = nearest.copy()
            for k in range(1, self.max_gap + 1):
                cand = np.minimum(np.roll(nearest, k), np.roll(nearest, -k))
                upd = ~np.isfinite(filled) & np.isfinite(cand)
                filled[upd] = cand[upd]
            nearest = filled
        nearest = np.where(np.isfinite(nearest), nearest, float(self.max_range))
        cell_ang = np.arctan2(self.cell_y - sy, self.cell_x - sx)
        cell_rng = np.hypot(self.cell_x - sx, self.cell_y - sy)
        cb = ((cell_ang + np.pi) / (2 * np.pi) * self.n_bearings).astype(int) % self.n_bearings
        grid = (cell_rng < nearest[cb]).astype(np.float32)
        if self.fov < 2 * np.pi - 1e-6:
            grid[np.abs(cell_ang) > self.fov / 2] = 0.5

        front = obs[(obs[:, 0] > 0) & (np.abs(obs[:, 1]) <= self.half_width)] if len(obs) else obs
        clearance = float(front[:, 0].min()) if len(front) else float("inf")
        return grid, clearance


# Backwards-compatible name used by the LiDAR node and the evaluation.
LidarLocalOcc = RangeLocalOcc


# ── sensor geometry helpers ─────────────────────────────────────────────────

def raycast_ranges(free: np.ndarray, res: float, row_m: float, col_m: float, yaw: float,
                   n_rays: int = 720, max_range: float = 10.0, step: float = 0.02):
    """
    2D ray casting on a top-down free-space grid (1 = free; row = south, col = east).
    @param yaw  robot heading in the map frame (0 = east, pi/2 = north)
    @return (angles, ranges): angles relative to the robot (CCW, 0 = forward), range to the first
            non-free pixel, inf where nothing is hit within max_range.
    """
    ang = np.linspace(-np.pi, np.pi, n_rays, endpoint=False)
    rng = np.arange(step, max_range, step)
    world = yaw + ang[:, None]
    rows = ((row_m - np.sin(world) * rng[None, :]) / res).astype(int)
    cols = ((col_m + np.cos(world) * rng[None, :]) / res).astype(int)
    inside = (rows >= 0) & (cols >= 0) & (rows < free.shape[0]) & (cols < free.shape[1])
    hit = ~inside.copy()
    hit[inside] = ~free[rows[inside], cols[inside]].astype(bool)
    any_hit = hit.any(axis=1)
    ranges = np.where(any_hit, rng[np.argmax(hit, axis=1)], np.inf)
    return ang, ranges


def ego_crop(free: np.ndarray, res: float, row_m: float, col_m: float, yaw: float,
             extent: float, n: int) -> np.ndarray:
    """
    Ground-truth egocentric local grid: crop a top-down free-space map (row = south, col = east)
    around the robot, rotated so robot-forward is the top row and robot-left the left column.
    Outside the map counts as occupied. Returns n x n float32, 1 = free.
    """
    res_out = extent / n
    coords = (np.arange(n) + 0.5) * res_out - extent / 2
    fx = -coords[:, None] * np.ones((1, n))           # forward
    ly = -coords[None, :] * np.ones((n, 1))           # left
    east = fx * np.cos(yaw) - ly * np.sin(yaw)
    north = fx * np.sin(yaw) + ly * np.cos(yaw)
    rows = np.floor((row_m - north) / res).astype(int)
    cols = np.floor((col_m + east) / res).astype(int)
    inside = (rows >= 0) & (cols >= 0) & (rows < free.shape[0]) & (cols < free.shape[1])
    out = np.zeros((n, n), dtype=np.float32)
    out[inside] = free[rows[inside], cols[inside]]
    return out


def scan_to_points(angles: np.ndarray, ranges: np.ndarray, heights=(0.3, 0.8)) -> np.ndarray:
    """Planar scan -> Nx3 points in the base frame, replicated at a few obstacle heights."""
    ok = np.isfinite(ranges)
    a, r = angles[ok], ranges[ok]
    return np.concatenate([np.stack([r * np.cos(a), r * np.sin(a), np.full_like(r, h)], axis=1)
                           for h in heights], axis=0)


def depth_to_points(depth: np.ndarray, hfov: float, cam_height: float, cam_x: float = 0.0,
                    max_depth: float = 6.0, stride: int = 2) -> np.ndarray:
    """
    Forward-facing pinhole depth image (metres along the optical axis) -> Nx3 base-frame points.
    @param hfov  horizontal field of view (rad); square pixels assumed.
    """
    d = depth[::stride, ::stride].astype(np.float32)
    h, w = d.shape
    f = (w / 2) / np.tan(hfov / 2)
    u = np.arange(w)[None, :] - (w - 1) / 2
    v = np.arange(h)[:, None] - (h - 1) / 2
    ok = np.isfinite(d) & (d > 0.05) & (d < max_depth)
    x = d
    y = -u / f * d
    z = cam_height - v / f * d
    return np.stack([x[ok] + cam_x, y[ok], z[ok]], axis=1)
