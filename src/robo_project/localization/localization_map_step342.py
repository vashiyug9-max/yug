"""Localization-only metric map for the new Fusion model.

Uses the existing processed sketch from the team's MapFrameManager,
without modifying that shared class.

Global map: Runner's 0.02 m/cell coordinate convention.
Local grid: 64x64, forward=top, left=left.
Occupancy: 1=free, 0=occupied.

Observation resolution is supplied by the Fusion message.
"""

import math
import cv2
import numpy as np

from robo_project.common.map_handler import MapFrameManager
from robo_project.common.basic_types import PoseMeters, PosePixels


def sample_local_grid(
    global_map, x, y, yaw,
    map_resolution, observation_resolution, size=64
):
    """Extract robot-relative occupancy using metric coordinates."""

    if global_map.ndim != 2:
        raise ValueError("Global map must be 2D")

    if not (np.isfinite(map_resolution)
            and np.isfinite(observation_resolution)):
        raise ValueError("Non-finite resolution")

    if map_resolution <= 0 or observation_resolution <= 0:
        raise ValueError("Resolution must be positive")

    size = int(size)
    if size <= 0:
        raise ValueError("Invalid grid size")

    row, col = np.indices((size, size), dtype=np.float32)

    # ML grid: row 0 = forward; col 0 = left.
    forward = ((size - 1) / 2.0 - row) * observation_resolution
    left = ((size - 1) / 2.0 - col) * observation_resolution

    c = math.cos(float(yaw))
    s = math.sin(float(yaw))

    wx = float(x) + c * forward - s * left
    wy = float(y) + s * forward + c * left

    h, w = global_map.shape

    # ROS /map uses a centred origin, with image rows
    # increasing downward while metric Y increases upward.
    px = wx / map_resolution + w / 2.0 - 0.5
    py = h / 2.0 - 0.5 - wy / map_resolution

    return cv2.remap(
        np.ascontiguousarray(global_map, dtype=np.float32),
        np.ascontiguousarray(px, dtype=np.float32),
        np.ascontiguousarray(py, dtype=np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0
    )


class LocalizationMapFrameManager(MapFrameManager):
    """Map interface compatible with the Step 308 PF candidate."""

    def __init__(self, observation_resolution):
        observation_resolution = float(observation_resolution)

        if not np.isfinite(observation_resolution) \
                or observation_resolution <= 0:
            raise ValueError("Invalid ML observation resolution")

        # Read the EXISTING sketch processing implementation.
        super().__init__(use_discrete_state_space=False)

        raw = np.asarray(self.map, dtype=np.float32)

        if raw.shape != (1024, 768):
            raise ValueError(
                f"Unexpected sketch map shape: {raw.shape}"
            )

        if not np.all((raw == 0) | (raw == 1)):
            raise ValueError("Global map must be binary")

        self.map_resolution_desired = 0.02

        self.obs_resolution = observation_resolution
        self.obs_height_px = 64
        self.obs_width_px = 64

        self.obs_height_px_on_map = int(round(
            64 * observation_resolution / 0.02
        ))
        self.obs_width_px_on_map = self.obs_height_px_on_map

        self.veh_px_horz_from_center_on_obs = 0.0
        self.veh_px_vert_from_bottom_on_obs = 32.0
        self.veh_px_horz_from_center_on_map = 0.0
        self.veh_px_vert_from_bottom_on_map = (
            32.0 * observation_resolution / 0.02
        )

        # Enough occupied padding for rotated observations.
        half_diagonal = (
            64 * observation_resolution
            * math.sqrt(2) / (2 * 0.02)
        )
        padding = max(64, math.ceil(half_diagonal) + 8)

        self.localization_padding = int(padding)

        self.map = raw.copy()
        self.occ_map = raw.copy()

        self.map_with_border = cv2.copyMakeBorder(
            raw,
            padding, padding, padding, padding,
            cv2.BORDER_CONSTANT,
            value=0
        )

        self.inv_map_with_border = np.logical_not(
            self.map_with_border
        ).astype(int)

        h, w = raw.shape
        self.map_x_min_meters = -w * 0.02 / 2
        self.map_x_max_meters = w * 0.02 / 2
        self.map_y_min_meters = -h * 0.02 / 2
        self.map_y_max_meters = h * 0.02 / 2

        self.initialized = True

    def transform_map_px_to_m(self, row, col):
        h, w = self.map_with_border.shape

        x = (float(col) + 0.5 - w / 2.0) * 0.02
        y = (h / 2.0 - float(row) - 0.5) * 0.02

        return x, y

    def transform_map_m_to_px(self, x, y):
        h, w = self.map_with_border.shape

        col = math.floor(float(x) / 0.02 + w / 2.0)
        row = h - 1 - math.floor(float(y) / 0.02 + h / 2.0)

        row = max(0, min(h - 1, row))
        col = max(0, min(w - 1, col))

        return int(row), int(col)

    def extract_observation_region(
        self, veh_pose, pose_in_meters=True
    ):
        if pose_in_meters:
            pose = veh_pose
        else:
            pose = self.transform_pose_px_to_m(veh_pose)

        grid = sample_local_grid(
            self.map_with_border,
            pose.x, pose.y, pose.yaw,
            0.02, self.obs_resolution, 64
        )

        return grid, None
