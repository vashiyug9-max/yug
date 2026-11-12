#!/usr/bin/env python3

"""
Particle Filter class implementation.
Can separately process its prediction and update steps at different, independent rates,
and can be polled for the most likely particle estimate at any time.
Migrated from ROS 1 to ROS 2.
"""

import yaml
import numpy as np
import cv2
from math import sin, cos, remainder, tau
from random import choices

# ROS 2: use ament_index instead of rospkg
from ament_index_python.packages import get_package_share_directory

from robo_project.common.map_handler import MapFrameManager
from robo_project.common.basic_types import PoseMeters, PosePixels


class ParticleFilter:
    # Config params.
    num_particles = None
    state_size = None
    num_to_resample_randomly = None
    # Utility class.
    mfm = None
    # Ongoing state.
    particle_set = None
    particle_weights = None
    # Filter output.
    best_weight = 0
    best_estimate = None

    def __init__(self):
        """
        Instantiate the particle filter and set params from the config yaml.
        """
        # ROS 2: use ament_index to find the package share directory
        pkg_path = get_package_share_directory('robo_project')
        with open(pkg_path + '/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
            self.num_particles = int(config["particle_filter"]["num_particles"])
            self.all_indices = list(range(self.num_particles))
            self.state_size = int(config["particle_filter"]["state_size"])
            random_sampling_rate = config["particle_filter"]["random_sampling_rate"]
            self.num_to_resample_randomly = int(random_sampling_rate * self.num_particles)

        # Init arrays with correct dimensions.
        self.particle_set = np.zeros((self.num_particles, self.state_size))
        self.particle_weights = np.zeros(self.num_particles)
        self.best_estimate = np.zeros(self.state_size)

    def set_map_frame_manager(self, mfm: MapFrameManager):
        """
        Set reference to the map frame manager for coordinate transforms.
        @param mfm - MapFrameManager instance already initialized with a map.
        """
        self.mfm = mfm

    def propagate_particles(self, fwd: float, ang: float):
        """
        Apply a relative motion to all particles.
        @param fwd - Commanded forward motion in meters.
        @param ang - Commanded angular motion in radians (CCW).
        """
        for i in range(self.num_particles):
            self.particle_set[i, 0] += fwd * cos(self.particle_set[i, 2])
            self.particle_set[i, 1] += fwd * sin(self.particle_set[i, 2])
            # Keep yaw normalized to (-pi, pi).
            self.particle_set[i, 2] = remainder(self.particle_set[i, 2] + ang, tau)

        # Propagate the overall filter estimate as well.
        if self.best_estimate is not None:
            self.best_estimate[0] += fwd * cos(self.best_estimate[2])
            self.best_estimate[1] += fwd * sin(self.best_estimate[2])
            self.best_estimate[2] = remainder(self.best_estimate[2] + ang, tau)

    def update_with_observation(self, observation) -> PoseMeters:
        """Bayesian measurement update using masked MSE similarity."""

        if self.mfm is None:
            raise RuntimeError("MapFrameManager not set.")

        def current_pose():
            p = self.best_estimate
            return PoseMeters(float(p[0]), float(p[1]), float(p[2]))

        self._measurement_valid = False

        if observation is None:
            return current_pose()

        actual = np.asarray(observation, dtype=np.float32)

        if actual.ndim != 2:
            raise ValueError("Observation must be a 2D occupancy grid.")

        known = np.isfinite(actual) & (
            np.isclose(actual, 0.0) |
            np.isclose(actual, 1.0)
        )

        # All unknown: no useful measurement evidence.
        if not np.any(known):
            return current_pose()

        likelihoods = np.zeros(self.num_particles, dtype=np.float64)

        for i in range(self.num_particles):
            pose = PoseMeters(
                float(self.particle_set[i, 0]),
                float(self.particle_set[i, 1]),
                float(self.particle_set[i, 2])
            )

            if self.mfm.veh_pose_m_in_collision(pose):
                continue

            expected, _ = self.mfm.extract_observation_region(pose)

            likelihoods[i] = self.compute_measurement_likelihood(
                expected, actual
            )

        # Experimental yaw refinement disabled pending regression tests.

        prior = np.asarray(self.particle_weights, dtype=np.float64)

        if (
            prior.shape != (self.num_particles,) or
            not np.all(np.isfinite(prior)) or
            prior.sum() <= 0
        ):
            prior = np.full(
                self.num_particles, 1.0 / self.num_particles
            )
        else:
            prior = prior / prior.sum()

        posterior = prior * likelihoods
        total = float(posterior.sum())

        if not np.isfinite(total) or total <= 1e-12:
            # Reject an observation that supports no particles.
            return current_pose()

        self.particle_weights = posterior / total

        i_best = int(np.argmax(self.particle_weights))

        # Current best particle, not historical maximum.
        self.best_weight = float(self.particle_weights[i_best])
        self.best_estimate = self.particle_set[i_best].copy()

        self._measurement_valid = True

        return current_pose()

    def compute_measurement_likelihood(
        self, obs_expected, obs_actual
    ) -> float:
        """Class-balanced occupancy likelihood."""

        if obs_expected is None or obs_actual is None:
            return 0.0

        expected = np.asarray(obs_expected, dtype=np.float32)
        actual = np.asarray(obs_actual, dtype=np.float32)

        if expected.shape != actual.shape:
            raise ValueError(
                f"Shape mismatch: {expected.shape} vs {actual.shape}"
            )

        valid = np.isfinite(expected) & np.isfinite(actual)

        occupied = valid & np.isclose(actual, 0.0)
        free = valid & np.isclose(actual, 1.0)

        has_occupied = bool(np.any(occupied))
        has_free = bool(np.any(free))

        if not has_occupied and not has_free:
            return 0.0

        # Actual OCCUPIED=0: expected should be 0.
        occupied_mse = (
            float(np.mean(expected[occupied] ** 2))
            if has_occupied else 0.0
        )

        # Actual FREE=1: expected should be 1.
        free_mse = (
            float(np.mean((1.0 - expected[free]) ** 2))
            if has_free else 0.0
        )

        if has_occupied and has_free:
            error = 0.5 * occupied_mse + 0.5 * free_mse
        elif has_occupied:
            error = occupied_mse
        else:
            error = free_mse

        return float(np.exp(-4.0 * error))


    def resample(self):
        """ESS-triggered resampling with jitter and global exploration."""

        if not getattr(self, "_measurement_valid", False):
            return False

        n = self.num_particles
        weights = np.asarray(self.particle_weights, dtype=np.float64)

        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            return False

        weights = weights / weights.sum()
        ess = 1.0 / np.sum(weights ** 2)

        # Avoid unnecessary particle collapse.
        if ess >= 0.5 * n:
            return False

        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()

        rng = self._rng

        n_random = min(max(self.num_to_resample_randomly, 0), n)
        n_selected = n - n_random

        indices = rng.choice(
            n, size=n_selected, replace=True, p=weights
        )

        new_set = np.zeros_like(self.particle_set)
        new_set[:n_selected] = self.particle_set[indices].copy()

        # Small perturbations avoid exact duplicate particles.
        xy_std = max(
            0.01, 0.5 * float(self.mfm.map_resolution_desired)
        )
        yaw_std = np.deg2rad(3.0)

        for i in range(n_selected):
            original = new_set[i].copy()
            candidate = original.copy()

            candidate[:2] += rng.normal(0.0, xy_std, 2)
            candidate[2] += rng.normal(0.0, yaw_std)
            candidate[2] = (
                candidate[2] + np.pi
            ) % (2.0 * np.pi) - np.pi

            # Do not keep a jittered particle in an obstacle.
            proposed_pose = PoseMeters(
                float(candidate[0]),
                float(candidate[1]),
                float(candidate[2])
            )

            if not self.mfm.veh_pose_m_in_collision(proposed_pose):
                new_set[i] = candidate

        # Preserve a small amount of global exploration.
        for i in range(n_selected, n):
            new_set[i] = (
                self.mfm.generate_random_valid_veh_pose()
                .as_np_array()
            )

        self.particle_set = new_set
        self.particle_weights = np.full(n, 1.0 / n)
        self._measurement_valid = False

        return True

    def search_global_candidates(self, observation, top_per_yaw=8):
        """Find coarse global pose hypotheses from a local occupancy grid.

        Input semantics: -1 unknown, 0 occupied, 1 free.
        The four cardinal orientations are searched.
        Does not modify particle state.
        """
        if self.mfm is None:
            raise RuntimeError("MapFrameManager has not been set.")

        mfm = self.mfm
        actual = np.asarray(observation, dtype=np.float32)

        expected_shape = (
            int(mfm.obs_height_px), int(mfm.obs_width_px)
        )
        if actual.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape}, got {actual.shape}"
            )

        h = int(mfm.obs_height_px_on_map)
        w = int(mfm.obs_width_px_on_map)

        if h < 1 or w < 1:
            raise ValueError("Invalid global-map crop dimensions.")

        known = np.isfinite(actual) & (
            (actual == 0) | (actual == 1)
        )

        # Preserve sparse wall measurements during downsampling.
        # These are coverage fractions, not binary majority votes.
        occupied_coverage = cv2.resize(
            (actual == 0).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_AREA
        )

        free_coverage = cv2.resize(
            (actual == 1).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_AREA
        )

        # Experimental thresholds:
        # Keep thin occupied features covering at least 1% of
        # the coarse cell. Free space needs 50% coverage.
        occupied_coarse = occupied_coverage >= 0.01
        free_coarse = free_coverage >= 0.50

        mask = occupied_coarse | free_coarse

        if mask.sum() < max(10, int(0.15 * h * w)):
            return []

        # Occupied evidence takes priority over free evidence.
        template_base = np.where(
            occupied_coarse, 0.0, 1.0
        ).astype(np.float32)

        # A completely free/occupied observation is ambiguous.
        if np.unique(template_base[mask]).size < 2:
            return []

        # Use the same padded map geometry as the PF
        # observation extraction. This allows candidates
        # close to the hand-drawn map boundaries.
        search_map = np.ascontiguousarray(
            mfm.map_with_border, dtype=np.float32
        )

        rotations = [
            (0.0, -1),
            (np.pi / 2, 0),
            (np.pi, 1),
            (-np.pi / 2, 2),
        ]

        candidates = []

        for yaw, k in rotations:
            templ = np.ascontiguousarray(
                np.rot90(template_base, k), dtype=np.float32
            )
            weight = np.ascontiguousarray(
                np.rot90(mask.astype(np.float32), k),
                dtype=np.float32
            )

            th, tw = templ.shape
            if th > search_map.shape[0] or tw > search_map.shape[1]:
                continue

            # Masked MSE for every possible map patch.
            # Unknown local cells contribute zero weight.
            map_sq = cv2.matchTemplate(
                search_map * search_map,
                weight,
                cv2.TM_CCORR
            )
            cross = cv2.matchTemplate(
                search_map,
                templ * weight,
                cv2.TM_CCORR
            )
            constant = float(np.sum(templ * templ * weight))

            mse = (
                map_sq - 2.0 * cross + constant
            ) / float(weight.sum())

            mse = np.maximum(mse, 0.0)

            half_row = (th - 1) // 2
            half_col = (tw - 1) // 2

            # Robot must occupy a free cell.
            center_free = (
                search_map[
                    half_row:half_row + mse.shape[0],
                    half_col:half_col + mse.shape[1]
                ] == 1
            )

            mse[~center_free] = np.inf

            suppression_radius = max(3, min(th, tw) // 2)

            for _ in range(max(0, int(top_per_yaw))):
                _, _, loc, _ = cv2.minMaxLoc(mse)
                col0, row0 = loc
                error = float(mse[row0, col0])

                if not np.isfinite(error):
                    break

                row = row0 + half_row
                col = col0 + half_col

                x, y = mfm.transform_map_px_to_m(row, col)

                candidates.append({
                    "x": float(x),
                    "y": float(y),
                    "yaw": float(yaw),
                    "similarity": float(
                        np.clip(1.0 - error, 0.0, 1.0)
                    )
                })

                r0 = max(0, row0 - suppression_radius)
                r1 = min(
                    mse.shape[0], row0 + suppression_radius + 1
                )
                c0 = max(0, col0 - suppression_radius)
                c1 = min(
                    mse.shape[1], col0 + suppression_radius + 1
                )

                mse[r0:r1, c0:c1] = np.inf

        # Refine each coarse global hypothesis using the
        # SAME full-resolution likelihood as the live PF.
        for candidate in candidates:
            candidate["coarse_similarity"] = float(
                candidate["similarity"]
            )

            pose = PoseMeters(
                candidate["x"],
                candidate["y"],
                candidate["yaw"]
            )

            if mfm.veh_pose_m_in_collision(pose):
                candidate["similarity"] = 0.0
                continue

            expected, _ = mfm.extract_observation_region(pose)

            candidate["similarity"] = float(
                self.compute_measurement_likelihood(
                    expected, actual
                )
            )

        candidates.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        return candidates

    def get_particle_set_px(self):
        """
        Convert the particle set to a list of PosePixels for visualization.
        @return List of PosePixels.
        """
        return [
            self.mfm.transform_pose_m_to_px(
                PoseMeters(self.particle_set[i, 0], self.particle_set[i, 1], self.particle_set[i, 2])
            )
            for i in range(self.num_particles)
        ]