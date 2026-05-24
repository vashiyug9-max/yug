#!/usr/bin/env python3

"""
Set of static functions to perform pure pursuit navigation.
Migrated from ROS 1 to ROS 2.
"""

# ROS 2: use rclpy logger instead of rospy
import rclpy.logging
from math import remainder, tau, pi, atan2, sqrt
from time import time

from robo_project.scripts.basic_types import PoseMeters

# Module-level logger
_logger = rclpy.logging.get_logger('pure_pursuit')


class PurePursuit:
    verbose = False
    # Pure pursuit params.
    use_finite_lookahead_dist = True  # If false, just use the goal point as the lookahead.
    lookahead_dist_init = 0.2         # meters.
    lookahead_dist_max = 2            # meters.
    k_p = 1.0    # Proportional gain.
    k_i = 0.0    # Integral gain.
    k_d = 0.0    # Derivative gain.
    k_fwd_lin = 1    # Multiplicative gain on forward velocity.
    k_fwd_power = 5  # Exponential term in forward velocity calculation.
    k_fwd_add = 0.0  # Additive term in forward velocity calculation.
    # Path to follow.
    path_meters = []
    # PID vars.
    integ = 0        # Accumulating integral term.
    err_prev = 0.0   # Error from last iteration (for derivative term).
    last_time = 0.0  # Time from last iteration (for computing dt).

    def compute_command(self, cur_pose_m: PoseMeters, path):
        """
        Determine velocity command to stay on the path.
        @param cur_pose_m - PoseMeters of current vehicle pose (x, y, yaw).
        @param path - List of PoseMeters making up the path to follow.
        @return tuple (fwd, ang) velocities in m/s and rad/s.
        """
        self.path_meters = path
        self.pare_path(cur_pose_m)

        if len(self.path_meters) < 1:
            # ROS 2: use rclpy logger
            _logger.warn("PP: Pure pursuit called with no path. Commanding zeros.")
            return 0.0, 0.0

        if self.use_finite_lookahead_dist:
            lookahead_pt = None
            lookahead_dist = self.lookahead_dist_init
            while lookahead_pt is None and lookahead_dist <= self.lookahead_dist_max:
                lookahead_pt = self.choose_lookahead_pt(cur_pose_m, lookahead_dist)
                lookahead_dist *= 1.25
            if lookahead_pt is None:
                # Can't see the path — go to the first point.
                lookahead_pt = self.path_meters[0]
        else:
            lookahead_pt = self.path_meters[-1]

        if self.verbose:
            _logger.info("PP: Choosing lookahead point ({:.2f}, {:.2f}).".format(
                lookahead_pt.x, lookahead_pt.y))

        # Compute global heading to lookahead point.
        gb = atan2(lookahead_pt.y - cur_pose_m.y, lookahead_pt.x - cur_pose_m.x)
        # Compute heading relative to vehicle pose.
        beta = remainder(gb - cur_pose_m.yaw, tau)

        if self.verbose:
            _logger.info("PP: Angle difference is {:.2f}, or {:.2f} relative to current vehicle pose.".format(
                gb, beta))

        # Compute time since last iteration.
        dt = 0
        if self.last_time != 0:
            dt = time() - self.last_time
        self.last_time = time()

        # Update PID terms.
        P = self.k_p * beta
        self.integ += beta * dt
        I = self.k_i * self.integ
        D = 0.0
        if dt != 0:
            D = self.k_d * (beta - self.err_prev) / dt
        self.err_prev = beta

        ang = P + I + D
        fwd = self.k_fwd_lin * (1 - abs(beta / pi)) ** self.k_fwd_power + self.k_fwd_add

        return fwd, ang

    def pare_path(self, cur_pose_m: PoseMeters):
        """
        If the vehicle is near a path point, cut the path up to that point.
        @param cur_pose_m - PoseMeters of vehicle pose (x, y, yaw).
        """
        for i in range(len(self.path_meters)):
            dist = ((cur_pose_m.x - self.path_meters[i].x) ** 2 +
                    (cur_pose_m.y - self.path_meters[i].y) ** 2) ** (1 / 2)
            if dist < 0.15:
                del self.path_meters[0:i + 1]
                return

    def choose_lookahead_pt(self, cur_pose_m: PoseMeters, lookahead_dist: float) -> PoseMeters:
        """
        Find the point on the path at the specified radius from the current vehicle position.
        @param cur_pose_m - PoseMeters of vehicle pose (x, y, yaw).
        @param lookahead_dist - Search radius in meters.
        @return PoseMeters of the chosen lookahead point, or None if not found.
        """
        if len(self.path_meters) == 1:
            return self.path_meters[0]

        lookahead_pt = None
        for i in range(1, len(self.path_meters)):
            diff = [self.path_meters[i].x - self.path_meters[i - 1].x,
                    self.path_meters[i].y - self.path_meters[i - 1].y]
            v1 = [self.path_meters[i - 1].x - cur_pose_m.x,
                  self.path_meters[i - 1].y - cur_pose_m.y]
            a = diff[0] ** 2 + diff[1] ** 2
            b = 2 * (v1[0] * diff[0] + v1[1] * diff[1])
            c = v1[0] ** 2 + v1[1] ** 2 - lookahead_dist ** 2
            try:
                discr = sqrt(b ** 2 - 4 * a * c)
            except ValueError:
                # Negative discriminant — no real roots (segment too far away).
                continue
            q = [(-b - discr) / (2 * a), (-b + discr) / (2 * a)]
            valid = [q[j] >= 0 and q[j] <= 1 for j in range(2)]
            if valid[0]:
                lookahead_pt = PoseMeters(
                    self.path_meters[i - 1].x + q[0] * diff[0],
                    self.path_meters[i - 1].y + q[0] * diff[1]
                )
            elif valid[1]:
                lookahead_pt = PoseMeters(
                    self.path_meters[i - 1].x + q[1] * diff[0],
                    self.path_meters[i - 1].y + q[1] * diff[1]
                )
        return lookahead_pt