#!/usr/bin/env python3

import math
import time
import numpy as np
import rclpy

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

from robo_project.localization.localization_map_step342 import (
    LocalizationMapFrameManager
)

from robo_project.localization.particle_filter_step308_candidate import (
    ParticleFilter
)


class LiveLocalization(Node):

    def __init__(self):
        super().__init__("step361_live_localization")

        self.latest_grid = None
        self.last_stamp = None
        self.mfm = None
        self.pf = None
        self.resolution = None

        self.pub = self.create_publisher(
            PoseStamped,
            "/localization/provisional_pose",
            10
        )

        self.sub = self.create_subscription(
            OccupancyGrid,
            "/fusion/local_grid",
            self.on_grid,
            10
        )

        self.timer = self.create_timer(
            10.0,
            self.localize
        )

        self.get_logger().info(
            "STEP 361 started. Waiting for ML occupancy."
        )

    def on_grid(self, msg):
        self.latest_grid = msg

    def localize(self):

        msg = self.latest_grid

        if msg is None:
            self.get_logger().warning("Waiting for ML grid...")
            return

        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

        if stamp == self.last_stamp:
            self.get_logger().warning("No new ML observation.")
            return

        self.last_stamp = stamp

        try:
            if (
                msg.info.width != 64
                or msg.info.height != 64
                or msg.header.frame_id != "base_footprint"
            ):
                self.get_logger().error("Invalid ML grid metadata.")
                return

            res = float(msg.info.resolution)

            raw = np.asarray(
                msg.data, dtype=np.int16
            )

            if raw.size != 4096 or res <= 0:
                self.get_logger().error("Invalid ML data.")
                return

            raw = raw.reshape(64, 64)

            # Restore Fusion ML robot-relative orientation.
            occ = raw[::-1, ::-1].T

            obs = np.full(
                (64, 64), np.nan, dtype=np.float32
            )

            obs[(occ >= 0) & (occ <= 25)] = 1.0
            obs[(occ >= 75) & (occ <= 100)] = 0.0

            free = int(np.count_nonzero(obs == 1.0))
            occupied = int(np.count_nonzero(obs == 0.0))

            if min(free, occupied) < 10:
                self.get_logger().warning(
                    "Insufficient occupied/free evidence."
                )
                return

            if self.mfm is None:
                self.get_logger().info(
                    "Loading hand-drawn map..."
                )

                self.mfm = LocalizationMapFrameManager(res)

                self.pf = ParticleFilter.__new__(ParticleFilter)
                self.pf.mfm = self.mfm

                self.resolution = res

                self.get_logger().info("Map ready.")

            if abs(res - self.resolution) > 1e-6:
                self.get_logger().error("Grid resolution changed.")
                return

            candidates = self.pf.search_global_candidates(
                obs,
                top_per_yaw=4
            )

            if not candidates:
                self.get_logger().warning(
                    "No localization candidates."
                )
                return

            best = candidates[0]

            x = float(best["x"])
            y = float(best["y"])
            yaw = float(best["yaw"])
            score = float(best["similarity"])

            pose = PoseStamped()

            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = "cmn_sketch_map"

            pose.pose.position.x = x
            pose.pose.position.y = y

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)

            self.pub.publish(pose)

            self.get_logger().info(
                f"POSE PUBLISHED: "
                f"x={x:.3f}, "
                f"y={y:.3f}, "
                f"yaw={math.degrees(yaw):.1f} deg, "
                f"similarity={score:.4f}"
            )

        except Exception as e:
            self.get_logger().error(
                f"{type(e).__name__}: {e}"
            )


def main():
    rclpy.init()
    node = LiveLocalization()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
