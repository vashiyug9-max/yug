#!/usr/bin/env python3

import math
import csv
import bisect
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def timestamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def yaw(q):
    return math.atan2(
        2 * (q.w*q.z + q.x*q.y),
        1 - 2 * (q.y*q.y + q.z*q.z)
    )


def difference(a, b):
    return math.atan2(
        math.sin(a - b),
        math.cos(a - b)
    )


class MotionAccuracy(Node):

    def __init__(self):
        super().__init__("step364_motion_accuracy")

        self.history = deque(maxlen=20000)
        self.pending = []
        self.results = []
        self.baseline = None
        self.move_detected = False

        path = Path("localization_debug_manual") / (
            "step364_accuracy_" +
            datetime.now().strftime("%Y%m%d_%H%M%S") +
            ".csv"
        )

        self.file = path.open("x", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp",
            "gt_x", "gt_y", "gt_yaw_deg",
            "ml_x", "ml_y", "ml_yaw_deg",
            "position_error_m",
            "yaw_error_deg",
            "timestamp_difference_s",
            "gt_displacement_m",
            "ml_displacement_m",
            "relative_yaw_change_error_deg",
            "relative_xy_error_m"
        ])

        self.create_subscription(
            Odometry,
            "/odom",
            self.on_odom,
            50
        )

        self.create_subscription(
            PoseStamped,
            "/localization/provisional_pose",
            self.on_pose,
            10
        )

        print("========== STEP 364 ==========", flush=True)
        print("LIVE GROUND TRUTH vs ML LOCALIZATION", flush=True)
        print("CSV:", path, flush=True)
        print("Keep Habitat and Fusion ML running.", flush=True)
        print("First collect TWO stationary estimates.", flush=True)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.history.append((
            timestamp(msg),
            float(p.x),
            float(p.y),
            yaw(msg.pose.pose.orientation)
        ))
        self.compare()

    def on_pose(self, msg):
        self.pending.append(msg)
        self.compare()

    def compare(self):

        if len(self.history) < 2:
            return

        records = list(self.history)
        times = [r[0] for r in records]
        remaining = []

        for msg in self.pending:
            ts = timestamp(msg)

            if ts > times[-1]:
                remaining.append(msg)
                continue

            if ts < times[0]:
                print("SKIP: pose older than GT history.", flush=True)
                continue

            index = bisect.bisect_left(times, ts)

            if index == 0 or index == len(records):
                remaining.append(msg)
                continue

            a = records[index - 1]
            b = records[index]

            gap = min(abs(ts-a[0]), abs(b[0]-ts))

            if gap > 0.15:
                print(
                    f"SKIP: timestamp gap {gap:.3f}s",
                    flush=True
                )
                continue

            # Interpolate the reference pose to the ML timestamp.
            ratio = (ts-a[0]) / max(b[0]-a[0], 1e-9)

            gx = a[1] + ratio*(b[1]-a[1])
            gy = a[2] + ratio*(b[2]-a[2])
            gyaw = a[3] + ratio*difference(b[3], a[3])

            ex = float(msg.pose.position.x)
            ey = float(msg.pose.position.y)
            eyaw = yaw(msg.pose.orientation)

            pos_err = math.hypot(ex-gx, ey-gy)
            yaw_err = abs(math.degrees(
                difference(eyaw, gyaw)
            ))

            if self.baseline is None:
                self.baseline = (gx, gy, gyaw, ex, ey, eyaw)

            bx, by, bya, bex, bey, beya = self.baseline

            dgx, dgy = gx-bx, gy-by
            dex, dey = ex-bex, ey-bey

            gt_move = math.hypot(dgx, dgy)
            ml_move = math.hypot(dex, dey)

            relative_xy_error = math.hypot(
                dex-dgx, dey-dgy
            )

            relative_yaw_error = abs(math.degrees(
                difference(
                    difference(eyaw, beya),
                    difference(gyaw, bya)
                )
            ))

            self.results.append((pos_err, yaw_err))

            n = len(self.results)

            print("\n" + "="*52, flush=True)
            print(f"MATCHED POSE #{n}", flush=True)

            print(
                f"GT: x={gx:.3f}, y={gy:.3f}, "
                f"yaw={math.degrees(gyaw):.2f} deg",
                flush=True
            )

            print(
                f"ML: x={ex:.3f}, y={ey:.3f}, "
                f"yaw={math.degrees(eyaw):.2f} deg",
                flush=True
            )

            print(f"POSITION ERROR: {pos_err:.3f} m", flush=True)
            print(f"YAW ERROR: {yaw_err:.2f} deg", flush=True)

            print(
                f"GT displacement from start: {gt_move:.3f} m",
                flush=True
            )
            print(
                f"ML displacement from start: {ml_move:.3f} m",
                flush=True
            )
            print(
                f"Relative yaw-change error: "
                f"{relative_yaw_error:.2f} deg",
                flush=True
            )
            print(
                f"Timestamp gap: {gap:.3f} s",
                flush=True
            )

            if not self.move_detected and (
                gt_move > 0.20 or
                abs(math.degrees(difference(gyaw, bya))) > 12
            ):
                self.move_detected = True
                print("ROBOT MOVEMENT DETECTED", flush=True)

            if n == 2 and not self.move_detected:
                print(
                    "\nBASELINE READY — MOVE THE ROBOT NOW",
                    flush=True
                )

            if gt_move < 0.08 and ml_move > 0.8:
                print(
                    "WARNING: ML pose jumped while "
                    "GT remained near baseline.",
                    flush=True
                )

            self.writer.writerow([
                ts,
                gx, gy, math.degrees(gyaw),
                ex, ey, math.degrees(eyaw),
                pos_err, yaw_err, gap,
                gt_move, ml_move,
                relative_yaw_error,
                relative_xy_error
            ])
            self.file.flush()

        self.pending = remaining

    def finish(self):

        print("\n========== STEP 364 SUMMARY ==========", flush=True)

        if self.results:
            a = np.asarray(self.results)

            print("Matched samples:", len(a))
            print(f"Mean position discrepancy: {a[:,0].mean():.3f} m")
            print(f"Max position discrepancy: {a[:,0].max():.3f} m")
            print(f"Mean yaw discrepancy: {a[:,1].mean():.2f} deg")
            print(f"Max yaw discrepancy: {a[:,1].max():.2f} deg")
        else:
            print("No synchronized poses received.")

        print("Ground truth used ONLY for evaluation.")
        print("Direct errors assume aligned coordinate frames.")

        self.file.close()


def main():
    rclpy.init()
    node = MotionAccuracy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
