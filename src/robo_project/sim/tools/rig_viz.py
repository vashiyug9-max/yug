#!/usr/bin/env python3
"""
Show the measured sensor rig (config/vinebot_rig.yaml) in RViz, attached to the moving robot.

  python3 src/robo_project/sim/tools/rig_viz.py

Publishes
  /tf_static        base_footprint -> camera_front / camera_left / camera_right / camera_back / lidar_rig
                    (the frame_ids of the Habitat camera images), so RViz can place the cameras
  /rig_markers      robot footprint (1.15 x 0.60 m), a pole + field-of-view wedge for every camera,
                    and the LiDAR at their mounting heights
Add a MarkerArray display on /rig_markers in RViz (display_vinebot.rviz already has one).
Display only: it does not change the simulation, the model or the driving.
"""

import math

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from robo_project.common.rig import CAM_ORDER, load_rig

FRAME = "base_footprint"
COLORS = {"front": (0.1, 0.8, 0.2), "left": (0.2, 0.5, 1.0), "right": (1.0, 0.6, 0.1),
          "back": (0.8, 0.2, 0.8), "lidar": (1.0, 0.1, 0.1)}
WEDGE_M = 0.6          # length of the drawn field-of-view wedge


def quat_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class RigViz(Node):
    def __init__(self):
        super().__init__("rig_viz")
        self.rig = load_rig("")
        self.tf = StaticTransformBroadcaster(self)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(MarkerArray, "/rig_markers", qos)
        self.publish_tf()
        self.markers = self.build_markers()
        self.create_timer(1.0, lambda: self.pub.publish(self.markers))
        cams = ", ".join(f"{n} {self.rig['cameras'][n]['height']:.2f} m" for n in CAM_ORDER)
        self.get_logger().info(f"rig {self.rig['path']}: {cams}, lidar {self.rig['lidar']['height']:.2f} m")

    def publish_tf(self):
        tfs = []
        sensors = [(f"camera_{n}", c["x"] if "x" in c else 0.0, c.get("y", 0.0), c["height"], c["yaw"])
                   for n, c in self.rig["cameras"].items()]
        lid = self.rig["lidar"]
        sensors.append(("lidar_rig", lid.get("x", 0.0), lid.get("y", 0.0), lid["height"], 0.0))
        for name, x, y, z, yaw in sensors:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = FRAME
            t.child_frame_id = name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(x), float(y), float(z)
            (t.transform.rotation.x, t.transform.rotation.y,
             t.transform.rotation.z, t.transform.rotation.w) = quat_yaw(float(yaw))
            tfs.append(t)
        self.tf.sendTransform(tfs)

    def marker(self, mid, mtype, rgb, alpha=1.0):
        m = Marker()
        m.header.frame_id = FRAME
        m.ns, m.id, m.type, m.action = "rig", mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b = rgb
        m.color.a = alpha
        m.frame_locked = True          # follow the robot as it drives
        return m

    def build_markers(self):
        out = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL       # drop markers from an earlier run (e.g. old labels)
        out.markers.append(clear)
        L, W = float(self.rig["robot"]["length"]), float(self.rig["robot"]["width"])
        # footprint outline on the ground
        fp = self.marker(0, Marker.LINE_STRIP, (1.0, 1.0, 0.0))
        fp.scale.x = 0.02
        fp.points = [Point(x=sx * L / 2, y=sy * W / 2, z=0.02) for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
        out.markers.append(fp)
        mid = 1
        sensors = [(n, c, COLORS[n]) for n, c in self.rig["cameras"].items()]
        sensors.append(("lidar", self.rig["lidar"], COLORS["lidar"]))
        for name, c, rgb in sensors:
            x, y, h = float(c.get("x", 0.0)), float(c.get("y", 0.0)), float(c["height"])
            # pole from the ground up to the sensor = its height
            pole = self.marker(mid, Marker.CYLINDER, rgb, 0.6); mid += 1
            pole.pose.position.x, pole.pose.position.y, pole.pose.position.z = x, y, h / 2
            pole.scale.x = pole.scale.y = 0.025
            pole.scale.z = h
            out.markers.append(pole)
            # the sensor itself
            head = self.marker(mid, Marker.CUBE if name != "lidar" else Marker.CYLINDER, rgb); mid += 1
            head.pose.position.x, head.pose.position.y, head.pose.position.z = x, y, h
            if name == "lidar":
                head.scale.x, head.scale.y, head.scale.z = 0.10, 0.10, 0.06
            else:
                (head.pose.orientation.x, head.pose.orientation.y,
                 head.pose.orientation.z, head.pose.orientation.w) = quat_yaw(float(c["yaw"]))
                head.scale.x, head.scale.y, head.scale.z = 0.04, 0.10, 0.04
            out.markers.append(head)
            # horizontal field of view, drawn at the camera height
            if name != "lidar":
                yaw, half = float(c["yaw"]), math.radians(float(c["hfov_deg"])) / 2
                wedge = self.marker(mid, Marker.TRIANGLE_LIST, rgb, 0.25); mid += 1
                wedge.scale.x = wedge.scale.y = wedge.scale.z = 1.0
                steps = 12
                for i in range(steps):
                    a0 = yaw - half + 2 * half * i / steps
                    a1 = yaw - half + 2 * half * (i + 1) / steps
                    wedge.points += [Point(x=x, y=y, z=h),
                                     Point(x=x + WEDGE_M * math.cos(a0), y=y + WEDGE_M * math.sin(a0), z=h),
                                     Point(x=x + WEDGE_M * math.cos(a1), y=y + WEDGE_M * math.sin(a1), z=h)]
                out.markers.append(wedge)
        return out


def main():
    rclpy.init()
    node = RigViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
