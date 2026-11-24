#!/usr/bin/env python3
"""
Confirm gate: the robot drives autonomously only after a human says yes - for EVERY waypoint.

  ros2 run vinebot_bringup motion_confirm          (run it in its OWN terminal: it asks there)

  in:  /cmd_vel                  speeds from our path follower (ros/runner_node.py)
       /runner_current_waypoint  the waypoint the path follower is driving to right now
       /planned_path             the planner's path (waypoint number and total)
       /localization/odom        the robot's pose (parameter pose_topic; /odom in the simulation's normal mode)
       /goal_pose                a new goal from RViz -> all approvals are cleared
       /motion_confirm           std_msgs/Bool - answer from another terminal / tool instead of typing
                                 (true = go to this waypoint, false = stop):
                                 ros2 topic pub --once /motion_confirm std_msgs/msg/Bool "{data: true}"
  out: /cmd_vel_confirmed        speeds that may reach the motors (twist_mux "navigation" input on the real
                                 robot; in the simulation remap the bridge: -r /cmd_vel:=/cmd_vel_confirmed)

Behaviour
  - Each time the path follower moves on to a new waypoint, the robot is held (zero speeds) and the terminal
    shows the waypoint: its number on the path, its position, the straight-line distance from the robot and
    its direction (ahead / N deg left / N deg right / behind), and asks "Move? [y/N]".
      y   drive to THIS waypoint; the next waypoint is asked again when the robot gets there
      a   drive to all remaining waypoints of this goal without asking again
      s   stop at any time (all approvals cleared)
      anything else: stay where you are
  - Re-plans (ML obstacle updates) that move a waypoint by less than same_wp_m (0.30 m) keep its approval;
    a waypoint further away is asked again. A new goal clears all approvals.
The joystick (L1) and the PS4 O button work as before: twist_mux gives them priority over this input.
"""
import math
import sys
import threading

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Bool


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def direction_text(rel_deg: float) -> str:
    a = abs(rel_deg)
    if a < 10:
        return "straight ahead"
    if a > 170:
        return "behind the robot (turn around)"
    return f"{a:.0f} deg to the {'left' if rel_deg > 0 else 'right'}"


class MotionConfirm(Node):
    def __init__(self):
        super().__init__("motion_confirm")
        p = self.declare_parameter
        out_topic = str(p("out_topic", "/cmd_vel_confirmed").value)
        pose_topic = str(p("pose_topic", "/localization/odom").value)
        self.same_wp = float(p("same_wp_m", 0.30).value)
        self.lock = threading.Lock()
        self.goal = None                 # (x, y) of the RViz goal
        self.path = []                   # [(x, y)] waypoints of the current path
        self.pose = None                 # (x, y, yaw)
        self.wp = None                   # waypoint the runner drives to now
        self.approved = []               # approved waypoints of this goal
        self.approve_all = False
        self.asked = None                # waypoint the open question is about
        self.pub = self.create_publisher(Twist, out_topic, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(PoseStamped, "/runner_current_waypoint", self.on_waypoint, 10)
        self.create_subscription(Path, "/planned_path", self.on_path, 10)
        self.create_subscription(Odometry, pose_topic, self.on_pose, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(Bool, "/motion_confirm", self.on_answer_msg, 10)
        self.tty = sys.stdin.isatty()
        self.get_logger().info(
            f"Confirm gate: /cmd_vel -> {out_topic} only after 'y' for EACH waypoint (pose from {pose_topic}). "
            + ("Answer here: y = this waypoint, a = all waypoints of this goal, s = stop." if self.tty else
               "No terminal input: answer with ros2 topic pub --once /motion_confirm std_msgs/msg/Bool "
               "\"{data: true}\""))
        if self.tty:
            threading.Thread(target=self.read_terminal, daemon=True).start()

    # ── helpers ─────────────────────────────────────────────────────────────
    def is_approved(self, wp) -> bool:
        return self.approve_all or any(math.hypot(wp[0] - a[0], wp[1] - a[1]) < self.same_wp for a in self.approved)

    def describe(self, wp) -> str:
        num = ""
        if self.path:
            k = min(range(len(self.path)), key=lambda i: math.hypot(self.path[i][0] - wp[0], self.path[i][1] - wp[1]))
            last = " = parking spot of the goal" if k == len(self.path) - 1 else ""
            num = f"waypoint {k}/{len(self.path) - 1}{last} "
        text = f"{num}at ({wp[0]:.2f}, {wp[1]:.2f})"
        if self.pose is not None:
            x, y, yaw = self.pose
            dist = math.hypot(wp[0] - x, wp[1] - y)
            rel = math.degrees(math.atan2(math.sin(math.atan2(wp[1] - y, wp[0] - x) - yaw),
                                          math.cos(math.atan2(wp[1] - y, wp[0] - x) - yaw)))
            text += f": {dist:.2f} m away, {direction_text(rel)}"
        return text

    # ── inputs ─────────────────────────────────────────────────────────────
    def on_goal(self, msg: PoseStamped):
        with self.lock:
            self.goal = (msg.pose.position.x, msg.pose.position.y)
            self.approved, self.approve_all, self.asked = [], False, None
        self.get_logger().warn(f"New goal ({self.goal[0]:.2f}, {self.goal[1]:.2f}): every waypoint will be asked.")

    def on_path(self, msg: Path):
        with self.lock:
            self.path = [(q.pose.position.x, q.pose.position.y) for q in msg.poses]

    def on_pose(self, msg: Odometry):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def on_waypoint(self, msg: PoseStamped):
        wp = (msg.pose.position.x, msg.pose.position.y)
        with self.lock:
            self.wp = wp
            ask = self.pose is not None and not self.is_approved(wp) and (   # ask once the pose is known

                self.asked is None or math.hypot(wp[0] - self.asked[0], wp[1] - self.asked[1]) >= self.same_wp)
            if ask:
                self.asked = wp
                text = self.describe(wp)
        if ask:
            self.get_logger().warn(f"NEXT WAYPOINT - robot held: {text}")
            if self.tty:
                print("Move to this waypoint? [y/N]  (a = all waypoints of this goal, s = stop): ", end="", flush=True)

    def on_cmd(self, msg: Twist):
        with self.lock:
            ok = self.wp is not None and self.is_approved(self.wp)
        self.pub.publish(msg if ok else Twist())      # not approved: active stop

    def on_answer_msg(self, msg: Bool):
        self.answer("y" if msg.data else "s")

    def read_terminal(self):
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            self.answer(line.strip().lower())

    # ── decisions ──────────────────────────────────────────────────────────
    def answer(self, a: str):
        with self.lock:
            wp = self.asked or self.wp
            if a in ("y", "yes"):
                if wp is None:
                    text = "Nothing to approve yet (no waypoint)."
                else:
                    self.approved.append(wp)
                    text = f"APPROVED - driving to {self.describe(wp)}"
            elif a in ("a", "all"):
                self.approve_all = True
                text = "APPROVED - driving through all remaining waypoints of this goal without asking"
            elif a in ("s", "stop"):
                self.approved, self.approve_all, self.asked = [], False, None
                text = "STOPPED - the current waypoint has to be approved again (y)"
            else:
                text = "Not approved - robot stays where it is (type y to approve this waypoint)"
        self.get_logger().warn(text)


def main():
    rclpy.init()
    node = MotionConfirm()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
