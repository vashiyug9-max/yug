#!/usr/bin/env python3
"""
Confirm gate: the robot drives autonomously only after a human says yes - once per new goal.

  ros2 run vinebot_bringup motion_confirm          (run it in its OWN terminal: it asks there)

  in:  /cmd_vel        speeds from our path follower (ros/runner_node.py, unchanged)
       /planned_path   the planner's path (for the summary shown before asking)
       /goal_pose      a new goal from RViz -> a new question
       /motion_confirm std_msgs/Bool - answer from another terminal / tool instead of typing
                       (true = go, false = stop):
                       ros2 topic pub --once /motion_confirm std_msgs/msg/Bool "{data: true}"
  out: /cmd_vel_confirmed  speeds that may reach the motors (twist_mux "navigation" input on the
                       real robot; in the simulation remap the bridge: -r /cmd_vel:=/cmd_vel_confirmed)

Behaviour
  - A new goal (and the start-up goal) is NOT approved: the robot gets zero speeds while the runner
    wants to move, and the terminal shows goal, path length and waypoints and asks "Move? [y/N]".
  - y -> speeds pass through until the goal changes. Re-plans to the same goal (ML obstacle updates)
    do not ask again.
  - anything else -> stays stopped; type y later to approve the current goal.
  - type s (or publish false on /motion_confirm) at any time -> stop, the goal must be approved again.
The joystick (L1) and the PS4 O button work as before: twist_mux gives them priority over this input.
"""
import math
import sys
import threading

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool


class MotionConfirm(Node):
    def __init__(self):
        super().__init__("motion_confirm")
        out_topic = str(self.declare_parameter("out_topic", "/cmd_vel_confirmed").value)
        self.goal = "start-up goal of the planner"     # what the current question is about
        self.goal_id = 0
        self.approved_id = None
        self.asked_id = None
        self.path = None
        self.lock = threading.Lock()
        self.pub = self.create_publisher(Twist, out_topic, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Path, "/planned_path", self.on_path, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal, 10)
        self.create_subscription(Bool, "/motion_confirm", self.on_answer_msg, 10)
        self.tty = sys.stdin.isatty()
        self.get_logger().info(
            f"Confirm gate: /cmd_vel -> {out_topic} only after 'y'. "
            + ("Answer in this terminal (y = go, s = stop)." if self.tty else
               "No terminal input: answer with ros2 topic pub --once /motion_confirm std_msgs/msg/Bool "
               "\"{data: true}\""))
        if self.tty:
            threading.Thread(target=self.read_terminal, daemon=True).start()

    # ── inputs ─────────────────────────────────────────────────────────────
    def on_goal(self, msg: PoseStamped):
        with self.lock:
            self.goal_id += 1
            self.goal = f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
            self.path = None
        self.get_logger().warn(f"New goal {self.goal}: robot held until you confirm.")

    def on_path(self, msg: Path):
        with self.lock:
            self.path = msg
            ask = self.approved_id != self.goal_id and self.asked_id != self.goal_id
            if ask:
                self.asked_id = self.goal_id
        if ask:
            self.ask()

    def on_cmd(self, msg: Twist):
        with self.lock:
            ok = self.approved_id == self.goal_id
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
    def summary(self):
        p = self.path
        if p is None or not p.poses:
            return f"goal {self.goal} (no path yet)"
        pts = [(q.pose.position.x, q.pose.position.y) for q in p.poses]
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        end = pts[-1]
        return (f"goal {self.goal}, robot will park at ({end[0]:.2f}, {end[1]:.2f}), "
                f"path {length:.1f} m, {len(pts)} waypoints")

    def ask(self):
        with self.lock:
            text = self.summary()
        self.get_logger().warn(f"MOVE REQUEST: {text}")
        if self.tty:
            print("Move? [y/N] (s = stop at any time): ", end="", flush=True)

    def answer(self, a: str):
        with self.lock:
            if a in ("y", "yes"):
                self.approved_id = self.goal_id
                text = f"APPROVED - driving to {self.goal}"
            elif a in ("s", "stop"):
                self.approved_id = None
                self.asked_id = None                    # ask again on the next path
                text = "STOPPED - the goal has to be approved again (y)"
            else:
                text = "Not approved - robot stays where it is (type y to approve this goal)"
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
