import os
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster
import habitat_sim

SCENE = "/home/yug/habitat_data/scene_datasets/gibson/gibson/Denmark.glb"

CAM_H = 240
CAM_W = 320
STEP_FORWARD = 0.10
STEP_TURN_DEG = 5.0

BASE_HEIGHT = 0.35
CAMERA_REL_HEIGHT = 0.25

CAMERA_CONFIG = {
    "front": {"pos": [0.20, CAMERA_REL_HEIGHT, 0.00], "yaw": 0.0, "frame_id": "front_camera_link"},
    "left":  {"pos": [0.00, CAMERA_REL_HEIGHT, 0.20], "yaw": 90.0, "frame_id": "left_camera_link"},
    "right": {"pos": [0.00, CAMERA_REL_HEIGHT, -0.20], "yaw": -90.0, "frame_id": "right_camera_link"},
    "rear":  {"pos": [-0.20, CAMERA_REL_HEIGHT, 0.00], "yaw": 180.0, "frame_id": "rear_camera_link"},
}

def habitat_rgb_to_bgr(img):
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def quat_xyzw_from_yaw(yaw):
    qx = 0.0
    qy = math.sin(yaw / 2.0)
    qz = 0.0
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw

class HabitatBridgeVinebot(Node):
    def __init__(self):
        super().__init__("habitat_bridge_vinebot_2")

        self.sim = self.create_sim()
        self.agent = self.sim.initialize_agent(0)
        self.tf_broadcaster = TransformBroadcaster(self)

        state = habitat_sim.AgentState()
        state.position = np.array([0.0, BASE_HEIGHT, 4.0], dtype=np.float32)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            np.deg2rad(180.0),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.agent.set_state(state)

        self.image_pubs = {
            "front": self.create_publisher(Image, "/camera/front/image_raw", 10),
            "left": self.create_publisher(Image, "/camera/left/image_raw", 10),
            "right": self.create_publisher(Image, "/camera/right/image_raw", 10),
            "rear": self.create_publisher(Image, "/camera/rear/image_raw", 10),
        }

        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10
        )

        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0

        self.timer = self.create_timer(0.1, self.timer_callback)

        cv2.namedWindow("Vinebot Habitat Bridge", cv2.WINDOW_NORMAL)
        self.get_logger().info("Habitat Bridge Vinebot 2 started")
        self.get_logger().info("WASD: manual control | /cmd_vel: ROS control | Q: quit")

    def make_sensor(self, uuid_name, pos_xyz, yaw_deg, h, w, hfov=60.0):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = uuid_name
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [h, w]
        sensor.position = pos_xyz
        sensor.orientation = [0.0, math.radians(yaw_deg), 0.0]
        sensor.hfov = hfov
        return sensor

    def create_sim(self):
        if not os.path.exists(SCENE):
            raise FileNotFoundError(f"Scene not found: {SCENE}")

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = SCENE
        backend_cfg.enable_physics = False
        backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        sensors = [
            self.make_sensor("front", CAMERA_CONFIG["front"]["pos"], CAMERA_CONFIG["front"]["yaw"], CAM_H, CAM_W),
            self.make_sensor("left", CAMERA_CONFIG["left"]["pos"], CAMERA_CONFIG["left"]["yaw"], CAM_H, CAM_W),
            self.make_sensor("right", CAMERA_CONFIG["right"]["pos"], CAMERA_CONFIG["right"]["yaw"], CAM_H, CAM_W),
            self.make_sensor("rear", CAMERA_CONFIG["rear"]["pos"], CAMERA_CONFIG["rear"]["yaw"], CAM_H, CAM_W),
        ]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        agent_cfg.height = BASE_HEIGHT
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=STEP_FORWARD),
            ),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward",
                habitat_sim.agent.ActuationSpec(amount=STEP_FORWARD),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left",
                habitat_sim.agent.ActuationSpec(amount=STEP_TURN_DEG),
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right",
                habitat_sim.agent.ActuationSpec(amount=STEP_TURN_DEG),
            ),
        }

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    def cmd_vel_callback(self, msg):
        self.last_cmd_linear = msg.linear.x
        self.last_cmd_angular = msg.angular.z

    def apply_ros_cmd(self):
        if self.last_cmd_linear > 0.01:
            self.agent.act("move_forward")
        elif self.last_cmd_linear < -0.01:
            self.agent.act("move_backward")

        if self.last_cmd_angular > 0.01:
            self.agent.act("turn_left")
        elif self.last_cmd_angular < -0.01:
            self.agent.act("turn_right")

    def add_label(self, img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (180, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    def np_to_image_msg(self, frame, frame_id):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = frame.shape[1] * frame.shape[2]
        msg.data = frame.tobytes()
        return msg

    def publish_image(self, name, frame):
        msg = self.np_to_image_msg(frame, CAMERA_CONFIG[name]["frame_id"])
        self.image_pubs[name].publish(msg)

    def publish_tf(self):
        state = self.agent.get_state()
        pos = state.position

        q = state.rotation
        try:
            qx = float(q.x)
            qy = float(q.y)
            qz = float(q.z)
            qw = float(q.w)
        except AttributeError:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(pos[0])
        t.transform.translation.y = float(pos[1])
        t.transform.translation.z = float(pos[2])

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        self.apply_ros_cmd()
        self.publish_tf()

        obs = self.sim.get_sensor_observations()

        front = habitat_rgb_to_bgr(obs["front"])
        left = habitat_rgb_to_bgr(obs["left"])
        right = habitat_rgb_to_bgr(obs["right"])
        rear = habitat_rgb_to_bgr(obs["rear"])

        self.publish_image("front", front)
        self.publish_image("left", left)
        self.publish_image("right", right)
        self.publish_image("rear", rear)

        front_view = self.add_label(front, "Front")
        left_view = self.add_label(left, "Left")
        right_view = self.add_label(right, "Right")
        rear_view = self.add_label(rear, "Rear")

        top_row = np.hstack([front_view, left_view])
        bottom_row = np.hstack([right_view, rear_view])
        grid = np.vstack([top_row, bottom_row])

        cv2.imshow("Vinebot Habitat Bridge", grid)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("w"):
            self.agent.act("move_forward")
        elif key == ord("s"):
            self.agent.act("move_backward")
        elif key == ord("a"):
            self.agent.act("turn_left")
        elif key == ord("d"):
            self.agent.act("turn_right")
        elif key == ord("q"):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()

def main():
    rclpy.init()
    node = HabitatBridgeVinebot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
