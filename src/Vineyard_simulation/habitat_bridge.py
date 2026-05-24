import os
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import habitat_sim

SCENE = "/home/yug/habitat_data/scene_datasets/gibson/gibson/Denmark.glb"

CAM_H = 240
CAM_W = 320
STEP_FORWARD = 0.10
STEP_TURN_DEG = 5.0

CAMERA_CONFIG = {
    "front": {"pos": [0.20, 1.50, 0.00], "yaw": 0.0, "frame_id": "front_camera_link"},
    "left":  {"pos": [0.00, 1.50, 0.20], "yaw": 90.0, "frame_id": "left_camera_link"},
    "right": {"pos": [0.00, 1.50, -0.20], "yaw": -90.0, "frame_id": "right_camera_link"},
    "rear":  {"pos": [-0.20, 1.50, 0.00], "yaw": 180.0, "frame_id": "rear_camera_link"},
}

def habitat_rgb_to_bgr(img):
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class HabitatRosBridge(Node):
    def __init__(self):
        super().__init__("habitat_ros_bridge")

        self.bridge = CvBridge()
        self.sim = self.create_sim()
        self.agent = self.sim.initialize_agent(0)

        state = habitat_sim.AgentState()
        state.position = np.array([0.0, 1.5, 4.0], dtype=np.float32)
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

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Habitat ROS bridge started")

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

    def add_label(self, img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (180, 30), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    def publish_image(self, name, frame):
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = CAMERA_CONFIG[name]["frame_id"]
        self.image_pubs[name].publish(msg)

    def timer_callback(self):
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

        cv2.imshow("Habitat ROS2 4-Camera Bridge", grid)
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
    node = HabitatRosBridge()
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
