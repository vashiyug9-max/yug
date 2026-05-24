import os
import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import habitat_sim

SCENE = "/home/yug/habitat_data/scene_datasets/gibson/gibson/Denmark.glb"

CAM_H = 240
CAM_W = 320
STEP_FORWARD = 0.10
STEP_TURN_DEG = 5.0

BASE_HEIGHT = 0.35
CAMERA_REL_HEIGHT = 0.25

FX = 280.0
FY = 280.0
CX = CAM_W / 2.0
CY = CAM_H / 2.0

SAVE_DIR = os.path.expanduser("~/vinebot_ws/habitat_photos")

CAMERA_CONFIG = {
    "front": {"pos": [0.20, CAMERA_REL_HEIGHT, 0.00], "yaw": 0.0, "frame_id": "front_camera_link", "topic": "/camera/front/image_raw", "info_topic": "/camera/front/camera_info"},
    "left": {"pos": [0.00, CAMERA_REL_HEIGHT, 0.20], "yaw": 90.0, "frame_id": "left_camera_link", "topic": "/camera/left/image_raw", "info_topic": "/camera/left/camera_info"},
    "right": {"pos": [0.00, CAMERA_REL_HEIGHT, -0.20], "yaw": -90.0, "frame_id": "right_camera_link", "topic": "/camera/right/image_raw", "info_topic": "/camera/right/camera_info"},
    "rear": {"pos": [-0.20, CAMERA_REL_HEIGHT, 0.00], "yaw": 180.0, "frame_id": "rear_camera_link", "topic": "/camera/rear/image_raw", "info_topic": "/camera/rear/camera_info"},
}

def habitat_rgb_to_bgr(img):
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class HabitatBridgeVinebot(Node):
    def __init__(self):
        super().__init__("habitat_bridge_vinebot_2")
        os.makedirs(SAVE_DIR, exist_ok=True)

        self.sim = self.create_sim()
        self.agent = self.sim.initialize_agent(0)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)

        state = habitat_sim.AgentState()
        state.position = np.array([0.0, BASE_HEIGHT, 4.0], dtype=np.float32)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            np.deg2rad(180.0),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.agent.set_state(state)

        self.image_pubs = {name: self.create_publisher(Image, cfg["topic"], 10) for name, cfg in CAMERA_CONFIG.items()}
        self.info_pubs = {name: self.create_publisher(CameraInfo, cfg["info_topic"], 10) for name, cfg in CAMERA_CONFIG.items()}

        self.cmd_sub = self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)

        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.last_pose_log_time = 0.0
        self.photo_counter = 0

        self.timer = self.create_timer(0.1, self.timer_callback)

        cv2.namedWindow("Vinebot Habitat Bridge", cv2.WINDOW_NORMAL)
        self.get_logger().info("Habitat Bridge Vinebot 2 started")
        self.get_logger().info("WASD: manual control | /cmd_vel: ROS control | P: save photos | Q: quit")
        self.get_logger().info(f"Photos will save to: {SAVE_DIR}")

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
            "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=STEP_FORWARD)),
            "move_backward": habitat_sim.agent.ActionSpec("move_backward", habitat_sim.agent.ActuationSpec(amount=STEP_FORWARD)),
            "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=STEP_TURN_DEG)),
            "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=STEP_TURN_DEG)),
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
        cv2.rectangle(out, (0, 0), (260, 32), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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

    def make_camera_info(self, frame_id):
        msg = CameraInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.width = CAM_W
        msg.height = CAM_H
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [FX, 0.0, CX, 0.0, 0.0, FY, CY, 0.0, 0.0, 0.0, 1.0, 0.0]
        return msg

    def publish_image_and_info(self, name, frame):
        frame_id = CAMERA_CONFIG[name]["frame_id"]
        self.image_pubs[name].publish(self.np_to_image_msg(frame, frame_id))
        self.info_pubs[name].publish(self.make_camera_info(frame_id))

    def get_yaw_from_quat(self, q):
        try:
            x = float(q.x)
            y = float(q.y)
            z = float(q.z)
            w = float(q.w)
        except AttributeError:
            return 0.0
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_tf_and_odom(self):
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

        odom = Odometry()
        odom.header.stamp = t.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.last_cmd_linear)
        odom.twist.twist.angular.z = float(self.last_cmd_angular)
        self.odom_pub.publish(odom)

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if now_sec - self.last_pose_log_time >= 1.0:
            yaw = self.get_yaw_from_quat(q)
            self.get_logger().info(
                f"Pose | x={float(pos[0]):.2f}, y={float(pos[1]):.2f}, z={float(pos[2]):.2f}, yaw={math.degrees(yaw):.1f} deg"
            )
            self.last_pose_log_time = now_sec

    def save_photos(self, front, left, right, rear):
        self.photo_counter += 1
        prefix = f"pose_{self.photo_counter:04d}"
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_front.jpg"), front)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_left.jpg"), left)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_right.jpg"), right)
        cv2.imwrite(os.path.join(SAVE_DIR, f"{prefix}_rear.jpg"), rear)

        state = self.agent.get_state()
        pos = state.position
        yaw = self.get_yaw_from_quat(state.rotation)
        self.get_logger().info(
            f"Saved photos #{self.photo_counter} to {SAVE_DIR} | x={float(pos[0]):.2f}, y={float(pos[1]):.2f}, z={float(pos[2]):.2f}, yaw={math.degrees(yaw):.1f} deg"
        )

    def timer_callback(self):
        self.apply_ros_cmd()
        self.publish_tf_and_odom()

        obs = self.sim.get_sensor_observations()
        front = habitat_rgb_to_bgr(obs["front"])
        left = habitat_rgb_to_bgr(obs["left"])
        right = habitat_rgb_to_bgr(obs["right"])
        rear = habitat_rgb_to_bgr(obs["rear"])

        self.publish_image_and_info("front", front)
        self.publish_image_and_info("left", left)
        self.publish_image_and_info("right", right)
        self.publish_image_and_info("rear", rear)

        state = self.agent.get_state()
        pos = state.position
        yaw_deg = math.degrees(self.get_yaw_from_quat(state.rotation))

        front_view = self.add_label(front, f"Front | x={float(pos[0]):.2f} y={float(pos[1]):.2f}")
        left_view = self.add_label(left, "Left")
        right_view = self.add_label(right, "Right")
        rear_view = self.add_label(rear, f"Rear | yaw={yaw_deg:.1f} deg")

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
        elif key == ord("p"):
            self.save_photos(front, left, right, rear)
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
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == "__main__":
    main()
