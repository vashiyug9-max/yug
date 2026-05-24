import os
import math
import cv2
import numpy as np
import habitat_sim

SCENE = "/home/yug/habitat_data/scene_datasets/gibson/gibson/Denmark.glb"

CAM_H = 240
CAM_W = 320
PANO_H = 180
PANO_CAM_W = 160
STEP_FORWARD = 0.10
STEP_TURN_DEG = 5.0

def rgb_to_bgr(img):
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

class HabitatInterface:
    def __init__(self):
        self.sim = self._create_sim()
        self.agent = self.sim.initialize_agent(0)
        self.reset()

    def reset(self):
        state = habitat_sim.AgentState()
        state.position = np.array([0.0, 1.5, 4.0], dtype=np.float32)
        state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
            np.deg2rad(180.0),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.agent.set_state(state)

    def _make_sensor(self, uuid_name, yaw_deg, h, w):
        sensor = habitat_sim.CameraSensorSpec()
        sensor.uuid = uuid_name
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [h, w]
        sensor.position = [0.0, 1.5, 0.0]
        sensor.orientation = [0.0, math.radians(yaw_deg), 0.0]
        sensor.hfov = 60.0
        return sensor

    def _create_sim(self):
        if not os.path.exists(SCENE):
            raise FileNotFoundError(f"Scene not found: {SCENE}")

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = SCENE
        backend_cfg.enable_physics = False
        backend_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

        sensors = [
            self._make_sensor("front", 0.0, CAM_H, CAM_W),
            self._make_sensor("left", 90.0, CAM_H, CAM_W),
            self._make_sensor("right", -90.0, CAM_H, CAM_W),
            self._make_sensor("rear", 180.0, CAM_H, CAM_W),
            self._make_sensor("p0", -157.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p1", -112.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p2", -67.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p3", -22.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p4", 22.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p5", 67.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p6", 112.5, PANO_H, PANO_CAM_W),
            self._make_sensor("p7", 157.5, PANO_H, PANO_CAM_W),
        ]

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
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

    def act(self, action_name):
        self.agent.act(action_name)

    def get_observations(self):
        obs = self.sim.get_sensor_observations()

        views = {
            "front": rgb_to_bgr(obs["front"]),
            "left": rgb_to_bgr(obs["left"]),
            "right": rgb_to_bgr(obs["right"]),
            "rear": rgb_to_bgr(obs["rear"]),
        }

        pano_parts = []
        for name in ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]:
            pano_parts.append(rgb_to_bgr(obs[name]))

        pano = np.hstack(pano_parts)
        return views, pano

    def get_pano_rgb(self):
        _, pano = self.get_observations()
        return pano
