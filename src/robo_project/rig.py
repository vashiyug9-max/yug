"""Load the VineBot sensor rig description (config/vinebot_rig.yaml). No ROS imports."""

from pathlib import Path

import numpy as np
import yaml

CAM_ORDER = ["front", "right", "back", "left"]   # panorama order used by the paper's model


def default_rig_path() -> str:
    for parent in Path(__file__).resolve().parents:
        for cand in (parent / "config" / "vinebot_rig.yaml",
                     parent / "src" / "robo_project" / "config" / "vinebot_rig.yaml",
                     parent / "share" / "robo_project" / "config" / "vinebot_rig.yaml"):
            if cand.exists():
                return str(cand)
    return ""


def load_rig(path: str = "") -> dict:
    path = path or default_rig_path()
    if not path:
        raise FileNotFoundError("vinebot_rig.yaml not found; pass its path explicitly")
    rig = yaml.safe_load(Path(path).read_text())
    for name in CAM_ORDER:
        cam = rig["cameras"][name]
        cam["yaw"] = np.deg2rad(float(cam["yaw_deg"]))
        cam["hfov"] = np.deg2rad(float(cam["hfov_deg"]))
    rig["path"] = path
    return rig
