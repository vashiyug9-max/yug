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


def rig_signature(rig: dict) -> dict:
    """The parts of the rig a trained model depends on (camera / LiDAR placement), rounded to 1 cm."""
    r = lambda v: round(float(v), 2)
    sig = {name: dict(x=r(c.get("x", 0.0)), y=r(c.get("y", 0.0)), height=r(c["height"]),
                      yaw_deg=r(c["yaw_deg"]), hfov_deg=r(c["hfov_deg"]))
           for name, c in rig["cameras"].items()}
    lid = rig["lidar"]
    sig["lidar"] = dict(x=r(lid.get("x", 0.0)), y=r(lid.get("y", 0.0)), height=r(lid["height"]))
    return sig


def lidar_to_base(pts: np.ndarray, rig: dict) -> np.ndarray:
    """Nx3 points in the LiDAR's own frame (as the Livox driver publishes them) -> base_footprint,
    with the mounting in the rig: rotate by lidar yaw_deg, then shift by x / y / height."""
    lid = rig["lidar"]
    yaw = np.deg2rad(float(lid.get("yaw_deg", 0.0)))
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1] + float(lid.get("x", 0.0))
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1] + float(lid.get("y", 0.0))
    out[:, 2] = pts[:, 2] + float(lid["height"])
    return out


def drop_self_hits(pts_base: np.ndarray, rig: dict, margin: float = 0.05, above_lidar: float = 0.10) -> np.ndarray:
    """Remove returns from the robot's own body: points inside the footprint box (robot length x width
    + margin) that are not higher than lidar height + above_lidar. In the real recording the
    Mid-360 (front mount, 0.45 m) hits the deck / e-stop 0.18-0.7 m behind it at 0.42-0.53 m:
    all inside this box. Higher points in the box are kept, so a wall touching the robot still shows up."""
    rob = rig.get("robot")
    if not rob or len(pts_base) == 0:
        return pts_base
    hx, hy = float(rob["length"]) / 2 + margin, float(rob["width"]) / 2 + margin
    top = float(rig["lidar"]["height"]) + above_lidar
    inside = (np.abs(pts_base[:, 0]) <= hx) & (np.abs(pts_base[:, 1]) <= hy) & (pts_base[:, 2] <= top)
    return pts_base[~inside]
