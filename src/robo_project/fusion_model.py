"""
Camera + LiDAR (+ front depth) fusion model for the local occupancy grid (no ROS imports).

Inputs
  pano  (B, 3, S, 4S)   front | right | back | left RGB images, ImageNet-normalised (paper's panorama)
  geo   (B, 3, G, G)    channel 0: LiDAR grid      (1 free, 0 occupied, 0.5 unknown / LiDAR missing)
                        channel 1: front depth grid (same encoding; 0.5 outside the camera FOV)
                        channel 2: LiDAR available  (1 / 0)
Output
  logits (B, 1, G, G)   free-space logit of the egocentric local grid (forward = top row), G = 64

Architecture
  camera branch: ResNet-18 trunk on the panorama -> pooled to 2 x 8 (8 horizontal sectors of 45 deg)
                 -> MLP -> 64 x 8 x 8 "top-down" features (learned image-to-map remapping, as in the paper)
  geometry branch: 3 strided convs, G x G -> 64 x 8 x 8
  fusion: concatenate -> conv -> transposed-conv decoder to G x G
The camera trunk can be initialised from the paper's trained LocalOccNet (trained on panoramas).
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def up_bn(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FusionOccNet(nn.Module):
    def __init__(self, grid: int = 64, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        assert grid == 64, "decoder is built for 64 x 64 output"
        r = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.cam_reduce = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d((2, 8)))
        self.cam_to_map = nn.Sequential(nn.Flatten(), nn.Linear(128 * 2 * 8, 1024), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(1024, 64 * 8 * 8), nn.ReLU(inplace=True))
        self.geo = nn.Sequential(conv_bn(3, 32, 2), conv_bn(32, 64, 2), conv_bn(64, 64, 2))   # 64 -> 8
        self.fuse = conv_bn(128, 128)
        self.decoder = nn.Sequential(up_bn(128, 64), up_bn(64, 32), up_bn(32, 16), nn.Conv2d(16, 1, 1))

    def forward(self, pano, geo):
        cam = self.cam_to_map(self.cam_reduce(self.trunk(pano))).view(-1, 64, 8, 8)
        x = self.fuse(torch.cat([cam, self.geo(geo)], dim=1))
        return self.decoder(x)

    def init_trunk_from_cmn(self, path: str) -> int:
        """Copy the ResNet-18 trunk of the paper's LocalOccNet (keys conv_layer.0 ... conv_layer.7)."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        own = self.trunk.state_dict()
        copied = 0
        for k, v in state.items():
            if k.startswith("conv_layer."):
                kk = k[len("conv_layer."):]
                if kk in own and own[kk].shape == v.shape:
                    own[kk] = v
                    copied += 1
        self.trunk.load_state_dict(own)
        return copied


# ── input preparation shared by training, evaluation and the ROS node ────────

def default_fusion_model() -> str:
    """The trained checkpoint shipped in the workspace (runs/fusion_full_e1.2/best.pt), or ''."""
    from pathlib import Path
    for parent in Path(__file__).resolve().parents:
        cand = parent / "runs" / "fusion_full_e1.2" / "best.pt"
        if cand.exists():
            return str(cand)
    return ""


def make_pano(rgb: dict, size: int, cams=("front", "right", "back", "left")) -> np.ndarray:
    """dict of HxWx3 uint8 RGB -> (3, size, 4*size) float32 normalised panorama. Missing camera -> zeros."""
    import cv2
    tiles = []
    for c in cams:
        im = rgb.get(c)
        if im is None:
            tiles.append(np.zeros((size, size, 3), np.float32))
        else:
            tiles.append((cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - MEAN) / STD)
    return np.concatenate(tiles, axis=1).transpose(2, 0, 1).astype(np.float32)


def make_geo(lidar_grid, depth_grid, grid: int = 64) -> np.ndarray:
    """LiDAR / depth grids (or None if missing) -> (3, grid, grid) geometry input."""
    import cv2
    def fit(g):
        g = np.asarray(g, np.float32)
        return g if g.shape == (grid, grid) else cv2.resize(g, (grid, grid), interpolation=cv2.INTER_AREA)
    lid = fit(lidar_grid) if lidar_grid is not None else np.full((grid, grid), 0.5, np.float32)
    dep = fit(depth_grid) if depth_grid is not None else np.full((grid, grid), 0.5, np.float32)
    avail = np.full((grid, grid), 1.0 if lidar_grid is not None else 0.0, np.float32)
    return np.stack([lid, dep, avail])


class FusionInputs:
    """
    Raw sensors -> LiDAR / depth grids, with the same settings FusionDataset uses for training
    (fusion_data.py), so the ROS node and the evaluation see exactly what the model was trained on.
    """

    def __init__(self, rig: dict, extent: float, grid: int = 64):
        from robo_project.lidar_local_occ import RangeLocalOcc
        band = rig.get("obstacle_band", [0.10, 1.80])
        self.front = rig["cameras"]["front"]
        self.lidar_z = float(rig["lidar"]["height"])
        self.depth_max = float(rig["depth_max"])
        self.lidar_occ = RangeLocalOcc(grid, extent, band[0], band[1], float(rig["lidar"]["max_range"]))
        self.depth_occ = RangeLocalOcc(grid, extent, band[0], band[1], self.depth_max, fov=self.front["hfov"])

    def lidar_grid(self, pts_base: np.ndarray, sensor_xy=(0.0, 0.0), sensor_z: float = None) -> np.ndarray:
        """Nx3 points in the base frame (x forward, y left, z up from the ground) -> grid."""
        z = self.lidar_z if sensor_z is None else sensor_z
        return self.lidar_occ.compute(pts_base, sensor_xy, z)[0]

    def depth_grid(self, depth: np.ndarray) -> np.ndarray:
        """Front depth image in metres -> grid (0.5 outside the camera FOV)."""
        from robo_project.lidar_local_occ import depth_to_points
        pts = depth_to_points(depth, self.front["hfov"], float(self.front["height"]), max_depth=self.depth_max)
        return self.depth_occ.compute(pts)[0]


def binarize_cells(free: np.ndarray, n: int = 3) -> np.ndarray:
    """Average a fine local grid into n x n cells and threshold at 0.5 -> crisp n x n grid (1 = free)."""
    import cv2
    return (cv2.resize(np.asarray(free, np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0.5).astype(np.float32)


def rule_fuse(lidar_grid, depth_grid):
    """No-model baseline: LiDAR grid, made occupied where the front depth sees an obstacle it missed."""
    if lidar_grid is None:
        return depth_grid
    if depth_grid is None:
        return lidar_grid
    known = depth_grid != 0.5
    out = lidar_grid.copy()
    out[known] = np.minimum(lidar_grid[known], depth_grid[known])
    return out


class FusionPredictor:
    """Load a trained FusionOccNet checkpoint and predict a free-probability grid."""

    def __init__(self, path: str, device: str = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = FusionOccNet(pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.extent = float(self.cfg["extent"])
        self.image_size = int(self.cfg["image_size"])

    @torch.no_grad()
    def predict(self, rgb: dict, lidar_grid, depth_grid) -> np.ndarray:
        pano = torch.from_numpy(make_pano(rgb, self.image_size))[None].to(self.device)
        geo = torch.from_numpy(make_geo(lidar_grid, depth_grid))[None].to(self.device)
        return torch.sigmoid(self.model(pano, geo))[0, 0].cpu().numpy()
