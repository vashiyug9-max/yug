"""
Camera + LiDAR (+ front depth) fusion model for the local occupancy grid (no ROS imports).

Inputs
  pano  (B, 3, S, 4S)   front | right | back | left RGB images, ImageNet-normalised (paper's panorama)
  geo   (B, 3, G, G)    channel 0: LiDAR grid      (1 free, 0 occupied, 0.5 unknown / LiDAR missing)
                        channel 1: front depth grid (same encoding; 0.5 outside the camera FOV)
                        channel 2: LiDAR available  (1 / 0)
Output
  n_classes = 1: logits (B, 1, G, G)  free-space logit of the egocentric local grid (forward = top row), G = 64
  n_classes = 3: logits (B, 3, G, G)  class scores per cell: 0 = occupied, 1 = free, 2 = unknown (target -1);
                 trained with cross-entropy on ALL cells, the output is the most likely class -> 1 / 0 / -1

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
    def __init__(self, grid: int = 64, pretrained: bool = True, dropout: float = 0.3, n_classes: int = 1):
        super().__init__()
        self.n_classes = n_classes
        assert grid == 64, "decoder is built for 64 x 64 output"
        r = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.cam_reduce = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d((2, 8)))
        self.cam_to_map = nn.Sequential(nn.Flatten(), nn.Linear(128 * 2 * 8, 1024), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(1024, 64 * 8 * 8), nn.ReLU(inplace=True))
        self.geo = nn.Sequential(conv_bn(3, 32, 2), conv_bn(32, 64, 2), conv_bn(64, 64, 2))   # 64 -> 8
        self.fuse = conv_bn(128, 128)
        self.decoder = nn.Sequential(up_bn(128, 64), up_bn(64, 32), up_bn(32, 16), nn.Conv2d(16, n_classes, 1))

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

# Checkpoints in <workspace>/runs, best first:
#   fusion_mid360_e3.0   measured rig + realistic Livox Mid-360 at the front mount (sim/lidar_sim.py), 3 m grid
#   fusion_real_rig_e3.0 measured cameras, LiDAR simulated as a flat 2D scan at the robot centre, 3 m grid
#   fusion_full_e1.2     original model (placeholder rig at 1.0 m, 1.2 m grid)
DEFAULT_MODELS = ("fusion_mid360_e3.0", "fusion_real_rig_e3.0", "fusion_full_e1.2")


def default_fusion_model() -> str:
    """The best trained checkpoint found in the workspace (see DEFAULT_MODELS), or ''."""
    from pathlib import Path
    for run in DEFAULT_MODELS:
        for parent in Path(__file__).resolve().parents:
            cand = parent / "runs" / run / "best.pt"
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
        from robo_project.ml.fusion.lidar_local_occ import RangeLocalOcc
        band = rig.get("obstacle_band", [0.10, 1.80])
        self.front = rig["cameras"]["front"]
        self.lidar_z = float(rig["lidar"]["height"])
        self.lidar_xy = (float(rig["lidar"].get("x", 0.0)), float(rig["lidar"].get("y", 0.0)))
        self.depth_max = float(rig["depth_max"])
        self.lidar_occ = RangeLocalOcc(grid, extent, band[0], band[1], float(rig["lidar"]["max_range"]))
        self.depth_occ = RangeLocalOcc(grid, extent, band[0], band[1], self.depth_max, fov=self.front["hfov"])

    def lidar_grid(self, pts_base: np.ndarray, sensor_xy=None, sensor_z: float = None) -> np.ndarray:
        """Nx3 points in the base frame (x forward, y left, z up from the ground) -> grid. Rays start at
        the LiDAR (rig x / y: at the front of the real robot), as in training."""
        z = self.lidar_z if sensor_z is None else sensor_z
        return self.lidar_occ.compute(pts_base, self.lidar_xy if sensor_xy is None else sensor_xy, z)[0]

    def depth_grid(self, depth: np.ndarray) -> np.ndarray:
        """Front depth image in metres -> grid (0.5 outside the camera FOV)."""
        from robo_project.ml.fusion.lidar_local_occ import depth_to_points
        # the camera can sit ahead of the robot centre (rig x): points and rays start at the camera
        cam_x = float(self.front.get("x", 0.0))
        pts = depth_to_points(depth, self.front["hfov"], float(self.front["height"]), cam_x=cam_x,
                              max_depth=self.depth_max)
        return self.depth_occ.compute(pts, (cam_x, 0.0))[0]


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


FREE, OCCUPIED, UNKNOWN = 1, 0, -1
CLS_OCC, CLS_FREE, CLS_UNK = 0, 1, 2          # class indices of the 3-class model
CLASS_TO_VALUE = np.array([OCCUPIED, FREE, UNKNOWN], np.int8)


def target_classes(label: np.ndarray) -> np.ndarray:
    """1 / 0 / -1 target grid (free values may be fractional after resizing) -> class index per cell."""
    lab = np.asarray(label)
    return np.where(lab < 0, CLS_UNK, np.where(lab >= 0.5, CLS_FREE, CLS_OCC)).astype(np.int64)


def load_fusion_net(path: str):
    """Checkpoint -> (FusionOccNet with the right number of outputs, config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = FusionOccNet(pretrained=False, n_classes=int(cfg.get("n_classes", 1)))
    net.load_state_dict(ckpt["model_state"])
    return net, cfg


def to_occupancy(free: np.ndarray, seen: np.ndarray = None) -> np.ndarray:
    """Free-probability grid (+ mask of the cells a sensor can see) -> int8 occupancy grid with the same
    values as the training target: 1 = free (p >= 0.5), 0 = occupied, -1 = unknown (no sensor sees it)."""
    out = np.where(np.asarray(free) >= 0.5, FREE, OCCUPIED).astype(np.int8)
    if seen is not None:
        out[~np.asarray(seen, bool)] = UNKNOWN
    return out


class FusionPredictor:
    """Load a trained FusionOccNet checkpoint and predict the local grid:
    predict()      -> free probability per cell (0..1), the raw network output after a sigmoid
    predict_grid() -> occupancy grid 1 = free, 0 = occupied, -1 = unknown (same values as the target)"""

    def __init__(self, path: str, device: str = None, rig: dict = None):
        self.model, self.cfg = load_fusion_net(path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.n_classes = self.model.n_classes
        self.extent = float(self.cfg["extent"])
        self.image_size = int(self.cfg["image_size"])
        self.vis = None
        if rig is not None:
            from robo_project.ml.fusion.visibility import Visibility, body_half, sensor_views
            self.vis = Visibility(64, self.extent, sensor_views(rig), body_half(rig))

    @torch.no_grad()
    def predict_full(self, rgb: dict, lidar_grid, depth_grid):
        """-> (free probability per cell, seen mask or None). 3-class model: free probability among the known
        classes, seen = the model's most likely class is not 'unknown'. 1-output model: sigmoid, seen from the
        rig's visibility if rig= was given (else None)."""
        pano = torch.from_numpy(make_pano(rgb, self.image_size))[None].to(self.device)
        geo = torch.from_numpy(make_geo(lidar_grid, depth_grid))[None].to(self.device)
        out = self.model(pano, geo)[0]
        if self.n_classes == 3:
            p = torch.softmax(out, dim=0).cpu().numpy()
            free = p[CLS_FREE] / np.maximum(p[CLS_FREE] + p[CLS_OCC], 1e-6)
            return free, p.argmax(axis=0) != CLS_UNK
        free = torch.sigmoid(out[0]).cpu().numpy()
        return free, (self.vis.visible(free) if self.vis is not None else None)

    def predict(self, rgb: dict, lidar_grid, depth_grid) -> np.ndarray:
        return self.predict_full(rgb, lidar_grid, depth_grid)[0]

    def predict_grid(self, rgb: dict, lidar_grid, depth_grid) -> np.ndarray:
        """64 x 64 int8: 1 = free, 0 = occupied, -1 = unknown (cells none of the rig's sensors can see,
        computed with the predicted obstacles; needs rig= in the constructor, else no -1 cells)."""
        free, seen = self.predict_full(rgb, lidar_grid, depth_grid)
        return to_occupancy(free, seen)
