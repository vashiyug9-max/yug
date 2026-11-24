#!/usr/bin/env python3
"""
Compare FusionOccNet checkpoints on a held-out test set (.npz samples from collect_data.py).

  python -m robo_project.ml.tools.eval_testset runs/fusion_mid360_e3.0/best.pt runs/fusion_mid360_unk_e3.0/best.pt \
      --data /home/yug/11_nov_robo_project_ws/data/fusion_mid360_test [--scene IMR_lab8]

3 classes, all cells: every cell's predicted value 1 / 0 / -1 against the target 1 / 0 / -1 (3-class models
predict -1 themselves; 1-output models get -1 from the rig's visibility on their prediction):
  acc, IoU of each class and mIoU = their mean.
Two binary targets:
  seen cells  the 1 / 0 / -1 target: only cells a sensor can see (-1 = unknown is left out)
  all cells   the old 1 / 0 target over the whole 3 x 3 m grid (inside obstacles included)
Scores (cell = 4.7 cm, prediction free if p >= 0.5):
  mIoU       mean of the free-space IoU and the occupied IoU
  occ IoU    IoU of occupied cells
  missed     share of occupied cells predicted free (dangerous)
  false      share of free cells predicted occupied (false alarms)
--save DIR [--save-n N]: for the first N test samples, per model: <sample>_target.npy and <sample>_pred.npy
(64 x 64 int8, 1 = free, 0 = occupied, -1 = unknown; the prediction's -1 cells come from the predicted
obstacles, as on the robot) and <sample>.png (target | prediction; white free, black occupied, grey unknown).
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from robo_project.common.rig import load_rig
from robo_project.ml.fusion.fusion_data import FusionDataset
from robo_project.ml.fusion.fusion_model import FusionOccNet, load_fusion_net, to_occupancy
from robo_project.ml.fusion.visibility import Visibility, body_half, sensor_views


def free_pred(model, pano, geo, device):
    """-> (B, 1, G, G) bool predicted free (3-class models: not 'occupied'), and the raw output."""
    out = model(pano.to(device), geo.to(device))
    if out.shape[1] == 3:
        return (out.argmax(1, keepdim=True) != 0), out
    return torch.sigmoid(out) >= 0.5, out


@torch.no_grad()
def scores3(model, loader, device, vis):
    """3 classes over all cells: predicted 1 / 0 / -1 vs target 1 / 0 / -1."""
    conf = np.zeros((3, 3))
    for pano, geo, label in loader:
        _, out = free_pred(model, pano, geo, device)
        if out.shape[1] == 3:
            pred = out.argmax(1).cpu().numpy()                       # 0 occ, 1 free, 2 unknown
        else:
            free = torch.sigmoid(out[:, 0]).cpu().numpy()
            pred = np.stack([np.where(vis.visible(f), (f >= 0.5).astype(int), 2) for f in free])
        lab = label[:, 0].numpy()
        tgt = np.where(lab < 0, 2, np.where(lab >= 0.5, 1, 0))
        conf += np.bincount((tgt * 3 + pred).ravel(), minlength=9).reshape(3, 3)
    tp = np.diag(conf)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    return dict(acc=tp.sum() / conf.sum(), miou=iou.mean(), occ=iou[0], free=iou[1], unk=iou[2])


@torch.no_grad()
def scores(model, loader, device):
    tp_o = fp_o = fn_o = tp_f = fp_f = fn_f = 0.0
    for pano, geo, label in loader:
        pred_free, _ = free_pred(model, pano, geo, device)
        lab = label.to(device)
        known = lab >= 0
        lab_free = lab >= 0.5
        po, lo = ~pred_free & known, ~lab_free & known
        pf, lf = pred_free & known, lab_free & known
        tp_o += (po & lo).sum().item(); fp_o += (po & ~lo).sum().item(); fn_o += (~po & lo).sum().item()
        tp_f += (pf & lf).sum().item(); fp_f += (pf & ~lf).sum().item(); fn_f += (~pf & lf).sum().item()
    iou_o = tp_o / max(tp_o + fp_o + fn_o, 1)
    iou_f = tp_f / max(tp_f + fp_f + fn_f, 1)
    occ, free = tp_o + fn_o, tp_f + fn_f
    return dict(miou=(iou_o + iou_f) / 2, occ_iou=iou_o, missed=fn_o / max(occ, 1), false=fn_f / max(free, 1),
                occ_share=occ / max(occ + free, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--scene", default="", help="only this scene folder, e.g. IMR_lab8")
    ap.add_argument("--rig", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save", default="", help="directory for target / prediction grids of the first samples")
    ap.add_argument("--save-n", type=int, default=20)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rig = load_rig(args.rig)
    files = sorted(glob.glob(str(Path(args.data) / (args.scene or "**") / "*.npz"), recursive=True))
    print(f"{len(files)} test samples{' (' + args.scene + ')' if args.scene else ''}")
    for ck in args.checkpoints:
        model, cfg = load_fusion_net(ck)
        model.to(device).eval()
        extent = float(cfg["extent"])
        print(f"\n{ck}")
        vis = Visibility(64, extent, sensor_views(rig), body_half(rig))
        dl = DataLoader(FusionDataset(files, extent, rig), batch_size=32, num_workers=args.workers)
        r = scores3(model, dl, device, vis)
        print(f"  3 classes, all cells  acc {r['acc']:6.1%}  mIoU {r['miou']:6.1%}  IoU occupied {r['occ']:6.1%}  "
              f"free {r['free']:6.1%}  unknown {r['unk']:6.1%}", flush=True)
        for name, unknown in (("seen cells (1/0/-1)", True), ("all cells (old 1/0)", False)):
            dl = DataLoader(FusionDataset(files, extent, rig, unknown=unknown), batch_size=32,
                            num_workers=args.workers)
            s = scores(model, dl, device)
            print(f"  {name:20s} mIoU {s['miou']:6.1%}  occ IoU {s['occ_iou']:6.1%}  missed {s['missed']:5.1%}  "
                  f"false {s['false']:5.1%}   (occupied = {s['occ_share']:.1%} of the cells)", flush=True)
        if args.save:
            save_pairs(model, files[:args.save_n], extent, rig, device, Path(args.save) / Path(ck).parent.name)


@torch.no_grad()
def save_pairs(model, files, extent, rig, device, out: Path):
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    ds = FusionDataset(files, extent, rig)
    vis = Visibility(64, extent, sensor_views(rig), body_half(rig))
    show = lambda g: cv2.resize(np.where(g < 0, 128, np.where(g > 0, 255, 0)).astype(np.uint8), (256, 256),
                                interpolation=cv2.INTER_NEAREST)
    for i, f in enumerate(files):
        pano, geo, label = ds[i]
        raw = model(torch.from_numpy(pano)[None].to(device), torch.from_numpy(geo)[None].to(device))[0]
        if raw.shape[0] == 3:
            pred = np.array([0, 1, -1], np.int8)[raw.argmax(0).cpu().numpy()]
        else:
            free = torch.sigmoid(raw[0]).cpu().numpy()
            pred = to_occupancy(free, vis.visible(free))
        target = np.where(label[0] < 0, -1, np.where(label[0] >= 0.5, 1, 0)).astype(np.int8)
        name = Path(f).stem
        np.save(out / f"{name}_target.npy", target)
        np.save(out / f"{name}_pred.npy", pred)
        cv2.imwrite(str(out / f"{name}.png"), np.hstack([show(target), np.full((256, 8), 200, np.uint8), show(pred)]))
    print(f"  saved {len(files)} target / prediction pairs in {out}")


if __name__ == "__main__":
    main()
