#!/usr/bin/env python3
"""
Train the camera + LiDAR (+ front depth) fusion model on data from fusion_data.py (GPU server).

  python -m robo_project.ml.fusion.train_fusion --data data/fusion_real_rig --extent 3.0 \
      --init-from runs/fusion_full_e1.2/best.pt --epochs 8 --out runs/fusion_real_rig_e3.0

--extent must equal 3 x the coarse cell size used for navigation (e.g. MPP 0.4 -> 1.2 m; on the real
VineBot cell_size 1.0 -> 3.0 m). Train one model per extent.

Reported every epoch on the validation set (all sensors on, no augmentation):
  cell3   3x3 cell accuracy: what the Bayes filter actually compares with the coarse map
  iou     IoU of occupied space on the 64 x 64 grid
plus, at the end, the same metrics for the LiDAR-only and rule-fusion baselines and for the model
with each sensor removed (shows what the cameras add).
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from robo_project.ml.fusion.fusion_data import FusionDataset, split_files
from robo_project.ml.fusion.fusion_model import FusionOccNet, rule_fuse
from robo_project.common.rig import load_rig, rig_signature


def metrics(pred_free: torch.Tensor, label: torch.Tensor):
    """pred_free, label: (B, 1, G, G) in [0, 1]. Returns (cell3 accuracy, occupied IoU) sums and count."""
    p3 = F.adaptive_avg_pool2d(pred_free, 3) > 0.5
    l3 = F.adaptive_avg_pool2d(label, 3) > 0.5
    cell3 = (p3 == l3).float().mean(dim=(1, 2, 3))
    po, lo = pred_free < 0.5, label < 0.5
    inter = (po & lo).float().sum(dim=(1, 2, 3))
    union = (po | lo).float().sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    return cell3.sum().item(), iou.sum().item(), len(cell3)


@torch.no_grad()
def evaluate(model, loader, device, mode="model"):
    """mode: 'model', 'lidar' (LiDAR grid as prediction) or 'rule' (rule_fuse of LiDAR + depth)."""
    if model is not None:
        model.eval()
    c3 = io = n = 0
    for pano, geo, label in loader:
        label = label.to(device)
        if mode == "model":
            pred = torch.sigmoid(model(pano.to(device), geo.to(device)))
        else:
            out = []
            for g in geo.numpy():
                lid = g[0] if g[2].mean() > 0.5 else None
                out.append(g[0] if mode == "lidar" else rule_fuse(lid, g[1]))
            pred = torch.from_numpy(np.stack(out))[:, None].to(device)
        a, b, k = metrics(pred, label)
        c3, io, n = c3 + a, io + b, n + k
    return c3 / max(n, 1), io / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="directories with .npz samples (searched recursively)")
    ap.add_argument("--extent", type=float, required=True, help="local grid side in m = 3 x coarse cell size")
    ap.add_argument("--rig", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-scenes", nargs="*", default=[], help="hold out whole scenes (better test of new places)")
    ap.add_argument("--init-cmn", default="",
                    help="paper LocalOccNet checkpoint to initialise the camera trunk ('' = none)")
    ap.add_argument("--init-from", default="",
                    help="fusion checkpoint to fine-tune from (e.g. runs/fusion_full_e1.2/best.pt)")
    ap.add_argument("--no-imagenet", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="limit training samples (quick tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/fusion")
    args = ap.parse_args()

    rig = load_rig(args.rig)
    files = sorted(f for d in args.data for f in glob.glob(str(Path(d) / "**" / "*.npz"), recursive=True))
    train_files, val_files = split_files(files, args.val_frac, args.val_scenes)
    if args.max_train:
        train_files = train_files[:args.max_train]
    print(f"{len(train_files)} training / {len(val_files)} validation samples, extent {args.extent} m")
    if not train_files or not val_files:
        raise SystemExit("Need both training and validation samples")

    device = torch.device(args.device)
    train_ds = FusionDataset(train_files, args.extent, rig, augment=True)
    val_ds = FusionDataset(val_files, args.extent, rig)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=args.batch, shuffle=shuffle, num_workers=args.workers,
                                        pin_memory=device.type == "cuda", drop_last=shuffle,
                                        persistent_workers=args.workers > 0)
    train_dl, val_dl = mk(train_ds, True), mk(val_ds, False)

    model = FusionOccNet(pretrained=not args.no_imagenet)
    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)["model_state"]
        model.load_state_dict(state)
        print(f"Fine-tuning from {args.init_from}")
    init = args.init_cmn
    if init and not args.init_from:
        print(f"Camera trunk initialised from the paper's model ({model.init_trunk_from_cmn(init)} tensors): {init}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(extent=args.extent, image_size=int(rig["image_size"]), grid=64, rig=rig["path"],
                  data=args.data, n_train=len(train_files), n_val=len(val_files), init_from=args.init_from,
                  # the sensor layout the model was trained for (the node warns if the robot differs)
                  rig_sensors=rig_signature(rig))
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for pano, geo, label in train_dl:
            pano, geo, label = pano.to(device, non_blocking=True), geo.to(device, non_blocking=True), label.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(pano, geo), label)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, steps = total + loss.item(), steps + 1
        cell3, iou = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch, loss=total / steps, cell3=cell3, iou=iou))
        print(f"epoch {epoch:3d}  loss {total / steps:.4f}  val cell3 {cell3:.1%}  iou {iou:.1%}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        ckpt = dict(model_state=model.state_dict(), config=config, epoch=epoch, val_cell3=cell3, val_iou=iou)
        torch.save(ckpt, out / "last.pt")
        if cell3 > best:
            best = cell3
            torch.save(ckpt, out / "best.pt")

    # Final comparison on the validation set.
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu", weights_only=False)["model_state"])
    report = {"model (all sensors)": evaluate(model, val_dl, device),
              "baseline: LiDAR only": evaluate(None, val_dl, device, "lidar"),
              "baseline: rule fusion LiDAR+depth": evaluate(None, val_dl, device, "rule")}
    for drop in ("lidar", "cams", "depth"):
        dl = mk(FusionDataset(val_files, args.extent, rig, drop=(drop,)), False)
        report[f"model without {drop}"] = evaluate(model, dl, device)
    print("\nValidation (cell3 = 3x3 accuracy used by CMN, iou = occupied IoU)")
    for k, (c, i) in report.items():
        print(f"  {k:36s} cell3 {c:6.1%}  iou {i:6.1%}")
    (out / "report.json").write_text(json.dumps(dict(history=history, final={k: dict(cell3=c, iou=i)
                                                                               for k, (c, i) in report.items()}),
                                                indent=2))
    save_examples(model, val_ds, device, out / "examples.png")
    print(f"Saved {out / 'best.pt'}, {out / 'report.json'} and {out / 'examples.png'}")


@torch.no_grad()
def save_examples(model, ds, device, path, n=6):
    """One row per validation sample: panorama | LiDAR grid | front depth grid | prediction | ground truth."""
    import cv2
    from robo_project.ml.fusion.fusion_model import MEAN, STD
    model.eval()
    rows = []
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    for i in idx:
        pano, geo, label = ds[int(i)]
        pred = torch.sigmoid(model(torch.from_numpy(pano)[None].to(device),
                                   torch.from_numpy(geo)[None].to(device)))[0, 0].cpu().numpy()
        img = np.clip((pano.transpose(1, 2, 0) * STD + MEAN) * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.resize(img, (512, 128)), cv2.COLOR_RGB2BGR)
        tiles = []
        for g in (geo[0], geo[1], pred, label[0]):
            t = cv2.resize((np.clip(g, 0, 1) * 255).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            for k in (1, 2):                                  # 3x3 coarse-cell lines
                cv2.line(t, (0, 128 * k // 3), (127, 128 * k // 3), (0, 160, 255), 1)
                cv2.line(t, (128 * k // 3, 0), (128 * k // 3, 127), (0, 160, 255), 1)
            cv2.circle(t, (64, 64), 3, (0, 0, 255), -1)
            tiles.append(t)
        rows.append(np.hstack([img] + tiles))
    header = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    for x, text in ((5, "cameras: front | right | back | left"), (517, "LiDAR"), (645, "front depth"),
                    (773, "prediction"), (901, "ground truth")):
        cv2.putText(header, text, (x, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(path), np.vstack([header] + rows))


if __name__ == "__main__":
    main()
