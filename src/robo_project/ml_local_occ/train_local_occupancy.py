import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from .model import SimpleLocalOccNet
except ImportError:
    from model import SimpleLocalOccNet


class LocalOccDataset(Dataset):
    def __init__(self, dataset_root, samples, image_size=64):
        self.dataset_root = Path(dataset_root)
        self.samples = samples
        self.cameras = ["front", "left", "right", "rear"]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_dir = self.samples[index]

        image_tensors = []
        for cam in self.cameras:
            img_path = sample_dir / f"{cam}_rgb.png"
            img = Image.open(img_path).convert("RGB")
            image_tensors.append(self.transform(img))

        # 4 RGB cameras = 12 input channels
        x = torch.cat(image_tensors, dim=0)

        occ = np.load(sample_dir / "local_occupancy_gt.npy").astype(np.float32)

        # Dataset values:
        # free = 1, occupied = 0, unknown = -1
        mask = (occ != -1).astype(np.float32)
        target = np.where(occ == 1, 1.0, 0.0).astype(np.float32)

        target = torch.from_numpy(target)
        mask = torch.from_numpy(mask)

        return x, target, mask


def masked_bce_loss(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_pixels = 0.0

    for x, target, mask in loader:
        x = x.to(device)
        target = target.to(device)
        mask = mask.to(device)

        logits = model(x)
        loss = masked_bce_loss(logits, target, mask)

        pred = (torch.sigmoid(logits) > 0.5).float()
        correct = ((pred == target).float() * mask).sum()
        pixels = mask.sum()

        total_loss += loss.item()
        total_correct += correct.item()
        total_pixels += pixels.item()

    avg_loss = total_loss / max(len(loader), 1)
    acc = total_correct / max(total_pixels, 1.0)

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="ml_runs/local_occ_rgb4_best.pth")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    samples = sorted([
        p for p in dataset_root.glob("sample_*")
        if (p / "front_rgb.png").exists()
        and (p / "left_rgb.png").exists()
        and (p / "right_rgb.png").exists()
        and (p / "rear_rgb.png").exists()
        and (p / "local_occupancy_gt.npy").exists()
    ])

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    random.seed(42)
    random.shuffle(samples)

    split_index = int(0.8 * len(samples))
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]

    print(f"Total samples: {len(samples)}", flush=True)
    print(f"Train samples: {len(train_samples)}", flush=True)
    print(f"Val samples: {len(val_samples)}", flush=True)

    train_dataset = LocalOccDataset(dataset_root, train_samples)
    val_dataset = LocalOccDataset(dataset_root, val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    model = SimpleLocalOccNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = 999999.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        for batch_id, (x, target, mask) in enumerate(train_loader, start=1):
            x = x.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = masked_bce_loss(logits, target, mask)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            if batch_id % 5 == 0 or batch_id == 1:
                print(f"  batch {batch_id}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch} result: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_pixel_acc={val_acc:.4f}",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_pixel_acc": val_acc,
            }, out_path)
            print(f"Saved best model: {out_path}", flush=True)

    print("\nTraining finished.", flush=True)


if __name__ == "__main__":
    main()
