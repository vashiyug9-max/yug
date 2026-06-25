import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HabitatDatasetV2(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.samples = sorted([
            f.replace("_pose.json", "")
            for f in os.listdir(dataset_path)
            if f.endswith("_pose.json")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample_id = self.samples[idx]

        image_path = os.path.join(
            self.dataset_path,
            f"{sample_id}_front.jpg"
        )

        pose_path = os.path.join(
            self.dataset_path,
            f"{sample_id}_pose.json"
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        with open(pose_path, "r") as f:
            pose = json.load(f)

        x = pose["position"]["x"]
        z = pose["position"]["z"]

        target = torch.tensor(
            [x, z],
            dtype=torch.float32
        )

        return image, target
