import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HabitatDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path

        self.csv_file = os.path.join(dataset_path, "ground_truth.csv")

        self.data = pd.read_csv(
            self.csv_file,
            dtype={"frame_id": str}
        )

        self.front_path = os.path.join(dataset_path, "front")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        frame_id = row["frame_id"].zfill(6)

        image_name = f"{frame_id}.png"

        image_path = os.path.join(self.front_path, image_name)

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        x = row["pos_x"]
        z = row["pos_z"]

        target = torch.tensor([x, z], dtype=torch.float32)

        return image, target

