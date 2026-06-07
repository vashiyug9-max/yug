import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset_loader import HabitatDataset
from model import PoseEstimator


dataset_path = "../habitat_dataset"

dataset = HabitatDataset(dataset_path)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

device = torch.device("cpu")

model = PoseEstimator().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):

    running_loss = 0.0

    for images, targets in dataloader:

        images = images.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {average_loss:.4f}")

torch.save(
    model.state_dict(),
    "pose_estimator.pth"
)

print("Model saved successfully.")
