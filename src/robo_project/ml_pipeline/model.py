import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet18_Weights


class PoseEstimator(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        num_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(self, x):

        return self.model(x)
