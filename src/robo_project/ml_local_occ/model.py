import torch.nn as nn


class SimpleLocalOccNet(nn.Module):
    """
    CNN model for local occupancy-grid prediction.

    Input:
        4 RGB camera views stacked together:
        front + left + right + rear = 12 channels

    Output:
        64 x 64 occupancy logits

    Note:
        The layer name is kept as self.net so that older trained
        checkpoints such as local_occ_rgb4_1000samples.pth can load correctly.
    """

    def __init__(self, input_channels=12):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)
