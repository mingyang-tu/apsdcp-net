import torch.nn as nn
from .arch_util import Conv2d


class ANet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            Conv2d(3, 16, 7, 2, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Conv2d(128, 3, 1, 1, 0),
        )

    def forward(self, x):
        x = self.net(x)
        return x
