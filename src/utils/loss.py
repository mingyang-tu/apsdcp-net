import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon2 = epsilon**2

    def forward(self, output, target):
        loss = torch.sqrt(torch.pow(output - target, 2) + self.epsilon2)
        return torch.mean(loss)
