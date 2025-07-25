import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        kwargs["padding_mode"] = "reflect"
        self.conv = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
        return self.conv(x)


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super().__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(dim, d, 1, bias=False), nn.ReLU(), nn.Conv2d(d, dim * height, 1, bias=False))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out
