import torch.nn as nn
import torch.nn.functional as F

from .arch_util import SKFusion, Conv2d


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        self.Wv = nn.Sequential(
            Conv2d(dim, dim, 1, 1, 0),
            Conv2d(dim, dim, 3, 1, 1, groups=dim),
        )
        self.Wg = nn.Sequential(
            Conv2d(dim, dim, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.proj = Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.Wv(x) * self.Wg(x)
        x = self.proj(x)
        return x + identity


class Layer(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()

        self.net = nn.Sequential(*[Block(dim) for _ in range(num_blocks)])

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = Conv2d(dim, dim * 2, 2, 2, 0)

    def forward(self, x):
        return self.proj(x)


class Up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            Conv2d(dim, dim * 2, 1, 1, 0),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.proj(x)


class gUNet_T(nn.Module):
    def __init__(self, in_channels, out_channels, dim=24, num_blocks=[2, 2, 2, 4]):
        super().__init__()

        self.conv_in = Conv2d(in_channels, dim, 5, 1, 2)

        self.enc1 = Layer(dim, num_blocks[0])
        self.down1 = Down(dim)

        self.enc2 = Layer(dim * 2, num_blocks[1])
        self.down2 = Down(dim * 2)

        self.enc3 = Layer(dim * 4, num_blocks[2])
        self.down3 = Down(dim * 4)

        self.bottleneck = Layer(dim * 8, num_blocks[3])

        self.up3 = Up(dim * 8)
        self.fuse3 = SKFusion(dim * 4)
        self.dec3 = Layer(dim * 4, num_blocks[2])

        self.up2 = Up(dim * 4)
        self.fuse2 = SKFusion(dim * 2)
        self.dec2 = Layer(dim * 2, num_blocks[1])

        self.up1 = Up(dim * 2)
        self.fuse1 = SKFusion(dim)
        self.dec1 = Layer(dim, num_blocks[0])

        self.conv_out = Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)

        enc1 = self.enc1(x)
        x = self.down1(enc1)

        enc2 = self.enc2(x)
        x = self.down2(enc2)

        enc3 = self.enc3(x)
        x = self.down3(enc3)

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self.fuse3([x, enc3])
        x = self.dec3(x)

        x = self.up2(x)
        x = self.fuse2([x, enc2])
        x = self.dec2(x)

        x = self.up1(x)
        x = self.fuse1([x, enc1])
        x = self.dec1(x)

        x = self.conv_out(x)
        return F.sigmoid(x)


class SUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_blocks=[1, 2]):
        super().__init__()

        self.conv_in = Conv2d(in_channels, dim, 3, 1, 1)

        self.enc1 = Layer(dim, num_blocks[0])
        self.down1 = Down(dim)

        self.bottleneck = Layer(dim * 2, num_blocks[1])

        self.up1 = Up(dim * 2)
        self.fuse1 = SKFusion(dim)
        self.dec1 = Layer(dim, num_blocks[0])

        self.conv_out = Conv2d(dim, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)

        enc1 = self.enc1(x)
        x = self.down1(enc1)

        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.fuse1([x, enc1])
        feat = self.dec1(x)

        out = self.conv_out(feat)
        return feat, out
