import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_

from .arch_util import Conv2d, SKFusion
from .gunet import SUNet


class IGMSA(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            nn.GELU(),
            Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

    def forward(self, x_in, illu_feat):
        """
        x_in :      [B, C, H, W]
        illu_feat : [B, C, H, W]
        return :    [B, C, H, W]
        """
        B, C, H, W = x_in.shape
        x = x_in.view(B, C, H * W).transpose(1, 2)  # [B, N, C]
        illu_feat = illu_feat.view(B, C, H * W).transpose(1, 2)  # [B, N, C]

        q_in = self.to_q(x)
        k_in = self.to_k(x)
        v_in = self.to_v(x)
        q = rearrange(q_in, "b n (h d) -> b h d n", h=self.num_heads)
        k = rearrange(k_in, "b n (h d) -> b h d n", h=self.num_heads)
        v = rearrange(v_in, "b n (h d) -> b h d n", h=self.num_heads)

        illu_feat = rearrange(illu_feat, "b n (h d) -> b h d n", h=self.num_heads)

        v = v * illu_feat

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # A = K^T*Q
        attn = k @ q.transpose(-2, -1)  # [B, H, D, D]
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # [B, H, D, N]
        x = x.permute(0, 3, 1, 2)  # [B, N, H, D]
        x = x.reshape(B, H * W, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        out_p = self.pos_emb(v_in.reshape(B, H, W, C).permute(0, 3, 1, 2))  # [B, C, H, W]
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x :      [B, C, H, W]
        return : [B, C, H, W]
        """
        return self.net(x)


class IGAB(nn.Module):
    def __init__(self, dim, num_blocks, dim_head, heads):
        super().__init__()
        self.num_blocks = num_blocks
        self.norm1 = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(num_blocks)])
        self.norm2 = nn.ModuleList([nn.BatchNorm2d(dim) for _ in range(num_blocks)])
        self.msa = nn.ModuleList([IGMSA(dim=dim, dim_head=dim_head, heads=heads) for _ in range(num_blocks)])
        self.ffn = nn.ModuleList([FeedForward(dim=dim) for _ in range(num_blocks)])

    def forward(self, x, illu_feat):
        """
        x :         [B, C, H, W]
        illu_feat : [B, C, H, W]
        return :    [B, C, H, W]
        """
        for i in range(self.num_blocks):
            norm_x = self.norm1[i](x)
            x = self.msa[i](norm_x, illu_feat) + x
            norm_x = self.norm2[i](x)
            x = self.ffn[i](norm_x) + x
        return x


class Down(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = Conv2d(dim, dim * 2, 2, 2, 0, bias=False)

    def forward(self, x):
        return self.proj(x)


class Up(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.ConvTranspose2d(dim, dim // 2, 2, 2, 0)

    def forward(self, x):
        return self.proj(x)


class IlluminationEstimator(nn.Module):
    def __init__(self, dim, dim_in, dim_out):
        super().__init__()
        self.net = SUNet(dim_in, dim_out, dim)

    def forward(self, x_in):
        # x_in :             [B, C=6, H, W]
        # return illu_feat : [B, C, H, W]
        # return illu_map :  [B, C=3, H, W]

        illu_feat, illu_map = self.net(x_in)
        return illu_feat, illu_map


class Denoiser(nn.Module):
    def __init__(self, dim, num_blocks):
        super().__init__()

        # Input projection
        self.conv_in = Conv2d(3, dim, 3, 1, 1, bias=False)

        # illu downsample
        self.illu_down1 = Down(dim)
        self.illu_down2 = Down(dim * 2)

        # Encoder
        self.enc1 = IGAB(dim, num_blocks[0], dim_head=dim, heads=1)
        self.down1 = Down(dim)

        self.enc2 = IGAB(dim * 2, num_blocks[1], dim_head=dim, heads=2)
        self.down2 = Down(dim * 2)

        # Bottleneck
        self.bottleneck = IGAB(dim * 4, num_blocks[2], dim_head=dim, heads=4)

        # Decoder
        self.up2 = Up(dim * 4)
        self.fuse2 = SKFusion(dim * 2)
        self.dec2 = IGAB(dim * 2, num_blocks[1], dim_head=dim, heads=2)

        self.up1 = Up(dim * 2)
        self.fuse1 = SKFusion(dim)
        self.dec1 = IGAB(dim, num_blocks[0], dim_head=dim, heads=1)

        # Output projection
        self.conv_out = Conv2d(dim, 3, 3, 1, 1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_in, illu_feat):
        """
        x_in :   [B, C=3, H, W]
        illu_feat : [B, C, H, W]
        return : [B, C=3, H, W]
        """

        x = self.conv_in(x_in)

        enc1 = self.enc1(x, illu_feat)
        x = self.down1(enc1)
        illu_feat2 = self.illu_down1(illu_feat)

        enc2 = self.enc2(x, illu_feat2)
        x = self.down2(enc2)
        illu_feat4 = self.illu_down2(illu_feat2)

        x = self.bottleneck(x, illu_feat4)

        x = self.up2(x)
        x = self.fuse2([enc2, x])
        x = self.dec2(x, illu_feat2)

        x = self.up1(x)
        x = self.fuse1([enc1, x])
        x = self.dec1(x, illu_feat)

        x = self.conv_out(x)
        return x + x_in


class RefineNet(nn.Module):
    def __init__(self, n_feat=40):
        super().__init__()
        self.estimator = IlluminationEstimator(dim=n_feat, dim_in=6, dim_out=3)
        self.denoiser = Denoiser(dim=n_feat, num_blocks=[1, 2, 2])

    def forward(self, y_in, x_in):
        """
        y_in :      [B, C=3, H, W]      # coarse clean image
        x_in :      [B, C=3, H, W]      # hazy image
        illu_feat : [B, C, H, W]
        illu_map :  [B, C=3, H, W]
        return    : [B, C=3, H, W]
        """

        illu_feat, illu_map = self.estimator(torch.cat([y_in, x_in], dim=1))
        x = x_in * illu_map + y_in * (1 - illu_map)
        output_img = self.denoiser(x, illu_feat)

        return output_img
