import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block"""

    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depthwise
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x


class GeoConvNeXt(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 1):
        super().__init__()
        # Stem: 下采样 4 倍 (64x64 -> 16x16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=4),
            nn.LayerNorm([64, 16, 16])
        )

        # Stage 1
        self.stage1 = nn.Sequential(ConvNeXtBlock(64))

        # Downsample: (16x16 -> 8x8)
        self.downsample = nn.Sequential(
            nn.LayerNorm([64, 16, 16]),
            nn.Conv2d(64, 128, kernel_size=2, stride=2)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(128),
            ConvNeXtBlock(128),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        return self.head(x).squeeze(1)
