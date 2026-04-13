"""U-Net with a lightweight ResNet-style encoder (4 levels + bottleneck)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from code.models.blocks import ConvBNReLU, ResidualBlock


class EncoderBlock(nn.Module):
    """Two residual convs, returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ResidualBlock(in_ch, out_ch)
        self.conv2 = ResidualBlock(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        # x: (B, in_ch, H, W) -> skip: (B, out_ch, H, W)
        skip = self.conv2(self.conv1(x))
        # pooled: (B, out_ch, H/2, W/2)
        pooled = self.pool(skip)
        return pooled, skip


class DecoderBlock(nn.Module):
    """Up-conv, concat skip, two ConvBNReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, h, w) -> (B, out_ch, 2h, 2w)
        x = self.up(x)
        # Pad if spatial dims mismatch (odd input sizes)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # concat -> (B, out_ch + skip_ch, 2h, 2w)
        x = torch.cat([x, skip], dim=1)
        # -> (B, out_ch, 2h, 2w)
        return self.conv2(self.conv1(x))


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = ISPRS_NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        channels=(64, 128, 256, 512),
        bottleneck_ch: int = 1024,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        # Encoder: 4 levels
        self.e1 = EncoderBlock(in_channels, c1)  # (B,in,H,W)   -> skip(B,c1,H,W),   pooled(B,c1,H/2,W/2)
        self.e2 = EncoderBlock(c1, c2)           # -> skip(B,c2,H/2,W/2),   pooled(B,c2,H/4,W/4)
        self.e3 = EncoderBlock(c2, c3)           # -> skip(B,c3,H/4,W/4),   pooled(B,c3,H/8,W/8)
        self.e4 = EncoderBlock(c3, c4)           # -> skip(B,c4,H/8,W/8),   pooled(B,c4,H/16,W/16)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBNReLU(c4, bottleneck_ch),
            ConvBNReLU(bottleneck_ch, bottleneck_ch),
        )  # -> (B, bottleneck_ch, H/16, W/16)

        # Decoder: mirror of encoder
        self.d4 = DecoderBlock(bottleneck_ch, c4, c4)  # -> (B, c4, H/8,  W/8)
        self.d3 = DecoderBlock(c4, c3, c3)             # -> (B, c3, H/4,  W/4)
        self.d2 = DecoderBlock(c3, c2, c2)             # -> (B, c2, H/2,  W/2)
        self.d1 = DecoderBlock(c2, c1, c1)             # -> (B, c1, H,    W)

        # Final 1x1 classifier -> (B, num_classes, H, W)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H, W)
        p1, s1 = self.e1(x)
        p2, s2 = self.e2(p1)
        p3, s3 = self.e3(p2)
        p4, s4 = self.e4(p3)
        b = self.bottleneck(p4)   # (B, bottleneck_ch, H/16, W/16)
        x = self.d4(b, s4)        # (B, c4, H/8,  W/8)
        x = self.d3(x, s3)        # (B, c3, H/4,  W/4)
        x = self.d2(x, s2)        # (B, c2, H/2,  W/2)
        x = self.d1(x, s1)        # (B, c1, H,    W)
        return self.head(x)       # (B, num_classes, H, W)
