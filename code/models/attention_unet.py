"""Attention U-Net: same encoder as UNet + MHSA bottleneck + AttentionGate skips."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from code.models.blocks import AttentionGate, ConvBNReLU, MultiHeadSelfAttention
from code.models.unet import EncoderBlock


class AttentionDecoderBlock(nn.Module):
    """Up-conv, gate skip with AttentionGate, concat, two ConvBNReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.gate = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(skip_ch // 2, 1))
        self.conv1 = ConvBNReLU(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, h, w) -> (B, out_ch, 2h, 2w)
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Apply gate on the skip using upsampled x as gating signal
        # skip: (B, skip_ch, 2h, 2w) -> gated (same shape)
        gated = self.gate(g=x, x=skip)
        # concat -> (B, out_ch + skip_ch, 2h, 2w)
        x = torch.cat([x, gated], dim=1)
        return self.conv2(self.conv1(x))


class AttentionUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = ISPRS_NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        channels=(64, 128, 256, 512),
        bottleneck_ch: int = 1024,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels

        self.e1 = EncoderBlock(in_channels, c1)
        self.e2 = EncoderBlock(c1, c2)
        self.e3 = EncoderBlock(c2, c3)
        self.e4 = EncoderBlock(c3, c4)

        # Bottleneck: ConvBNReLU -> MHSA -> ConvBNReLU, shape (B, bottleneck_ch, H/16, W/16)
        self.b_pre = ConvBNReLU(c4, bottleneck_ch)
        self.b_mhsa = MultiHeadSelfAttention(bottleneck_ch, num_heads=num_heads)
        self.b_post = ConvBNReLU(bottleneck_ch, bottleneck_ch)

        self.d4 = AttentionDecoderBlock(bottleneck_ch, c4, c4)
        self.d3 = AttentionDecoderBlock(c4, c3, c3)
        self.d2 = AttentionDecoderBlock(c3, c2, c2)
        self.d1 = AttentionDecoderBlock(c2, c1, c1)

        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H, W)
        p1, s1 = self.e1(x)
        p2, s2 = self.e2(p1)
        p3, s3 = self.e3(p2)
        p4, s4 = self.e4(p3)
        b = self.b_pre(p4)       # (B, bottleneck_ch, H/16, W/16)
        b = self.b_mhsa(b)       # same shape, self-attention over H*W tokens
        b = self.b_post(b)       # (B, bottleneck_ch, H/16, W/16)
        x = self.d4(b, s4)       # (B, c4, H/8,  W/8)
        x = self.d3(x, s3)       # (B, c3, H/4,  W/4)
        x = self.d2(x, s2)       # (B, c2, H/2,  W/2)
        x = self.d1(x, s1)       # (B, c1, H,    W)
        return self.head(x)      # (B, num_classes, H, W)
