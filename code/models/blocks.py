"""Shared building blocks for all segmentation models.

All blocks take (B, C, H, W) tensors unless noted.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv2d -> BN -> ReLU. Preserves (H, W) when stride=1, padding=dilation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        padding: int = 1,
        dilation: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel,
            padding=padding, dilation=dilation, stride=stride, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, H, W) -> (B, out_ch, H, W) when stride=1
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Basic two-conv residual block with a 1x1 projection when channels differ."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=dilation,
            dilation=dilation, stride=stride, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=dilation,
            dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, H, W) -> (B, out_ch, H/stride, W/stride)
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + identity
        return F.relu(out, inplace=True)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (DeepLabV3+ style)."""

    def __init__(self, in_channels: int, out_channels: int, rates=(6, 12, 18)) -> None:
        super().__init__()
        # 1x1 branch
        self.b0 = ConvBNReLU(in_channels, out_channels, kernel=1, padding=0)
        # Three atrous 3x3 branches
        self.branches = nn.ModuleList([
            ConvBNReLU(in_channels, out_channels, kernel=3, padding=r, dilation=r)
            for r in rates
        ])
        # Global image pooling branch
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, kernel=1, padding=0),
        )
        # Project concatenated 5 branches back to out_channels
        self.project = nn.Sequential(
            ConvBNReLU(out_channels * (2 + len(rates)), out_channels, kernel=1, padding=0),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        h, w = x.shape[-2:]
        feats = [self.b0(x)]  # (B, out_channels, H, W)
        for br in self.branches:
            feats.append(br(x))  # (B, out_channels, H, W) each
        g = self.pool(x)  # (B, out_channels, 1, 1)
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        # g: (B, out_channels, H, W)
        feats.append(g)
        # concat along channels -> (B, out_channels*(2+len(rates)), H, W)
        y = torch.cat(feats, dim=1)
        return self.project(y)  # (B, out_channels, H, W)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018).

    g : gating signal from coarser scale, shape (B, g_ch, Hg, Wg)
    x : skip feature (to be attended), shape (B, x_ch, Hx, Wx)
    returns x * alpha, same shape as x.
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int) -> None:
        super().__init__()
        # 1x1 projection of gating signal: (B, g_ch, Hg, Wg) -> (B, inter_ch, Hg, Wg)
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # 1x1 projection of skip feature: (B, x_ch, Hx, Wx) -> (B, inter_ch, Hx, Wx)
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # 1x1 -> scalar attention map per spatial location
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Project gating: (B, g_ch, Hg, Wg) -> (B, inter_ch, Hg, Wg)
        g1 = self.W_g(g)
        # Project skip:   (B, x_ch, Hx, Wx) -> (B, inter_ch, Hx, Wx)
        x1 = self.W_x(x)
        # Upsample/downsample g1 to match x1's spatial size -> (B, inter_ch, Hx, Wx)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        # Sum + ReLU -> (B, inter_ch, Hx, Wx)
        psi = F.relu(g1 + x1, inplace=True)
        # -> (B, 1, Hx, Wx) attention map in [0,1]
        alpha = self.psi(psi)
        # Gate skip features -> (B, x_ch, Hx, Wx)
        return x * alpha


class MultiHeadSelfAttention(nn.Module):
    """2D MHSA. (B, C, H, W) -> (B, C, H, W) with a residual skip.

    Sequence length is H*W; keep spatial size small (e.g. bottleneck).
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if channels % num_heads != 0:
            # pick largest divisor of channels that is <= num_heads
            for h in range(num_heads, 0, -1):
                if channels % h == 0:
                    num_heads = h
                    break
        self.norm = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H, W) -> (B, C, H*W)
        y = x.flatten(2)
        # (B, C, H*W) -> (B, H*W, C)  tokens-last for nn.MultiheadAttention(batch_first=True)
        y = y.transpose(1, 2)
        # LayerNorm over channels: (B, H*W, C) -> (B, H*W, C)
        y_n = self.norm(y)
        # MHA: (B, H*W, C) -> (B, H*W, C)
        attn_out, _ = self.mha(y_n, y_n, y_n, need_weights=False)
        # Residual in token space: (B, H*W, C)
        y = y + attn_out
        # (B, H*W, C) -> (B, C, H*W)
        y = y.transpose(1, 2)
        # (B, C, H*W) -> (B, C, H, W)
        return y.reshape(b, c, h, w)
