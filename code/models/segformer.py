"""Simplified SegFormer: Mix Transformer (MiT) encoder + MLP decoder.

Four stages with overlapping patch embeddings and efficient self-attention
(spatial reduction). Lightweight so forward+backward runs on CPU at 2x3x64x64.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES


class OverlapPatchEmbed(nn.Module):
    """Conv2d-based overlapping patch embedding with LayerNorm on channels."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int, padding: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, H, W) -> (B, out_ch, H', W')
        x = self.proj(x)
        b, c, h, w = x.shape
        # (B, C, H', W') -> (B, H'*W', C) for LayerNorm over C
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # Back to 4D -> (B, C, H', W')
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class EfficientSelfAttention(nn.Module):
    """Self-attention with spatial reduction (SegFormer MiT) on 4D feature maps."""

    def __init__(self, channels: int, num_heads: int, sr_ratio: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        if channels % num_heads != 0:
            for h in range(num_heads, 0, -1):
                if channels % h == 0:
                    num_heads = h
                    break
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.q = nn.Linear(channels, channels, bias=True)
        self.kv = nn.Linear(channels, channels * 2, bias=True)
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(channels, channels, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(channels)
        else:
            self.sr = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        # (B, C, H, W) -> (B, N, C) token sequence
        x_seq = x.flatten(2).transpose(1, 2)

        # Q: (B, N, C) -> (B, num_heads, N, C/num_heads)
        q = self.q(x_seq).reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Spatially reduce keys/values: (B, C, H, W) -> (B, C, H/sr, W/sr)
            kv_in = self.sr(x)
            h2, w2 = kv_in.shape[-2:]
            # (B, C, H/sr, W/sr) -> (B, N', C)
            kv_in = kv_in.flatten(2).transpose(1, 2)
            kv_in = self.sr_norm(kv_in)
            n2 = h2 * w2
        else:
            kv_in = x_seq
            n2 = n

        # kv: (B, N', 2C) -> split -> each (B, num_heads, N', C/num_heads)
        kv = self.kv(kv_in).reshape(b, n2, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # attn: (B, num_heads, N, N')
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # (B, num_heads, N, C/num_heads) -> (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        # (B, N, C) -> (B, C, H, W)
        return out.transpose(1, 2).reshape(b, c, h, w)


class MixFFN(nn.Module):
    """SegFormer Mix-FFN: Linear -> 3x3 DWConv -> GELU -> Linear, operating on (B,C,H,W)."""

    def __init__(self, channels: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = channels * mlp_ratio
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, hidden, H, W) -> (B, C, H, W)
        return self.fc2(self.act(self.dw(self.fc1(x))))


class MiTBlock(nn.Module):
    """One transformer block: LN -> EfficientAttn -> residual; LN -> MixFFN -> residual."""

    def __init__(self, channels: int, num_heads: int, sr_ratio: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)  # LN over C on 4D feature map
        self.attn = EfficientSelfAttention(channels, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = MixFFN(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> same shape after each residual stage
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiTStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        patch_kernel: int,
        patch_stride: int,
        patch_pad: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(in_ch, out_ch, patch_kernel, patch_stride, patch_pad)
        self.blocks = nn.ModuleList([
            MiTBlock(out_ch, num_heads=num_heads, sr_ratio=sr_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.GroupNorm(1, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, H, W) -> (B, out_ch, H', W')
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class SegFormer(nn.Module):
    def __init__(
        self,
        num_classes: int = ISPRS_NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        embed_dims: List[int] = (32, 64, 128, 192),
        depths: List[int] = (2, 2, 2, 2),
        num_heads: List[int] = (1, 2, 4, 8),
        sr_ratios: List[int] = (8, 4, 2, 1),
        decoder_ch: int = 128,
    ) -> None:
        super().__init__()
        e = list(embed_dims)

        # Stage 1: 7x7 stride 4 -> (B, e0, H/4,  W/4)
        self.stage1 = MiTStage(in_channels, e[0], 7, 4, 3, depths[0], num_heads[0], sr_ratios[0])
        # Stage 2: 3x3 stride 2 -> (B, e1, H/8,  W/8)
        self.stage2 = MiTStage(e[0], e[1], 3, 2, 1, depths[1], num_heads[1], sr_ratios[1])
        # Stage 3: 3x3 stride 2 -> (B, e2, H/16, W/16)
        self.stage3 = MiTStage(e[1], e[2], 3, 2, 1, depths[2], num_heads[2], sr_ratios[2])
        # Stage 4: 3x3 stride 2 -> (B, e3, H/32, W/32)
        self.stage4 = MiTStage(e[2], e[3], 3, 2, 1, depths[3], num_heads[3], sr_ratios[3])

        # MLP decoder: 1x1 conv per stage -> decoder_ch, upsample to 1/4, concat, fuse
        self.lin1 = nn.Conv2d(e[0], decoder_ch, kernel_size=1)
        self.lin2 = nn.Conv2d(e[1], decoder_ch, kernel_size=1)
        self.lin3 = nn.Conv2d(e[2], decoder_ch, kernel_size=1)
        self.lin4 = nn.Conv2d(e[3], decoder_ch, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_ch * 4, decoder_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(decoder_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, in_channels, H, W)
        h, w = x.shape[-2:]
        f1 = self.stage1(x)   # (B, e0, H/4,  W/4)
        f2 = self.stage2(f1)  # (B, e1, H/8,  W/8)
        f3 = self.stage3(f2)  # (B, e2, H/16, W/16)
        f4 = self.stage4(f3)  # (B, e3, H/32, W/32)

        target_size = f1.shape[-2:]  # 1/4 resolution
        # Project each to decoder_ch, then upsample to 1/4 -> (B, decoder_ch, H/4, W/4)
        y1 = self.lin1(f1)
        y2 = F.interpolate(self.lin2(f2), size=target_size, mode="bilinear", align_corners=False)
        y3 = F.interpolate(self.lin3(f3), size=target_size, mode="bilinear", align_corners=False)
        y4 = F.interpolate(self.lin4(f4), size=target_size, mode="bilinear", align_corners=False)

        # Concat along channels -> (B, decoder_ch*4, H/4, W/4)
        y = torch.cat([y4, y3, y2, y1], dim=1)
        y = self.fuse(y)                # (B, decoder_ch, H/4, W/4)
        y = self.classifier(y)          # (B, num_classes, H/4, W/4)
        # Upsample to input resolution -> (B, num_classes, H, W)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
