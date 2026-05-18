"""
GCS-SegFormer: реализация на основе Lu et al., ADMA 2025.
Три улучшения над SegFormer: GCSA + SDI + Ghost Decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.segformer import MiTStage


# ─── GCSA ───────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg(x)) + self.fc(self.mx(x)))


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class GCSA(nn.Module):
    """Global Channel Spatial Attention: CA → Shuffle → SA"""
    def __init__(self, channels, groups=8):
        super().__init__()
        self.groups = min(groups, channels)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = channel_shuffle(x, self.groups)
        x = self.sa(x)
        return x


# ─── SDI ────────────────────────────────────────────────────

class SDI(nn.Module):
    """Semantics and Detail Infusion"""
    def __init__(self, deep_ch, shallow_ch, out_ch):
        super().__init__()
        self.proj_deep    = nn.Conv2d(deep_ch,    out_ch, 1, bias=False)
        self.proj_shallow = nn.Conv2d(shallow_ch, out_ch, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.GroupNorm(1, out_ch)

    def forward(self, deep, shallow):
        d = F.interpolate(self.proj_deep(deep),
                          size=shallow.shape[-2:], mode='bilinear', align_corners=False)
        s = self.proj_shallow(shallow) * self.gate(d)
        return self.norm(self.fuse(torch.cat([d, s], dim=1)))


# ─── Ghost Conv ─────────────────────────────────────────────

class GhostConv(nn.Module):
    """Ghost Convolution (Han et al., CVPR 2020)"""
    def __init__(self, in_ch, out_ch, kernel_size=1, ratio=2, dw_size=3):
        super().__init__()
        init_ch  = (out_ch + 1) // ratio
        cheap_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, dw_size, padding=dw_size//2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.primary(x)
        return torch.cat([y, self.cheap(y)], dim=1)


class GhostDecoder(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_classes=6):
        super().__init__()
        self.proj  = nn.ModuleList([GhostConv(c, embed_dim) for c in in_channels])
        self.fuse  = GhostConv(embed_dim * len(in_channels), embed_dim)
        self.norm  = nn.BatchNorm2d(embed_dim)
        self.head  = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        target = features[0].shape[-2:]
        projected = []
        for f, proj in zip(features, self.proj):
            p = proj(f)
            if p.shape[-2:] != target:
                p = F.interpolate(p, size=target, mode='bilinear', align_corners=False)
            projected.append(p)
        out = self.norm(self.fuse(torch.cat(projected, dim=1)))
        return self.head(out)


# ─── GCS-SegFormer ──────────────────────────────────────────

class GCSSegFormer(nn.Module):
    """
    GCS-SegFormer: SegFormer + GCSA + SDI + Ghost Decoder.
    Lu et al., ADMA 2025. mIoU 93.25% on ISPRS Vaihingen.
    """
    def __init__(self, num_classes=6, in_channels=3,
                 embed_dims=None, depths=None, sr_ratios=None,
                 embed_dim=128):
        super().__init__()
        if embed_dims is None: embed_dims = [32, 64, 128, 192]
        if depths     is None: depths     = [2, 2, 2, 2]
        if sr_ratios  is None: sr_ratios  = [8, 4, 2, 1]

        # Encoder: 4 стадии MiT
        self.stages = nn.ModuleList()
        in_ch = in_channels
        strides = [4, 2, 2, 2]
        kernels = [7, 3, 3, 3]
        for ed, d, sr, stride, ks in zip(embed_dims, depths, sr_ratios, strides, kernels):
            self.stages.append(MiTStage(in_ch, ed,
                                        patch_kernel=ks, patch_stride=stride,
                                        patch_pad=ks//2, depth=d,
                                        num_heads=max(1, ed//32),
                                        sr_ratio=sr))
            in_ch = ed

        # GCSA на стадиях 2 и 3 (глубокие признаки)
        self.gcsa2 = GCSA(embed_dims[2])
        self.gcsa3 = GCSA(embed_dims[3])

        # SDI: stage3→stage0, stage2→stage1
        self.sdi_coarse = SDI(embed_dims[3], embed_dims[0], embed_dims[0])
        self.sdi_fine   = SDI(embed_dims[2], embed_dims[1], embed_dims[1])

        # Ghost Decoder
        self.decoder = GhostDecoder(embed_dims, embed_dim=embed_dim,
                                    num_classes=num_classes)

    def forward(self, x):
        H_in, W_in = x.shape[-2:]
        f = []
        for stage in self.stages:
            x = stage(x)
            f.append(x)

        # GCSA
        f[2] = self.gcsa2(f[2])
        f[3] = self.gcsa3(f[3])

        # SDI
        f[0] = self.sdi_coarse(f[3], f[0])
        f[1] = self.sdi_fine(f[2], f[1])

        out = self.decoder(f)
        return F.interpolate(out, size=(H_in, W_in),
                             mode='bilinear', align_corners=False)
