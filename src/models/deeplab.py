"""DeepLabV3+ with a lightweight ResNet-style encoder and ASPP."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from src.models.blocks import ASPP, ConvBNReLU, ResidualBlock


def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    layers = [ResidualBlock(in_ch, out_ch, stride=stride, dilation=dilation)]
    for _ in range(blocks - 1):
        layers.append(ResidualBlock(out_ch, out_ch, stride=1, dilation=dilation))
    return nn.Sequential(*layers)


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes: int = ISPRS_NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        aspp_out: int = 128,
        low_level_out: int = 48,
    ) -> None:
        super().__init__()
        # Stem -> (B, 32, H/2, W/2)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, kernel=3, padding=1, stride=2),
            ConvBNReLU(32, 32),
        )
        # Layer1: (B,32,H/2,W/2) -> (B,64,H/4,W/4)   (low-level feats here)
        self.layer1 = _make_layer(32, 64, blocks=2, stride=2)
        # Layer2: -> (B,128,H/8,W/8)
        self.layer2 = _make_layer(64, 128, blocks=2, stride=2)
        # Layer3: atrous, stride=1 dilation=2 -> keep H/8,W/8
        self.layer3 = _make_layer(128, 256, blocks=2, stride=1, dilation=2)
        # Layer4: atrous, stride=1 dilation=4 -> keep H/8,W/8
        self.layer4 = _make_layer(256, 256, blocks=2, stride=1, dilation=4)

        # ASPP on deepest features -> (B, aspp_out, H/8, W/8)
        self.aspp = ASPP(256, aspp_out, rates=[6, 12, 18])

        # Low-level projection (from layer1) -> (B, low_level_out, H/4, W/4)
        self.low_proj = ConvBNReLU(64, low_level_out, kernel=1, padding=0)

        # Decoder: concat (upsampled ASPP, low-level) -> ConvBNReLU x2 -> classifier
        self.decoder = nn.Sequential(
            ConvBNReLU(aspp_out + low_level_out, aspp_out),
            ConvBNReLU(aspp_out, aspp_out),
        )
        self.classifier = nn.Conv2d(aspp_out, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # (B, in_channels, H, W) -> (B, 32, H/2, W/2)
        x = self.stem(x)
        low = self.layer1(x)          # (B, 64,  H/4, W/4)
        x = self.layer2(low)          # (B, 128, H/8, W/8)
        x = self.layer3(x)            # (B, 256, H/8, W/8)  dilation=2
        x = self.layer4(x)            # (B, 256, H/8, W/8)  dilation=4
        x = self.aspp(x)              # (B, aspp_out, H/8, W/8)
        # Upsample ASPP to low-level resolution -> (B, aspp_out, H/4, W/4)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low = self.low_proj(low)      # (B, low_level_out, H/4, W/4)
        # concat -> (B, aspp_out + low_level_out, H/4, W/4)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)           # (B, aspp_out, H/4, W/4)
        x = self.classifier(x)        # (B, num_classes, H/4, W/4)
        # Final upsample -> (B, num_classes, H, W)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
