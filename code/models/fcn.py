"""FCN baseline: VGG-style encoder + bilinear upsampling decoder."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from code.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from code.models.blocks import ConvBNReLU


def _vgg_block(in_ch: int, out_ch: int, n_convs: int = 2) -> nn.Sequential:
    layers = [ConvBNReLU(in_ch, out_ch)]
    for _ in range(n_convs - 1):
        layers.append(ConvBNReLU(out_ch, out_ch))
    return nn.Sequential(*layers)


class FCN(nn.Module):
    def __init__(
        self,
        num_classes: int = ISPRS_NUM_CLASSES,
        in_channels: int = IN_CHANNELS,
        base: int = 32,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 8

        # 5 VGG-style stages, each followed by MaxPool(2)
        self.block1 = _vgg_block(in_channels, c1, 2)   # (B,in,H,W)    -> (B,c1,H,W)
        self.block2 = _vgg_block(c1, c2, 2)            # -> (B,c2,H/2,W/2)  after pool
        self.block3 = _vgg_block(c2, c3, 3)            # -> (B,c3,H/4,W/4)
        self.block4 = _vgg_block(c3, c4, 3)            # -> (B,c4,H/8,W/8)
        self.block5 = _vgg_block(c4, c5, 3)            # -> (B,c5,H/16,W/16)
        self.pool = nn.MaxPool2d(2, 2)

        # 1x1 classifier on the deepest feature map
        self.classifier = nn.Conv2d(c5, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, H, W)
        h, w = x.shape[-2:]
        x = self.block1(x)                 # (B, c1,  H,    W)
        x = self.pool(self.block2(x))      # (B, c2,  H/2,  W/2)
        x = self.pool(self.block3(x))      # (B, c3,  H/4,  W/4)
        x = self.pool(self.block4(x))      # (B, c4,  H/8,  W/8)
        x = self.pool(self.block5(x))      # (B, c5,  H/16, W/16)
        x = self.classifier(x)             # (B, num_classes, H/16, W/16)
        # Upsample back to input resolution -> (B, num_classes, H, W)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
