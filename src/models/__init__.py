"""Model factory."""
from __future__ import annotations

import torch.nn as nn

from src.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from src.models.attention_unet import AttentionUNet
from src.models.attention_unet_pretrained import AttentionUNetPretrained
from src.models.deeplab import DeepLabV3Plus
from src.models.fcn import FCN
from src.models.segformer import SegFormer
from src.models.unet import UNet
from src.models.unet_pretrained import UNetPretrained

__all__ = [
    "get_model",
    "FCN", "UNet", "DeepLabV3Plus",
    "AttentionUNet", "SegFormer",
    "UNetPretrained", "AttentionUNetPretrained",
]


def get_model(
    name: str,
    num_classes: int = ISPRS_NUM_CLASSES,
    in_channels: int = IN_CHANNELS,
) -> nn.Module:
    key = name.lower().strip()
    if key == "fcn":
        return FCN(num_classes=num_classes, in_channels=in_channels)
    if key == "unet":
        return UNet(num_classes=num_classes, in_channels=in_channels)
    if key == "deeplab":
        return DeepLabV3Plus(num_classes=num_classes, in_channels=in_channels)
    if key == "attention":
        return AttentionUNet(num_classes=num_classes, in_channels=in_channels)
    if key == "segformer":
        return SegFormer(num_classes=num_classes, in_channels=in_channels)
    if key == "unet_pt":
        return UNetPretrained(num_classes=num_classes, pretrained=True)
    if key == "attention_pt":
        return AttentionUNetPretrained(num_classes=num_classes, pretrained=True)
    raise ValueError(
        f"Unknown model '{name}'. Choose from: "
        "fcn, unet, deeplab, attention, segformer, unet_pt, attention_pt"
    )
