"""Model factory."""
from __future__ import annotations

import torch.nn as nn

from src.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from src.models.attention_unet import AttentionUNet
from src.models.deeplab import DeepLabV3Plus
from src.models.fcn import FCN
from src.models.segformer import SegFormer
from src.models.unet import UNet

__all__ = [
    "get_model",
    "FCN",
    "UNet",
    "DeepLabV3Plus",
    "AttentionUNet",
    "SegFormer",
]


def get_model(
    name: str,
    num_classes: int = ISPRS_NUM_CLASSES,
    in_channels: int = IN_CHANNELS,
) -> nn.Module:
    """Return one of the five segmentation models.

    name: "fcn" | "unet" | "deeplab" | "attention" | "segformer"
    """
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
    raise ValueError(
        f"unknown model name '{name}'. Expected one of: "
        "'fcn', 'unet', 'deeplab', 'attention', 'segformer'."
    )

from src.models.unet_pretrained import UNetPretrained
from src.models.attention_unet_pretrained import AttentionUNetPretrained

_PRETRAINED_REGISTRY = {
    "unet_pt":      lambda nc, ic: UNetPretrained(num_classes=nc, pretrained=True),
    "attention_pt": lambda nc, ic: AttentionUNetPretrained(num_classes=nc, pretrained=True),
}

_ORIGINAL_GET_MODEL = get_model

def get_model(name: str, num_classes: int = ISPRS_NUM_CLASSES,
              in_channels: int = IN_CHANNELS) -> nn.Module:
    if name in _PRETRAINED_REGISTRY:
        return _PRETRAINED_REGISTRY[name](num_classes, in_channels)
    return _ORIGINAL_GET_MODEL(name, num_classes, in_channels)
