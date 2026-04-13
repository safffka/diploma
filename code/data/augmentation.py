"""Augmentation pipelines for semantic segmentation.

Geometric augmentations (flip, rotate) are applied synchronously to image and
mask. Color augmentations (jitter, noise) are applied only to the image.
albumentations handles this automatically when image and mask are passed
together.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_stats(stats_path: str | Path) -> Dict[str, Any]:
    """Load dataset statistics JSON (per-dataset dict)."""
    with open(stats_path, "r") as f:
        return json.load(f)


def get_transforms(stats: Dict[str, Any], train: bool = True) -> A.Compose:
    """Build an albumentations Compose pipeline.

    Args:
        stats: dict containing 'pixel_mean' and 'pixel_std' lists.
        train: if True, include geometric + color augmentations.
    """
    mean = stats["pixel_mean"]
    std = stats["pixel_std"]

    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_train_transforms(stats: Dict[str, Any]) -> A.Compose:
    return get_transforms(stats, train=True)


def get_val_transforms(stats: Dict[str, Any]) -> A.Compose:
    return get_transforms(stats, train=False)
