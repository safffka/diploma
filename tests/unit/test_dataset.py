"""Tests for code.data.dataset constants and helpers."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from code.data.dataset import (
    IN_CHANNELS,
    ISPRS_CLASSES,
    ISPRS_COLORMAP,
    ISPRS_NUM_CLASSES,
    ISPRSDataset,
    LOVEDA_CLASSES,
    LOVEDA_NUM_CLASSES,
    get_dataset_info,
    rgb_to_mask,
)


def test_constants_defined():
    assert ISPRS_NUM_CLASSES == len(ISPRS_CLASSES)
    assert LOVEDA_NUM_CLASSES == len(LOVEDA_CLASSES)
    assert IN_CHANNELS == 3
    assert ISPRS_NUM_CLASSES > 0


def test_colormap_has_all_classes():
    assert len(ISPRS_COLORMAP) == ISPRS_NUM_CLASSES
    indices = sorted(ISPRS_COLORMAP.values())
    assert indices == list(range(ISPRS_NUM_CLASSES))


def test_isprs_dataset_len(tmp_path):
    """Build a tiny on-disk synthetic ISPRS-shaped dataset and verify length."""
    from PIL import Image

    root = tmp_path
    name = "vaihingen"
    img_dir = root / name / "train" / "images"
    msk_dir = root / name / "train" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    n = 3
    for i in range(n):
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            img_dir / f"{i:02d}.png"
        )
        Image.fromarray(
            np.random.randint(0, ISPRS_NUM_CLASSES, (64, 64), dtype=np.uint8)
        ).save(msk_dir / f"{i:02d}.png")

    ds = ISPRSDataset(root, "train", dataset_name=name)
    assert len(ds) == n


def test_isprs_getitem_shapes(tmp_path):
    from PIL import Image

    root = tmp_path
    name = "vaihingen"
    img_dir = root / name / "val" / "images"
    msk_dir = root / name / "val" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        img_dir / "0.png"
    )
    Image.fromarray(
        np.random.randint(0, ISPRS_NUM_CLASSES, (64, 64), dtype=np.uint8)
    ).save(msk_dir / "0.png")

    ds = ISPRSDataset(root, "val", dataset_name=name)
    item = ds[0]
    assert "image" in item and "mask" in item
    assert item["image"].shape == (IN_CHANNELS, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert item["mask"].dtype == torch.long


def test_mask_values_in_range(tmp_path):
    from PIL import Image

    root = tmp_path
    name = "vaihingen"
    img_dir = root / name / "val" / "images"
    msk_dir = root / name / "val" / "masks"
    img_dir.mkdir(parents=True)
    msk_dir.mkdir(parents=True)
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(
        img_dir / "0.png"
    )
    Image.fromarray(
        np.random.randint(0, ISPRS_NUM_CLASSES, (64, 64), dtype=np.uint8)
    ).save(msk_dir / "0.png")

    ds = ISPRSDataset(root, "val", dataset_name=name)
    m = ds[0]["mask"]
    assert int(m.min()) >= 0
    assert int(m.max()) < ISPRS_NUM_CLASSES


def test_class_weights_shape_and_positive():
    info = get_dataset_info("vaihingen")
    weights = torch.tensor(info["class_weights"], dtype=torch.float32)
    assert weights.shape == (ISPRS_NUM_CLASSES,)
    assert (weights > 0).all()


def test_get_dataset_info_keys():
    info = get_dataset_info("vaihingen")
    for k in (
        "num_classes",
        "in_channels",
        "class_names",
        "class_weights",
        "pixel_mean",
        "pixel_std",
    ):
        assert k in info
    assert info["num_classes"] == ISPRS_NUM_CLASSES
    assert info["in_channels"] == IN_CHANNELS


def test_rgb_to_mask_roundtrip():
    h, w = 8, 8
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # paint each row with a different class colour
    colours = list(ISPRS_COLORMAP.keys())
    for i in range(min(h, len(colours))):
        rgb[i, :, :] = np.array(colours[i], dtype=np.uint8)
    out = rgb_to_mask(rgb)
    for i in range(min(h, len(colours))):
        expected = ISPRS_COLORMAP[colours[i]]
        assert (out[i, :] == expected).all()
