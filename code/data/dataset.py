"""PyTorch Dataset classes for ISPRS (Vaihingen/Potsdam) and LoveDA.

Masks on disk are single-channel uint8 class indices. An RGB -> index helper
(`rgb_to_mask`) is also provided for converting ISPRS-style RGB masks when
needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from code.data.augmentation import get_transforms, load_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ISPRS_CLASSES: List[str] = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "background",
]
ISPRS_NUM_CLASSES: int = 6

ISPRS_COLORMAP: Dict[tuple, int] = {
    (255, 255, 255): 0,  # impervious_surface
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low_vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # background
}

LOVEDA_CLASSES: List[str] = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]
LOVEDA_NUM_CLASSES: int = 7

IN_CHANNELS: int = 3

STATS_PATH = Path(__file__).resolve().parents[2] / "experiments" / "eda" / "dataset_stats.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rgb_to_mask(rgb: np.ndarray, colormap: Dict[tuple, int] = ISPRS_COLORMAP) -> np.ndarray:
    """Convert an (H,W,3) uint8 RGB mask to an (H,W) int64 class index mask."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) rgb mask, got shape {rgb.shape}")
    h, w, _ = rgb.shape
    out = np.zeros((h, w), dtype=np.int64)
    for color, idx in colormap.items():
        match = (rgb[..., 0] == color[0]) & (rgb[..., 1] == color[1]) & (rgb[..., 2] == color[2])
        out[match] = idx
    return out


def _list_files(dir_path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in exts])


def _load_stats_for(dataset_name: str) -> Dict[str, Any]:
    all_stats = load_stats(STATS_PATH)
    if dataset_name not in all_stats:
        raise KeyError(f"dataset '{dataset_name}' not found in {STATS_PATH}")
    return all_stats[dataset_name]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _BaseSegDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        dataset_name: str,
        transform: Optional[Any] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.dataset_name = dataset_name

        self.image_dir = self.root_dir / dataset_name / split / "images"
        self.mask_dir = self.root_dir / dataset_name / split / "masks"
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"missing image dir: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"missing mask dir: {self.mask_dir}")

        self.images = _list_files(self.image_dir)
        self.masks = _list_files(self.mask_dir)
        if len(self.images) != len(self.masks):
            raise RuntimeError(
                f"image/mask count mismatch in {dataset_name}/{split}: "
                f"{len(self.images)} vs {len(self.masks)}"
            )

        if transform is None:
            stats = _load_stats_for(dataset_name)
            transform = get_transforms(stats, train=(split == "train"))
        self.transform = transform

        self._stats = _load_stats_for(dataset_name)

    def __len__(self) -> int:
        return len(self.images)

    def _load_pair(self, idx: int):
        img = np.array(Image.open(self.images[idx]).convert("RGB"), dtype=np.uint8)
        mask_img = Image.open(self.masks[idx])
        mask_arr = np.array(mask_img)
        if mask_arr.ndim == 3:
            # RGB mask -> indices
            mask_arr = rgb_to_mask(mask_arr)
        mask_arr = mask_arr.astype(np.int64)
        return img, mask_arr

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img, mask = self._load_pair(idx)
        out = self.transform(image=img, mask=mask)
        image_t = out["image"].float()
        mask_t = out["mask"].long()
        return {"image": image_t, "mask": mask_t}

    def compute_class_weights(self) -> torch.Tensor:
        return torch.tensor(self._stats["class_weights"], dtype=torch.float32)


class ISPRSDataset(_BaseSegDataset):
    """ISPRS Vaihingen/Potsdam dataset."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        dataset_name: str = "vaihingen",
        transform: Optional[Any] = None,
    ) -> None:
        if dataset_name not in ("vaihingen", "potsdam"):
            raise ValueError(f"ISPRSDataset expects 'vaihingen' or 'potsdam', got {dataset_name}")
        super().__init__(root_dir, split, dataset_name, transform)


class LoveDADataset(_BaseSegDataset):
    """LoveDA dataset (urban/rural domain)."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        domain: str = "urban",
        transform: Optional[Any] = None,
    ) -> None:
        self.domain = domain
        super().__init__(root_dir, split, "loveda", transform)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Return a dict with num_classes, in_channels, class_names, class_weights,
    pixel_mean, pixel_std — read from experiments/eda/dataset_stats.json."""
    s = _load_stats_for(dataset_name)
    return {
        "num_classes": s["num_classes"],
        "in_channels": s["in_channels"],
        "class_names": s["class_names"],
        "class_weights": s["class_weights"],
        "pixel_mean": s["pixel_mean"],
        "pixel_std": s["pixel_std"],
    }


def get_dataloader(
    dataset_name: str,
    root_dir: str | Path,
    split: str,
    batch_size: int,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader for the requested dataset/split."""
    if dataset_name in ("vaihingen", "potsdam"):
        ds: Dataset = ISPRSDataset(root_dir, split, dataset_name=dataset_name)
    elif dataset_name == "loveda":
        ds = LoveDADataset(root_dir, split)
    else:
        raise ValueError(f"unknown dataset_name: {dataset_name}")

    shuffle = split == "train"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
