"""Preprocessor: convert mock raw tiles to processed train-ready tiles.

- Source: data/mock/{vaihingen,potsdam,loveda}
- Output: data/processed/{dataset}/{split}/{images,masks}/*.png
- Masks are converted from RGB palette (ISPRS) or grayscale (LoveDA) into
  single-channel class indices [0..num_classes-1].
- Tiles where >90% pixels belong to background class are filtered out.
- Updates experiments/eda/dataset_stats.json with actual tile counts.

Note: mock tiles are already 512x512 and pre-split, so no tiling/splitting
is needed at this stage. The pipeline will be reused for real data later.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/diploma")
MOCK = ROOT / "data" / "mock"
OUT = ROOT / "data" / "processed"
STATS_PATH = ROOT / "experiments" / "eda" / "dataset_stats.json"

# ISPRS RGB palette -> class index (from context.md)
ISPRS_PALETTE = {
    (255, 255, 255): 0,  # impervious_surface
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low_vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # background
}
ISPRS_BACKGROUND = 5

# LoveDA mock encodes 7 classes via distinct grayscale values; remap by sort order.
LOVEDA_BACKGROUND = 0

BG_FILTER_RATIO = 0.90
SPLITS = ("train", "val", "test")


def rgb_mask_to_index(arr: np.ndarray, palette: dict) -> np.ndarray:
    h, w, _ = arr.shape
    out = np.zeros((h, w), dtype=np.uint8)
    flat = arr.reshape(-1, 3)
    out_flat = out.reshape(-1)
    matched = np.zeros(flat.shape[0], dtype=bool)
    for rgb, idx in palette.items():
        m = np.all(flat == np.array(rgb, dtype=arr.dtype), axis=1)
        out_flat[m] = idx
        matched |= m
    # Unknown pixels -> background
    out_flat[~matched] = ISPRS_BACKGROUND
    return out


def loveda_mask_to_index(arr: np.ndarray, num_classes: int = 7) -> np.ndarray:
    if arr.ndim == 3:
        # mock stored as RGB but values identical across channels
        arr = arr[..., 0]
    uniq = sorted(np.unique(arr).tolist())
    # Map sorted unique grayscale levels to consecutive class ids
    mapping = {v: i for i, v in enumerate(uniq[:num_classes])}
    out = np.zeros_like(arr, dtype=np.uint8)
    for v, idx in mapping.items():
        out[arr == v] = idx
    # Any leftover (extra unique > num_classes) -> background 0
    return out


def collect_pairs(dataset: str, split: str):
    """Return list of (image_path, mask_path) for given dataset/split."""
    base = MOCK / dataset / split
    pairs = []
    if dataset == "loveda":
        # nested by domain (urban/, rural/)
        for domain_dir in sorted(base.iterdir()):
            if not domain_dir.is_dir():
                continue
            img_dir = domain_dir / "images"
            mask_dir = domain_dir / "masks"
            for img in sorted(img_dir.glob("*.png")):
                mask = mask_dir / img.name
                if mask.exists():
                    pairs.append((img, mask))
    else:
        img_dir = base / "images"
        mask_dir = base / "masks"
        for img in sorted(img_dir.glob("*.png")):
            mask = mask_dir / img.name
            if mask.exists():
                pairs.append((img, mask))
    return pairs


def process_dataset(dataset: str, num_classes: int):
    counts = {"train": 0, "val": 0, "test": 0, "filtered": 0}
    background = ISPRS_BACKGROUND if dataset != "loveda" else LOVEDA_BACKGROUND

    for split in SPLITS:
        pairs = collect_pairs(dataset, split)
        out_img = OUT / dataset / split / "images"
        out_mask = OUT / dataset / split / "masks"
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            img = np.array(Image.open(img_path).convert("RGB"))
            mask_raw = np.array(Image.open(mask_path))

            if dataset == "loveda":
                mask_idx = loveda_mask_to_index(mask_raw, num_classes)
            else:
                if mask_raw.ndim == 2:
                    mask_raw = np.stack([mask_raw] * 3, axis=-1)
                mask_idx = rgb_mask_to_index(mask_raw, ISPRS_PALETTE)

            # Verify class range
            assert mask_idx.min() >= 0 and mask_idx.max() < num_classes, (
                f"{mask_path}: classes out of range "
                f"[{mask_idx.min()}, {mask_idx.max()}]"
            )

            # Filter >90% background
            bg_ratio = float((mask_idx == background).mean())
            if bg_ratio > BG_FILTER_RATIO:
                counts["filtered"] += 1
                continue

            Image.fromarray(img).save(out_img / img_path.name)
            Image.fromarray(mask_idx).save(out_mask / img_path.name)
            counts[split] += 1

    return counts


def main():
    stats = json.loads(STATS_PATH.read_text())
    totals = {"filtered": 0}
    for ds in ("vaihingen", "potsdam", "loveda"):
        nc = stats[ds]["num_classes"]
        c = process_dataset(ds, nc)
        stats[ds]["actual_train_tiles"] = c["train"]
        stats[ds]["actual_val_tiles"] = c["val"]
        stats[ds]["actual_test_tiles"] = c["test"]
        stats[ds]["filtered_tiles"] = c["filtered"]
        totals[ds] = c
        totals["filtered"] += c["filtered"]
        print(f"{ds}: {c}")

    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Updated {STATS_PATH}")


if __name__ == "__main__":
    main()
