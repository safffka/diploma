"""Generate synthetic mock data mimicking ISPRS Vaihingen/Potsdam and LoveDA structures.

No real datasets used. Produces RGB tiles 512x512 and RGB masks with proper colormaps.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/root/diploma")
MOCK = ROOT / "data" / "mock"
SIZE = 512
RNG_SEED = 42

ISPRS_COLORS = {
    0: (255, 255, 255),  # impervious_surface
    1: (0, 0, 255),      # building
    2: (0, 255, 255),    # low_vegetation
    3: (0, 255, 0),      # tree
    4: (255, 255, 0),    # car
    5: (255, 0, 0),      # background
}
ISPRS_NAMES = ["impervious_surface", "building", "low_vegetation", "tree", "car", "background"]

# LoveDA standard palette (commonly used)
LOVEDA_COLORS = {
    0: (255, 255, 255),  # background
    1: (255, 0, 0),      # building
    2: (255, 255, 0),    # road
    3: (0, 0, 255),      # water
    4: (159, 129, 183),  # barren
    5: (0, 255, 0),      # forest
    6: (255, 195, 128),  # agriculture
}
LOVEDA_NAMES = ["background", "building", "road", "water", "barren", "forest", "agriculture"]


def _draw_rect(arr: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    h, w = arr.shape[:2]
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        arr[y0:y1, x0:x1] = value


def _rgb_noise_for_class(rng: np.random.Generator, class_id: int, dataset: str, shape) -> np.ndarray:
    """Generate plausible RGB texture for given class."""
    h, w = shape
    if dataset == "isprs":
        base_palette = {
            0: (180, 180, 180),  # impervious / road grey
            1: (140, 90, 70),    # building roof brown/red
            2: (140, 200, 110),  # low veg light green
            3: (40, 110, 50),    # tree dark green
            4: (220, 220, 60),   # car yellow
            5: (90, 70, 50),     # background dirt
        }
    else:  # loveda
        base_palette = {
            0: (200, 200, 200),
            1: (160, 80, 60),
            2: (170, 170, 170),
            3: (60, 100, 180),
            4: (170, 140, 110),
            5: (40, 110, 50),
            6: (210, 190, 130),
        }
    base = np.array(base_palette[class_id], dtype=np.int16)
    noise = rng.integers(-25, 26, size=(h, w, 3), dtype=np.int16)
    out = np.clip(base[None, None, :] + noise, 0, 255).astype(np.uint8)
    return out


def _generate_pair(rng: np.random.Generator, dataset: str):
    """Return (image_rgb_uint8, mask_class_uint8) of size SIZExSIZE."""
    label = np.zeros((SIZE, SIZE), dtype=np.uint8)

    if dataset == "isprs":
        # Start filled with background (id 5) then add zones to hit ~ target distribution:
        # background 40%, building 20%, tree 20%, impervious 7%, low_veg 7%, car 6%
        label[:] = 5  # background everywhere
        # Add big tree patches (target 20%)
        for _ in range(rng.integers(3, 6)):
            cx, cy = rng.integers(0, SIZE, size=2)
            r = rng.integers(60, 130)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            label[mask] = 3
        # Low vegetation patches
        for _ in range(rng.integers(2, 5)):
            cx, cy = rng.integers(0, SIZE, size=2)
            r = rng.integers(30, 70)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            label[mask] = 2
        # Roads (impervious): a few horizontal/vertical strips
        for _ in range(rng.integers(2, 4)):
            if rng.random() < 0.5:
                y = rng.integers(0, SIZE - 20)
                _draw_rect(label, 0, y, SIZE, y + rng.integers(10, 22), 0)
            else:
                x = rng.integers(0, SIZE - 20)
                _draw_rect(label, x, 0, x + rng.integers(10, 22), SIZE, 0)
        # Buildings: rectangular blocks
        for _ in range(rng.integers(6, 12)):
            x0 = rng.integers(0, SIZE - 40)
            y0 = rng.integers(0, SIZE - 40)
            w = rng.integers(30, 90)
            h = rng.integers(30, 90)
            _draw_rect(label, x0, y0, x0 + w, y0 + h, 1)
        # Cars: small rectangles
        for _ in range(rng.integers(8, 16)):
            x0 = rng.integers(0, SIZE - 10)
            y0 = rng.integers(0, SIZE - 10)
            _draw_rect(label, x0, y0, x0 + rng.integers(4, 10), y0 + rng.integers(6, 14), 4)
        num_classes = 6
    else:  # loveda
        # Distribution-ish: background 25%, agriculture 25%, forest 20%, building 10%,
        # road 8%, water 7%, barren 5%
        label[:] = 0
        # agriculture big blobs
        for _ in range(rng.integers(2, 4)):
            cx, cy = rng.integers(0, SIZE, size=2)
            r = rng.integers(100, 180)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            label[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 6
        # forest
        for _ in range(rng.integers(2, 4)):
            cx, cy = rng.integers(0, SIZE, size=2)
            r = rng.integers(70, 140)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            label[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 5
        # barren
        for _ in range(rng.integers(1, 3)):
            cx, cy = rng.integers(0, SIZE, size=2)
            r = rng.integers(40, 80)
            yy, xx = np.ogrid[:SIZE, :SIZE]
            label[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 4
        # water
        for _ in range(rng.integers(1, 3)):
            x0 = rng.integers(0, SIZE - 60)
            y0 = rng.integers(0, SIZE - 30)
            _draw_rect(label, x0, y0, x0 + rng.integers(80, 200), y0 + rng.integers(20, 60), 3)
        # roads
        for _ in range(rng.integers(2, 4)):
            if rng.random() < 0.5:
                y = rng.integers(0, SIZE - 15)
                _draw_rect(label, 0, y, SIZE, y + rng.integers(8, 16), 2)
            else:
                x = rng.integers(0, SIZE - 15)
                _draw_rect(label, x, 0, x + rng.integers(8, 16), SIZE, 2)
        # buildings
        for _ in range(rng.integers(4, 9)):
            x0 = rng.integers(0, SIZE - 30)
            y0 = rng.integers(0, SIZE - 30)
            _draw_rect(label, x0, y0, x0 + rng.integers(20, 60), y0 + rng.integers(20, 60), 1)
        num_classes = 7

    # Build RGB image from label using class textures
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    for c in range(num_classes):
        mask = label == c
        if not mask.any():
            continue
        tex = _rgb_noise_for_class(rng, c, dataset, (SIZE, SIZE))
        img[mask] = tex[mask]

    return img, label


def _label_to_rgb(label: np.ndarray, dataset: str) -> np.ndarray:
    palette = ISPRS_COLORS if dataset == "isprs" else LOVEDA_COLORS
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for c, color in palette.items():
        rgb[label == c] = color
    return rgb


def generate_split(out_dir: Path, n: int, dataset: str, rng: np.random.Generator, prefix: str):
    img_dir = out_dir / "images"
    msk_dir = out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    counts = np.zeros(7 if dataset == "loveda" else 6, dtype=np.int64)
    for i in range(n):
        img, lbl = _generate_pair(rng, dataset)
        rgb_mask = _label_to_rgb(lbl, dataset)
        name = f"{prefix}_{i:03d}.png"
        Image.fromarray(img).save(img_dir / name)
        Image.fromarray(rgb_mask).save(msk_dir / name)
        for c in range(counts.shape[0]):
            counts[c] += int((lbl == c).sum())
    return counts


def main():
    rng = np.random.default_rng(RNG_SEED)
    summary = {}

    # ISPRS Vaihingen + Potsdam: same structure, train/val/test per spec
    for ds_name in ["vaihingen", "potsdam"]:
        ds_root = MOCK / ds_name
        total = np.zeros(6, dtype=np.int64)
        per_split = {}
        for split, n in [("train", 20), ("val", 5), ("test", 5)]:
            counts = generate_split(ds_root / split, n, "isprs", rng, f"{ds_name}_{split}")
            per_split[split] = int(n)
            total += counts
        # also flat directory expected by spec? spec lists data/mock/vaihingen/images/*.png
        # Provide flat copies for train as well via symlink? Instead create flat dirs by linking train split images.
        flat_img = ds_root / "images"
        flat_msk = ds_root / "masks"
        flat_img.mkdir(exist_ok=True)
        flat_msk.mkdir(exist_ok=True)
        # Copy/symlink all generated files into flat for spec compliance
        for split in ["train", "val", "test"]:
            for f in (ds_root / split / "images").iterdir():
                tgt = flat_img / f.name
                if not tgt.exists():
                    tgt.symlink_to(f)
            for f in (ds_root / split / "masks").iterdir():
                tgt = flat_msk / f.name
                if not tgt.exists():
                    tgt.symlink_to(f)
        dist = {ISPRS_NAMES[i]: round(float(total[i] / total.sum()), 4) for i in range(6)}
        summary[ds_name] = {"splits": per_split, "class_distribution": dist, "num_classes": 6,
                            "image_size": [SIZE, SIZE]}

    # LoveDA: train/urban + val/urban + test/urban
    loveda_total = np.zeros(7, dtype=np.int64)
    loveda_splits = {}
    for split, n in [("train", 20), ("val", 5), ("test", 5)]:
        counts = generate_split(MOCK / "loveda" / split / "urban", n, "loveda", rng,
                                f"loveda_{split}_urban")
        loveda_splits[split] = int(n)
        loveda_total += counts
    dist = {LOVEDA_NAMES[i]: round(float(loveda_total[i] / loveda_total.sum()), 4) for i in range(7)}
    summary["loveda"] = {"splits": loveda_splits, "class_distribution": dist, "num_classes": 7,
                         "image_size": [SIZE, SIZE], "domain": "urban"}

    # Save summary
    with open(MOCK / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Update status.json
    status_path = ROOT / "status.json"
    with open(status_path) as f:
        status = json.load(f)
    status["wave1"]["mock_data"] = "done"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    # Print report
    total_files = 0
    for ds in ["vaihingen", "potsdam"]:
        n = sum(summary[ds]["splits"].values())
        total_files += n * 2  # image + mask
    n = sum(summary["loveda"]["splits"].values())
    total_files += n * 2
    print(f"Total image+mask files generated: {total_files}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
