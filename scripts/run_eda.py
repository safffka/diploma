"""EDA agent: compute dataset statistics from train splits only.

Outputs experiments/eda/dataset_stats.json consumed by all downstream agents.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "data" / "mock"
OUT = ROOT / "experiments" / "eda" / "dataset_stats.json"
STATUS = ROOT / "status.json"

ISPRS_CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "background",
]
ISPRS_PALETTE = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
}
LOVEDA_PALETTE = {
    (255, 255, 255): 0,  # background
    (255, 0, 0): 1,      # building
    (255, 255, 0): 2,    # road
    (0, 0, 255): 3,      # water
    (159, 129, 183): 4,  # barren
    (0, 255, 0): 5,      # forest
    (255, 195, 128): 6,  # agriculture
}
LOVEDA_CLASS_NAMES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]

TILE_SIZE = 512


def list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])


def rgb_to_labels(mask_rgb: np.ndarray, palette: dict) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.full((h, w), 255, dtype=np.uint8)
    for rgb, idx in palette.items():
        m = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        out[m] = idx
    return out


def compute_split(images_dir: Path, masks_dir: Path, num_classes: int, mask_kind: str):
    """Return (pixel_mean, pixel_std, class_counts, n_images, total_pixels)."""
    img_paths = list_images(images_dir)
    mask_paths = list_images(masks_dir)
    assert len(img_paths) == len(mask_paths), f"image/mask count mismatch in {images_dir}"

    # Welford-like accumulators per channel
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for ip, mp in zip(img_paths, mask_paths):
        img = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = img.shape
        sum_c += img.reshape(-1, 3).sum(axis=0)
        sumsq_c += (img.reshape(-1, 3) ** 2).sum(axis=0)
        n_pixels += h * w

        mask_img = Image.open(mp)
        if mask_kind == "isprs":
            labels = rgb_to_labels(np.asarray(mask_img.convert("RGB"), dtype=np.uint8), ISPRS_PALETTE)
        elif mask_kind == "loveda":
            labels = rgb_to_labels(np.asarray(mask_img.convert("RGB"), dtype=np.uint8), LOVEDA_PALETTE)
        else:
            labels = np.asarray(mask_img, dtype=np.int64)
            if labels.ndim == 3:
                labels = labels[..., 0]
        valid = labels < num_classes
        binc = np.bincount(labels[valid].ravel(), minlength=num_classes)
        class_counts += binc[:num_classes]

    mean = sum_c / n_pixels
    var = sumsq_c / n_pixels - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std, class_counts, len(img_paths)


def estimate_tiles(images_dir: Path, tile_size: int) -> int:
    total = 0
    for p in list_images(images_dir):
        with Image.open(p) as im:
            w, h = im.size
        total += max(1, (h // tile_size)) * max(1, (w // tile_size))
    return total


def class_weights_inverse_freq(counts: np.ndarray) -> list[float]:
    counts = counts.astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    freq = counts / counts.sum()
    w = 1.0 / freq
    w = w / w.mean()  # normalize so mean weight == 1
    return [float(x) for x in w]


def build_entry(name: str, splits: dict[str, tuple[Path, Path]], class_names: list[str], mask_kind: str) -> dict:
    num_classes = len(class_names)
    train_imgs, train_masks = splits["train"]
    mean, std, counts, _ = compute_split(train_imgs, train_masks, num_classes, mask_kind)

    distribution_pct = (counts / counts.sum()).tolist()
    weights = class_weights_inverse_freq(counts)

    return {
        "num_classes": num_classes,
        "in_channels": 3,
        "class_names": class_names,
        "class_weights": weights,
        "pixel_mean": [float(x) for x in mean],
        "pixel_std": [float(x) for x in std],
        "tile_size": TILE_SIZE,
        "estimated_train_tiles": estimate_tiles(splits["train"][0], TILE_SIZE),
        "estimated_val_tiles": estimate_tiles(splits["val"][0], TILE_SIZE),
        "estimated_test_tiles": estimate_tiles(splits["test"][0], TILE_SIZE),
        "class_distribution": {cn: float(p) for cn, p in zip(class_names, distribution_pct)},
    }


def vaihingen_splits() -> dict[str, tuple[Path, Path]]:
    base = MOCK / "vaihingen"
    return {s: (base / s / "images", base / s / "masks") for s in ("train", "val", "test")}


def potsdam_splits() -> dict[str, tuple[Path, Path]]:
    base = MOCK / "potsdam"
    return {s: (base / s / "images", base / s / "masks") for s in ("train", "val", "test")}


def loveda_splits() -> dict[str, tuple[Path, Path]]:
    base = MOCK / "loveda"
    # mock is "urban" domain only
    return {s: (base / s / "urban" / "images", base / s / "urban" / "masks") for s in ("train", "val", "test")}


def main() -> None:
    stats = {
        "vaihingen": build_entry("vaihingen", vaihingen_splits(), ISPRS_CLASS_NAMES, "isprs"),
        "potsdam": build_entry("potsdam", potsdam_splits(), ISPRS_CLASS_NAMES, "isprs"),
        "loveda": build_entry("loveda", loveda_splits(), LOVEDA_CLASS_NAMES, "loveda"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {OUT}")

    # Update status
    with STATUS.open() as f:
        status = json.load(f)
    status["wave1"]["eda"] = "done"
    with STATUS.open("w") as f:
        json.dump(status, f, indent=2)
    print("status.json: wave1.eda = done")


if __name__ == "__main__":
    main()
