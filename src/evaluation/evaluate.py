"""Evaluate trained models on the test split and produce a comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from src.data.dataset import get_dataloader, get_dataset_info
from src.evaluation.metrics import MetricsTracker, compute_flops_params
from src.models import get_model


def evaluate_dataset(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    class_names: List[str],
    device: torch.device,
) -> Dict:
    model.eval().to(device)
    tracker = MetricsTracker(num_classes=num_classes, class_names=class_names)
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images, targets = batch["image"], batch["mask"]
            images = images.to(device)
            logits = model(images)
            if isinstance(logits, dict):
                logits = logits.get("out", next(iter(logits.values())))
            tracker.update(logits, targets)
    return tracker.compute()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--compare_all", action="store_true")
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--models", nargs="+",
                   default=["fcn", "unet", "deeplab", "attention", "segformer"])
    p.add_argument("--dataset", type=str, default="vaihingen")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = get_dataset_info(args.dataset)
    num_classes = int(info["num_classes"])
    class_names = list(info["class_names"])

    test_loader = get_dataloader(
        args.dataset, args.dataset_path, "test",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    summary: Dict[str, Dict] = {}
    for name in args.models:
        ckpt_path = results_dir / name / "best.pth"
        if not ckpt_path.exists():
            print(f"[skip] {name}: missing {ckpt_path}")
            continue
        model = get_model(name, num_classes=num_classes)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)

        metrics = evaluate_dataset(model, test_loader, num_classes, class_names, device)
        try:
            fp = compute_flops_params(model.to(device))
        except Exception as e:
            print(f"[warn] FLOPs failed for {name}: {e}")
            fp = {"flops_G": float("nan"), "params_M": float("nan")}

        summary[name] = {
            "miou": metrics["miou"],
            "boundary_iou": metrics["boundary_iou"],
            "iou_per_class": metrics["iou_per_class"],
            "flops_G": fp["flops_G"],
            "params_M": fp["params_M"],
        }
        print(f"{name}: mIoU={metrics['miou']:.4f} BIoU={metrics['boundary_iou']:.4f} "
              f"params={fp['params_M']:.1f}M flops={fp['flops_G']:.1f}G")

    out = save_dir / "compare_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved comparison to {out}")

    print(f"\n{'Model':<12} {'mIoU':>8} {'BIoU':>8} {'Params_M':>10} {'FLOPs_G':>10}")
    print("-" * 52)
    for name, m in summary.items():
        print(f"{name:<12} {m['miou']:>8.4f} {m['boundary_iou']:>8.4f} "
              f"{m['params_M']:>10.1f} {m['flops_G']:>10.1f}")


if __name__ == "__main__":
    main()
