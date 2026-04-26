"""Entry point for training a segmentation model.

Example:
    python -m code.training.train --model unet --dataset vaihingen \
        --dataset_path /data --save_dir runs/unet_vaihingen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import get_dataloader, get_dataset_info
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a segmentation model.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fcn", "unet", "deeplab", "attention", "segformer"],
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["vaihingen", "potsdam", "loveda"],
    )
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = get_dataset_info(args.dataset)
    num_classes = int(info["num_classes"])
    class_names = list(info["class_names"])
    class_weights = torch.tensor(info["class_weights"], dtype=torch.float32)

    # Model is imported here at runtime (not at file-write time).
    from src.models import get_model

    model = get_model(args.model, num_classes=num_classes)

    train_loader = get_dataloader(
        args.dataset, args.dataset_path, "train",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        args.dataset, args.dataset_path, "val",
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    loss_fn = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5, class_weights=class_weights
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        num_classes=num_classes,
        class_names=class_names,
        save_dir=save_dir,
    )

    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        start_epoch = int(start_epoch) + 1

    result = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs, patience=args.patience,
        start_epoch=start_epoch,
    )

    results_path = save_dir / "results.json"
    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "num_classes": num_classes,
        "class_names": class_names,
        "epochs_run": len(result["history"]),
        "best_miou": result["best_miou"],
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "args": vars(args),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
