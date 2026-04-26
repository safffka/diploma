"""Trainer class for semantic segmentation.

Handles train/validate loops, TensorBoard logging, checkpointing,
and early stopping on validation mIoU.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from src.evaluation.metrics import MetricsTracker


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: nn.Module,
        device: torch.device,
        num_classes: int,
        class_names: Sequence[str],
        save_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn.to(device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.device = device
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if SummaryWriter is not None:
            self.writer = SummaryWriter(log_dir=str(self.save_dir / "tb"))

        self.best_miou: float = -1.0
        self.best_epoch: int = -1
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        tracker = MetricsTracker(self.num_classes, self.class_names)
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
            with torch.no_grad():
                tracker.update(logits.detach(), masks.detach())

        avg_loss = total_loss / max(n, 1)
        results = tracker.compute()
        return {"loss": avg_loss, "miou": results["miou"]}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricsTracker(self.num_classes, self.class_names)
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            bs = images.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
            tracker.update(logits, masks)

        avg_loss = total_loss / max(n, 1)
        results = tracker.compute()
        return {
            "loss": avg_loss,
            "miou": results["miou"],
            "boundary_iou": results["boundary_iou"],
            "iou_per_class": results["iou_per_class"],
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        start_epoch: int = 0,
    ) -> Dict[str, Any]:
        no_improve = 0
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_m = self.train_epoch(train_loader)
            val_m = self.validate_epoch(val_loader)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_miou": train_m["miou"],
                "val_loss": val_m["loss"],
                "val_miou": val_m["miou"],
                "val_boundary_iou": val_m["boundary_iou"],
                "time_sec": elapsed,
            }
            self.history.append(record)

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_m["loss"], epoch)
                self.writer.add_scalar("train/miou", train_m["miou"], epoch)
                self.writer.add_scalar("val/loss", val_m["loss"], epoch)
                self.writer.add_scalar("val/miou", val_m["miou"], epoch)
                self.writer.add_scalar("val/boundary_iou", val_m["boundary_iou"], epoch)
                for name, v in val_m["iou_per_class"].items():
                    try:
                        if v == v:  # not NaN
                            self.writer.add_scalar(f"val/iou_{name}", v, epoch)
                    except Exception:
                        pass

            is_best = val_m["miou"] > self.best_miou
            if is_best:
                self.best_miou = val_m["miou"]
                self.best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            self.save_checkpoint(epoch, record, is_best=is_best)
            print(
                f"[epoch {epoch:03d}] train_loss={train_m['loss']:.4f} "
                f"train_miou={train_m['miou']:.4f} val_loss={val_m['loss']:.4f} "
                f"val_miou={val_m['miou']:.4f} val_bIoU={val_m['boundary_iou']:.4f} "
                f"best={self.best_miou:.4f}@{self.best_epoch} ({elapsed:.1f}s)"
            )

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience}).")
                break

        if self.writer is not None:
            self.writer.flush()

        return {
            "best_miou": self.best_miou,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }

    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict()
                if self.scheduler is not None and hasattr(self.scheduler, "state_dict")
                else None
            ),
            "metrics": metrics,
            "best_miou": self.best_miou,
            "best_epoch": self.best_epoch,
        }
        last_path = self.save_dir / "last.pth"
        torch.save(state, last_path)
        if is_best:
            torch.save(state, self.save_dir / "best.pth")

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass
        if (
            self.scheduler is not None
            and ckpt.get("scheduler_state") is not None
            and hasattr(self.scheduler, "load_state_dict")
        ):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            except Exception:
                pass
        self.best_miou = float(ckpt.get("best_miou", -1.0))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        return ckpt.get("epoch", 0), ckpt.get("metrics", {})
