"""Evaluation metrics for semantic segmentation.

Implements mIoU, per-class IoU, Boundary IoU, and FLOPs/Params computation,
as well as a MetricsTracker that aggregates batch-level predictions over an
epoch and produces a summary dict suitable for logging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import binary_dilation


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _to_long_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.long()


# -----------------------------------------------------------------------------
# Per-class IoU
# -----------------------------------------------------------------------------

def iou_per_class(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Compute IoU for each class.

    Args:
        pred:   (B,H,W) long tensor of predicted class indices.
        target: (B,H,W) long tensor of ground-truth class indices.
        num_classes: number of classes.
        ignore_index: label to ignore.

    Returns:
        Tensor of shape (num_classes,) with IoU per class. Classes with no
        ground-truth and no prediction receive NaN.
    """
    pred = _to_long_tensor(pred)
    target = _to_long_tensor(target)

    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    ious = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        pred_c = pred == c
        target_c = target == c
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious[c] = float("nan")
        else:
            ious[c] = inter / union
    return ious


def compute_miou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Compute mean IoU over valid (non-NaN) classes."""
    ious = iou_per_class(pred, target, num_classes, ignore_index=ignore_index)
    valid = ~torch.isnan(ious)
    if valid.sum() == 0:
        return 0.0
    return float(ious[valid].mean().item())


# -----------------------------------------------------------------------------
# Boundary IoU
# -----------------------------------------------------------------------------

def get_boundary_mask(
    mask: Union[torch.Tensor, np.ndarray],
    dilation: int = 3,
) -> np.ndarray:
    """Extract boundary pixels from a label mask via morphological dilation.

    Args:
        mask: (H,W) or (B,H,W) integer label mask.
        dilation: structuring-element iterations.

    Returns:
        Boolean numpy array of same shape where True indicates a boundary pixel.
    """
    if isinstance(mask, torch.Tensor):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.asarray(mask)

    def _single(m: np.ndarray) -> np.ndarray:
        # Boundary = pixels where any neighbor has different label.
        # Use per-class dilation then XOR with class mask, union across classes.
        boundary = np.zeros_like(m, dtype=bool)
        classes = np.unique(m)
        for c in classes:
            cm = m == c
            dilated = binary_dilation(cm, iterations=dilation)
            eroded = ~binary_dilation(~cm, iterations=dilation)
            edge = dilated & (~eroded)
            boundary |= edge & cm
        return boundary

    if mask_np.ndim == 2:
        return _single(mask_np)
    elif mask_np.ndim == 3:
        out = np.zeros_like(mask_np, dtype=bool)
        for i in range(mask_np.shape[0]):
            out[i] = _single(mask_np[i])
        return out
    else:
        raise ValueError(f"Unsupported mask ndim={mask_np.ndim}")


def compute_boundary_iou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    num_classes: int,
    dilation: int = 3,
    ignore_index: int = 255,
) -> float:
    """Mean IoU computed only on boundary pixels of the ground-truth mask."""
    pred = _to_long_tensor(pred)
    target = _to_long_tensor(target)

    target_np = target.detach().cpu().numpy()
    boundary = get_boundary_mask(target_np, dilation=dilation)
    boundary_t = torch.from_numpy(boundary)

    valid = (target != ignore_index) & boundary_t
    if valid.sum() == 0:
        return 0.0

    pred_b = pred[valid]
    target_b = target[valid]

    ious = torch.full((num_classes,), float("nan"))
    for c in range(num_classes):
        p = pred_b == c
        t = target_b == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            ious[c] = float("nan")
        else:
            ious[c] = inter / union
    valid_c = ~torch.isnan(ious)
    if valid_c.sum() == 0:
        return 0.0
    return float(ious[valid_c].mean().item())


# -----------------------------------------------------------------------------
# FLOPs / Params
# -----------------------------------------------------------------------------

def compute_flops_params(
    model: torch.nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 512, 512),
) -> Dict[str, float]:
    """Compute FLOPs (GigaFLOPs) and parameter count (Millions) using thop.

    Returns dict with keys 'flops_G' and 'params_M'.
    """
    from thop import profile

    model.eval()
    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return {
        "flops_G": float(flops) / 1e9,
        "params_M": float(params) / 1e6,
    }


# -----------------------------------------------------------------------------
# MetricsTracker
# -----------------------------------------------------------------------------

class MetricsTracker:
    """Accumulates a confusion matrix over batches and produces summary metrics.

    - miou: mean IoU across valid classes.
    - boundary_iou: accumulated over batches using the ground-truth boundary.
    - iou_per_class: dict {class_name: iou}.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        ignore_index: int = 255,
        boundary_dilation: int = 3,
    ) -> None:
        self.num_classes = num_classes
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        assert len(class_names) == num_classes, (
            "class_names length must equal num_classes"
        )
        self.class_names: List[str] = list(class_names)
        self.ignore_index = ignore_index
        self.boundary_dilation = boundary_dilation
        self.reset()

    def reset(self) -> None:
        self._conf = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self._b_inter = np.zeros(self.num_classes, dtype=np.int64)
        self._b_union = np.zeros(self.num_classes, dtype=np.int64)

    def _update_conf(
        self, pred: np.ndarray, target: np.ndarray
    ) -> None:
        valid = (target != self.ignore_index) & (target >= 0) & (
            target < self.num_classes
        )
        p = pred[valid].astype(np.int64)
        t = target[valid].astype(np.int64)
        idx = t * self.num_classes + p
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self._conf += binc.reshape(self.num_classes, self.num_classes)

    def update(
        self,
        pred_batch: Union[torch.Tensor, np.ndarray],
        target_batch: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Update running statistics with one batch.

        If predictions are logits (B,C,H,W), argmax is taken along dim=1.
        """
        if isinstance(pred_batch, torch.Tensor):
            if pred_batch.ndim == 4:
                pred_batch = pred_batch.argmax(dim=1)
            pred_np = pred_batch.detach().cpu().numpy()
        else:
            pred_np = np.asarray(pred_batch)
            if pred_np.ndim == 4:
                pred_np = pred_np.argmax(axis=1)

        if isinstance(target_batch, torch.Tensor):
            target_np = target_batch.detach().cpu().numpy()
        else:
            target_np = np.asarray(target_batch)

        pred_np = pred_np.astype(np.int64)
        target_np = target_np.astype(np.int64)

        # Global confusion matrix
        self._update_conf(pred_np, target_np)

        # Boundary stats
        if pred_np.ndim == 2:
            pred_np = pred_np[None]
            target_np = target_np[None]

        for i in range(target_np.shape[0]):
            boundary = get_boundary_mask(
                target_np[i], dilation=self.boundary_dilation
            )
            valid = (target_np[i] != self.ignore_index) & boundary
            if not valid.any():
                continue
            p = pred_np[i][valid]
            t = target_np[i][valid]
            for c in range(self.num_classes):
                pc = p == c
                tc = t == c
                self._b_inter[c] += int(np.logical_and(pc, tc).sum())
                self._b_union[c] += int(np.logical_or(pc, tc).sum())

    def compute(self) -> Dict:
        conf = self._conf.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp
        union = tp + fp + fn

        ious = np.full(self.num_classes, np.nan)
        for c in range(self.num_classes):
            if union[c] > 0:
                ious[c] = tp[c] / union[c]

        valid = ~np.isnan(ious)
        miou = float(np.nanmean(ious)) if valid.any() else 0.0

        b_ious = np.full(self.num_classes, np.nan)
        for c in range(self.num_classes):
            if self._b_union[c] > 0:
                b_ious[c] = self._b_inter[c] / self._b_union[c]
        b_valid = ~np.isnan(b_ious)
        boundary_iou = float(np.nanmean(b_ious)) if b_valid.any() else 0.0

        iou_per_class_dict = {
            name: (float(ious[i]) if not np.isnan(ious[i]) else float("nan"))
            for i, name in enumerate(self.class_names)
        }

        return {
            "miou": miou,
            "boundary_iou": boundary_iou,
            "iou_per_class": iou_per_class_dict,
        }

    def log_to_tensorboard(
        self,
        writer,
        epoch: int,
        prefix: str = "val",
    ) -> None:
        """Write metrics to a TensorBoard SummaryWriter."""
        results = self.compute()
        writer.add_scalar(f"{prefix}/miou", results["miou"], epoch)
        writer.add_scalar(
            f"{prefix}/boundary_iou", results["boundary_iou"], epoch
        )
        for name, val in results["iou_per_class"].items():
            if not np.isnan(val):
                writer.add_scalar(f"{prefix}/iou_{name}", val, epoch)
