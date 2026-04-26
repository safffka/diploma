"""Loss functions for semantic segmentation.

- DiceLoss: multi-class soft Dice via softmax + one-hot target.
- CombinedLoss: weighted CE (optionally class-weighted) + Dice.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class soft Dice loss.

    Args:
        smooth: numerical smoothing constant.
        ignore_index: label value to ignore in target.
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), target: (B, H, W) long
        if logits.dim() != 4:
            raise ValueError(f"expected logits (B,C,H,W), got {tuple(logits.shape)}")
        if target.dim() != 3:
            raise ValueError(f"expected target (B,H,W), got {tuple(target.shape)}")

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        valid = (target != self.ignore_index).unsqueeze(1).float()  # (B,1,H,W)
        target_clamped = target.clone()
        target_clamped[target == self.ignore_index] = 0
        one_hot = F.one_hot(target_clamped, num_classes=num_classes)  # (B,H,W,C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B,C,H,W)

        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        inter = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted CrossEntropy + Dice loss.

    Args:
        ce_weight: weight on cross-entropy term.
        dice_weight: weight on Dice term.
        class_weights: optional (num_classes,) tensor or list for CE weighting.
        ignore_index: label value to ignore.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.ignore_index = ignore_index

        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        # Register as buffer so .to(device) moves it with the module.
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.dice = DiceLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if isinstance(self.class_weights, torch.Tensor) else None
        ce = F.cross_entropy(
            logits, target, weight=weight, ignore_index=self.ignore_index
        )
        dice = self.dice(logits, target)
        return self.ce_weight * ce + self.dice_weight * dice
