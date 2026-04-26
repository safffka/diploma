"""Tests for DiceLoss and CombinedLoss."""
from __future__ import annotations

import pytest
import torch

from src.data.dataset import ISPRS_NUM_CLASSES
from src.training.losses import CombinedLoss, DiceLoss


def _perfect_logits(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    b, h, w = target.shape
    logits = torch.full((b, num_classes, h, w), -10.0)
    for c in range(num_classes):
        logits[:, c][target == c] = 10.0
    return logits


def test_dice_perfect():
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    logits = _perfect_logits(target, ISPRS_NUM_CLASSES)
    loss = DiceLoss()(logits, target)
    assert float(loss) == pytest.approx(0.0, abs=1e-3)


def test_dice_range():
    torch.manual_seed(0)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    logits = torch.randn(2, ISPRS_NUM_CLASSES, 16, 16)
    loss = float(DiceLoss()(logits, target))
    assert 0.0 <= loss <= 1.0


def test_combined_positive():
    torch.manual_seed(0)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    logits = torch.randn(2, ISPRS_NUM_CLASSES, 16, 16)
    loss = float(CombinedLoss()(logits, target))
    assert loss > 0.0


def test_combined_backward():
    torch.manual_seed(0)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    logits = torch.randn(2, ISPRS_NUM_CLASSES, 16, 16, requires_grad=True)
    loss = CombinedLoss()(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_combined_with_class_weights():
    torch.manual_seed(0)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    logits = torch.randn(2, ISPRS_NUM_CLASSES, 16, 16, requires_grad=True)
    weights = torch.ones(ISPRS_NUM_CLASSES) * 0.5
    weights[0] = 2.0
    loss_fn = CombinedLoss(class_weights=weights)
    loss = loss_fn(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
