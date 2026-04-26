"""Tests for code.evaluation.metrics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import ISPRS_NUM_CLASSES
from src.evaluation.metrics import (
    MetricsTracker,
    compute_boundary_iou,
    compute_flops_params,
    compute_miou,
)


def test_miou_perfect():
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 32, 32), dtype=torch.long)
    pred = target.clone()
    miou = compute_miou(pred, target, num_classes=ISPRS_NUM_CLASSES)
    assert miou == pytest.approx(1.0, abs=1e-6)


def test_miou_range():
    torch.manual_seed(0)
    pred = torch.randint(0, ISPRS_NUM_CLASSES, (2, 32, 32), dtype=torch.long)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 32, 32), dtype=torch.long)
    miou = compute_miou(pred, target, num_classes=ISPRS_NUM_CLASSES)
    assert 0.0 <= miou <= 1.0


def test_miou_zero():
    # Two-class disjoint case
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    pred = torch.ones((1, 8, 8), dtype=torch.long)
    miou = compute_miou(pred, target, num_classes=ISPRS_NUM_CLASSES)
    assert miou == pytest.approx(0.0, abs=1e-6)


def test_boundary_iou_range():
    torch.manual_seed(1)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (1, 32, 32), dtype=torch.long)
    pred = torch.randint(0, ISPRS_NUM_CLASSES, (1, 32, 32), dtype=torch.long)
    val = compute_boundary_iou(pred, target, num_classes=ISPRS_NUM_CLASSES)
    assert 0.0 <= val <= 1.0


def test_boundary_iou_perfect():
    target = torch.zeros((1, 16, 16), dtype=torch.long)
    target[:, :, 8:] = 1
    pred = target.clone()
    val = compute_boundary_iou(pred, target, num_classes=ISPRS_NUM_CLASSES)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_tracker_accumulate():
    tracker = MetricsTracker(num_classes=ISPRS_NUM_CLASSES)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 16, 16), dtype=torch.long)
    # Logits where argmax == target (perfect prediction)
    logits = torch.zeros(2, ISPRS_NUM_CLASSES, 16, 16)
    for c in range(ISPRS_NUM_CLASSES):
        logits[:, c][target == c] = 10.0
    tracker.update(logits, target)
    res = tracker.compute()
    assert "miou" in res and "boundary_iou" in res and "iou_per_class" in res
    assert res["miou"] == pytest.approx(1.0, abs=1e-6)


def test_flops_params_keys():
    # Use a tiny model to keep this fast; thop must succeed.
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, ISPRS_NUM_CLASSES, kernel_size=1),
    )
    out = compute_flops_params(model, input_size=(1, 3, 32, 32))
    assert "flops_G" in out and "params_M" in out
    assert out["flops_G"] >= 0.0
    assert out["params_M"] >= 0.0
