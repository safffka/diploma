"""Forward-pass / gradient tests for all 5 models."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from code.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from tests.conftest import SANDBOX_TORCH_BAD, SKIP_REASON_SIGFPE


pytestmark = pytest.mark.skipif(SANDBOX_TORCH_BAD, reason=SKIP_REASON_SIGFPE)


MODEL_NAMES = ["fcn", "unet", "deeplab", "attention", "segformer"]


def _get(name, fixtures):
    return fixtures[f"model_{name}"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_output_shape(
    name, model_fcn, model_unet, model_deeplab, model_attention, model_segformer
):
    fixtures = {
        "model_fcn": model_fcn, "model_unet": model_unet,
        "model_deeplab": model_deeplab, "model_attention": model_attention,
        "model_segformer": model_segformer,
    }
    model = _get(name, fixtures)
    x = torch.randn(2, IN_CHANNELS, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, ISPRS_NUM_CLASSES, 64, 64)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_no_nan(
    name, model_fcn, model_unet, model_deeplab, model_attention, model_segformer
):
    fixtures = {
        "model_fcn": model_fcn, "model_unet": model_unet,
        "model_deeplab": model_deeplab, "model_attention": model_attention,
        "model_segformer": model_segformer,
    }
    model = _get(name, fixtures)
    x = torch.randn(2, IN_CHANNELS, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_gradient_flow(
    name, model_fcn, model_unet, model_deeplab, model_attention, model_segformer
):
    fixtures = {
        "model_fcn": model_fcn, "model_unet": model_unet,
        "model_deeplab": model_deeplab, "model_attention": model_attention,
        "model_segformer": model_segformer,
    }
    model = _get(name, fixtures)
    model.train()
    x = torch.randn(2, IN_CHANNELS, 64, 64, requires_grad=False)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 64, 64), dtype=torch.long)
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    # Verify at least one parameter received a non-zero gradient.
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and torch.any(p.grad != 0):
            has_grad = True
            break
    assert has_grad
