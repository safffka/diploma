"""Tests for shared building blocks."""
from __future__ import annotations

import pytest
import torch

from code.models.blocks import (
    AttentionGate,
    ConvBNReLU,
    MultiHeadSelfAttention,
)
from tests.conftest import SANDBOX_TORCH_BAD, SKIP_REASON_SIGFPE


@pytest.mark.skipif(SANDBOX_TORCH_BAD, reason=SKIP_REASON_SIGFPE)
def test_conv_bn_relu_shape():
    block = ConvBNReLU(3, 8).eval()
    x = torch.randn(2, 3, 16, 16)
    y = block(x)
    assert y.shape == (2, 8, 16, 16)


@pytest.mark.skipif(SANDBOX_TORCH_BAD, reason=SKIP_REASON_SIGFPE)
def test_attention_gate_shape():
    g = torch.randn(2, 16, 8, 8)
    x = torch.randn(2, 8, 16, 16)
    gate = AttentionGate(g_ch=16, x_ch=8, inter_ch=4).eval()
    out = gate(g, x)
    assert out.shape == x.shape


@pytest.mark.skipif(SANDBOX_TORCH_BAD, reason=SKIP_REASON_SIGFPE)
def test_attention_gate_weights_range():
    """Inspect the alpha map (sigmoid output) is within [0,1]."""
    g = torch.randn(2, 16, 8, 8)
    x = torch.ones(2, 8, 16, 16)
    gate = AttentionGate(g_ch=16, x_ch=8, inter_ch=4).eval()
    out = gate(g, x)
    # x is all ones so out == alpha broadcast across channels
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0 + 1e-6


def test_mhsa_shape_preserved():
    mhsa = MultiHeadSelfAttention(channels=16, num_heads=4).eval()
    x = torch.randn(2, 16, 4, 4)
    y = mhsa(x)
    assert y.shape == x.shape


def test_mhsa_no_nan():
    mhsa = MultiHeadSelfAttention(channels=16, num_heads=4).eval()
    x = torch.randn(2, 16, 4, 4)
    y = mhsa(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
