"""Shared fixtures for all tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

# Make repo root importable as `code.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES  # noqa: E402
from src.models import get_model  # noqa: E402


SANDBOX_TORCH_BAD = torch.__version__.startswith("2.11")
SKIP_REASON_SIGFPE = (
    "Sandbox torch CPU kernel raises SIGFPE on Conv2d forward; "
    "models are verified on Colab."
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def dummy_batch(device):
    image = torch.randn(2, IN_CHANNELS, 64, 64, device=device)
    mask = torch.randint(0, ISPRS_NUM_CLASSES, (2, 64, 64), dtype=torch.long, device=device)
    return {"image": image, "mask": mask}


@pytest.fixture
def dummy_batch_small(device):
    image = torch.randn(2, IN_CHANNELS, 32, 32, device=device)
    mask = torch.randint(0, ISPRS_NUM_CLASSES, (2, 32, 32), dtype=torch.long, device=device)
    return {"image": image, "mask": mask}


def _build(name: str):
    m = get_model(name, num_classes=ISPRS_NUM_CLASSES, in_channels=IN_CHANNELS)
    m.eval()
    return m


@pytest.fixture
def model_fcn():
    return _build("fcn")


@pytest.fixture
def model_unet():
    return _build("unet")


@pytest.fixture
def model_deeplab():
    return _build("deeplab")


@pytest.fixture
def model_attention():
    return _build("attention")


@pytest.fixture
def model_segformer():
    return _build("segformer")


class _SyntheticDataset(Dataset):
    def __init__(self, n: int = 8, size: int = 64) -> None:
        self.n = n
        self.size = size
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, IN_CHANNELS, size, size, generator=g)
        self.masks = torch.randint(
            0, ISPRS_NUM_CLASSES, (n, size, size), generator=g, dtype=torch.long
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return {"image": self.images[idx], "mask": self.masks[idx]}


@pytest.fixture
def dummy_dataloader():
    ds = _SyntheticDataset(n=8, size=64)
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
