"""Integration tests for the Trainer + small overfit run."""
from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import IN_CHANNELS, ISPRS_NUM_CLASSES
from src.evaluation.metrics import compute_miou
from src.models import get_model
from src.training.losses import CombinedLoss
from src.training.trainer import Trainer
from tests.conftest import SANDBOX_TORCH_BAD, SKIP_REASON_SIGFPE


pytestmark = pytest.mark.skipif(SANDBOX_TORCH_BAD, reason=SKIP_REASON_SIGFPE)


def _tiny_loader(n: int = 4, size: int = 32, batch_size: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, IN_CHANNELS, size, size, generator=g)
    masks = torch.randint(0, ISPRS_NUM_CLASSES, (n, size, size), generator=g, dtype=torch.long)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i): return {"image": images[i], "mask": masks[i]}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False, num_workers=0)


def test_single_optimizer_step():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = get_model("unet").to(device)
    optim = AdamW(model.parameters(), lr=1e-3)
    loss_fn = CombinedLoss()

    x = torch.randn(2, IN_CHANNELS, 64, 64)
    target = torch.randint(0, ISPRS_NUM_CLASSES, (2, 64, 64), dtype=torch.long)

    model.train()
    losses = []
    for _ in range(3):
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        optim.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_trainer_save_load_checkpoint(tmp_path):
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = get_model("fcn").to(device)
    optim = AdamW(model.parameters(), lr=1e-3)
    loss_fn = CombinedLoss()
    trainer = Trainer(
        model=model, optimizer=optim, scheduler=None, loss_fn=loss_fn,
        device=device, num_classes=ISPRS_NUM_CLASSES,
        class_names=[f"c{i}" for i in range(ISPRS_NUM_CLASSES)],
        save_dir=tmp_path,
    )
    trainer.best_miou = 0.42
    trainer.best_epoch = 7
    trainer.save_checkpoint(epoch=7, metrics={"val_miou": 0.42}, is_best=True)
    ckpt_path = tmp_path / "best.pth"
    assert ckpt_path.exists()

    # Build a fresh trainer and load
    model2 = get_model("fcn").to(device)
    optim2 = AdamW(model2.parameters(), lr=1e-3)
    trainer2 = Trainer(
        model=model2, optimizer=optim2, scheduler=None, loss_fn=loss_fn,
        device=device, num_classes=ISPRS_NUM_CLASSES,
        class_names=[f"c{i}" for i in range(ISPRS_NUM_CLASSES)],
        save_dir=tmp_path,
    )
    epoch, _ = trainer2.load_checkpoint(ckpt_path)
    assert epoch == 7
    assert trainer2.best_miou == pytest.approx(0.42)

    # Weights should match
    sd1 = model.state_dict()
    sd2 = model2.state_dict()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k])


def test_overfit_tiny():
    """Overfit a tiny UNet on 4 samples 32x32 — expect mIoU > 0.5 within 20 epochs."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    n, size = 4, 32
    g = torch.Generator().manual_seed(0)
    images = torch.randn(n, IN_CHANNELS, size, size, generator=g)
    masks = torch.randint(0, ISPRS_NUM_CLASSES, (n, size, size), generator=g, dtype=torch.long)

    model = get_model("unet").to(device)
    optim = AdamW(model.parameters(), lr=3e-3)
    loss_fn = CombinedLoss()

    model.train()
    for _ in range(20):
        optim.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(images)
        pred = logits.argmax(dim=1)
    miou = compute_miou(pred, masks, num_classes=ISPRS_NUM_CLASSES)
    assert miou > 0.5, f"expected mIoU>0.5 after overfit, got {miou:.3f}"
