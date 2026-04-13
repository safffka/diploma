# Роль: Training Agent

Ты агент реализации тренировочной инфраструктуры.

## Зависимости (читай перед стартом)
- code/data/dataset.py
- code/evaluation/metrics.py
- code/models/__init__.py

## FILE: code/training/losses.py

DiceLoss(smooth=1e-6):
  pred: (B, num_classes, H, W) logits
  target: (B, H, W) long
  → scalar

CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=None):
  class_weights из dataset.compute_class_weights()
  → weighted CE + Dice

## FILE: code/training/trainer.py

class Trainer:
  __init__(model, optimizer, scheduler, loss_fn, device,
           num_classes, class_names, save_dir)

  train_epoch(dataloader) → {"loss": float, "miou": float}
  validate_epoch(dataloader) → {"loss": float, "miou": float, "boundary_iou": float}
  fit(train_loader, val_loader, epochs=100, patience=15)
    - early stopping по val miou
    - сохраняет лучший чекпоинт в save_dir/best.pth
    - логирует в tensorboard
  save_checkpoint(epoch, metrics, is_best=False)
  load_checkpoint(path) → epoch, metrics

## FILE: code/training/train.py
argparse:
  --model [fcn|unet|deeplab|attention|segformer]
  --dataset [vaihingen|potsdam|loveda]
  --dataset_path
  --epochs (default=100)
  --lr (default=1e-4)
  --batch_size (default=4)
  --save_dir
  --resume

main():
  1. info = get_dataset_info(args.dataset)
  2. model = get_model(args.model, info["num_classes"])
  3. class_weights = из info
  4. loss = CombinedLoss(class_weights=class_weights)
  5. optimizer = AdamW
  6. scheduler = CosineAnnealingLR
  7. trainer.fit()
  8. сохранить results.json в save_dir

## После завершения
Обнови status.json: wave5.* = "done"
