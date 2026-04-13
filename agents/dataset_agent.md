# Роль: Dataset Agent

Ты агент реализации Dataset-классов PyTorch.

## Зависимости (читай перед стартом)
- experiments/eda/dataset_stats.json
- code/data/augmentation.py

## Задача
Реализовать code/data/dataset.py

## Константы (экспортируются отсюда, все остальные импортируют отсюда)
ISPRS_CLASSES = [6 имён]
ISPRS_NUM_CLASSES = 6
ISPRS_COLORMAP = {(R,G,B): index, ...}
LOVEDA_CLASSES = [7 имён]
LOVEDA_NUM_CLASSES = 7
IN_CHANNELS = 3

## Классы
ISPRSDataset(root_dir, split, transform=None):
  - читает data/processed/vaihingen/{split}/
  - rgb_to_mask(img) конвертирует RGB mask → индексы классов
  - __getitem__ возвращает {"image": (3,H,W) float32, "mask": (H,W) long}
  - compute_class_weights() → tensor (6,)

LoveDADataset(root_dir, split, domain="urban", transform=None):
  - то же для LoveDA, 7 классов

get_dataset_info(dataset_name) → dict:
  Читает из dataset_stats.json
  Возвращает: num_classes, in_channels, class_names, class_weights, pixel_mean, pixel_std

get_dataloader(dataset_name, root_dir, split, batch_size, num_workers=2) → DataLoader

## После завершения
Обнови status.json: wave3.dataset_class = "done"
