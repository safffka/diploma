# Роль: Augmentation Agent

Ты агент аугментации данных.

## Зависимости
- experiments/eda/dataset_stats.json — mean/std для нормализации

## Задача
Реализовать code/data/augmentation.py

## Два пайплайна

TRAIN_TRANSFORMS (аугментация + нормализация):
- HorizontalFlip(p=0.5)
- VerticalFlip(p=0.5)
- RandomRotate90(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.3)
- GaussNoise(p=0.2)
- Normalize(mean, std) ← из dataset_stats.json
- ToTensorV2()

VAL_TRANSFORMS (только нормализация):
- Normalize(mean, std) ← из dataset_stats.json
- ToTensorV2()

## Критичное правило
Геометрические аугментации (flip, rotate) применяются СИНХРОННО к image и mask.
Цветовые аугментации (jitter, noise) применяются ТОЛЬКО к image, не к mask.
Использовать albumentations — он это делает автоматически через additional_targets.

## После завершения
Обнови status.json: wave2.augmentation = "done"
