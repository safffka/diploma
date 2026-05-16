# Семантическая сегментация аэрофотоснимков

**ВКР** Спицин С.Е., УрФУ, РИМ-240950, 2026

## Результаты

| Модель | mIoU | BIoU | Params | FLOPs |
|--------|------|------|--------|-------|
| U-Net PT | 0.618 | 0.340 | 67.1M | 75.9G |
| Attention U-Net PT | 0.614 | 0.347 | 87.0M | 83.4G |
| U-Net | 0.558 | 0.315 | 37.5M | 298.4G |
| Attention U-Net | 0.544 | 0.322 | 37.8M | 302.8G |
| FCN | 0.532 | 0.298 | 3.7M | 73.1G |
| DeepLabV3+ | 0.459 | 0.252 | 6.5M | 33.2G |
| SegFormer | 0.459 | 0.253 | 2.3M | 2.9G |

## Установка

pip install -r requirements.txt

## Данные

kaggle datasets download -d bkfateam/potsdamvaihingen -p data/raw/
python3 scripts/preprocess_real.py

## Обучение

python3 src/training/train.py --model unet_pt --dataset vaihingen --dataset_path data/processed --epochs 100 --batch_size 4 --lr 5e-5 --save_dir experiments/unet_pt --device cuda

## Инференс

python3 src/inference.py --model unet_pt --checkpoint experiments/unet_pt/best.pth --image image.png --output mask.png

## Воспроизведение

pip install -r requirements.txt && python3 scripts/preprocess_real.py && python3 src/training/train.py --model unet_pt --dataset vaihingen --dataset_path data/processed --epochs 100 --save_dir experiments/unet_pt

## Jupyter ноутбук

Для интерактивного запуска используй ноутбук:

```bash
pip install jupyterlab
jupyter lab notebooks/train_local.ipynb
```

Ноутбук содержит все этапы:
1. Установка зависимостей
2. Загрузка датасета (Kaggle)
3. Предобработка
4. Проверка всех 7 моделей
5. Обучение (пропускает уже обученные модели)
6. Оценка на тестовой выборке
7. Проверка гипотез H1/H2/H3
8. Графики кривых обучения
9. Инференс на примере
