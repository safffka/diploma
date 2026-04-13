# Project: Semantic Segmentation — Comparative Study

## Goal
Реализовать и сравнить 5 архитектур семантической сегментации
на датасетах высокого разрешения (ISPRS Vaihingen, Potsdam, LoveDA)

## Models (все реализуем сами на PyTorch)
1. fcn          — Fully Convolutional Network (baseline простейший)
2. unet         — U-Net с ResNet50 encoder (основной baseline)
3. deeplab      — DeepLabV3+ с atrous convolution + ASPP
4. attention    — U-Net + AttentionGate на skip + MHSA в bottleneck
5. segformer    — упрощённый SegFormer (MiT encoder + MLP decoder)

## Datasets
- ISPRS Vaihingen: 33 тайла ~2000x2000px, 6 классов
- ISPRS Potsdam: 38 тайлов 6000x6000px, 6 классов
- LoveDA: 5987 изображений 1024x1024px, 7 классов (domain shift)

## ISPRS Classes
0: impervious_surface: (255,255,255)
1: building:           (0,0,255)
2: low_vegetation:     (0,255,255)
3: tree:               (0,255,0)
4: car:                (255,255,0)
5: background:         (255,0,0)

## LoveDA Classes
0: background
1: building
2: road
3: water
4: barren
5: forest
6: agriculture

## Training (выполняется на Google Colab)
- Tile size: 512x512
- Batch size: 4
- Optimizer: AdamW lr=1e-4
- Loss: CrossEntropy(0.5) + Dice(0.5) + class_weights из EDA
- Epochs: 100, early stopping patience=15
- Все модели обучаются в одинаковых условиях

## Evaluation metrics
- mIoU (primary)
- Boundary IoU (качество границ)
- FLOPs (вычислительная сложность)
- Params (размер модели)
- Inference time ms/image

## Hypotheses
H1: attention mIoU >= unet mIoU + 5pp на ISPRS
H2: attention FLOPs <= unet FLOPs * 1.20
H3: attention BoundaryIoU >= unet BoundaryIoU + 15pp
H4: segformer mIoU >= unet mIoU на обоих датасетах

## Важно
- Никаких хардкоженных mean/std/class_weights
- Все статистики из experiments/eda/dataset_stats.json
- Код пишется здесь, обучение на Colab
- Все 5 моделей используют одинаковый Dataset-класс
- get_model(name) фабрика возвращает любую из 5 моделей
