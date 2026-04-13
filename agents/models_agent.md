# Роль: Models Agent

Ты агент реализации всех 5 архитектур моделей.

## Зависимости (читай ПЕРВЫМ)
- code/data/dataset.py — импортировать IN_CHANNELS, ISPRS_NUM_CLASSES
- context.md — описания архитектур

## Задача
Реализовать 5 моделей в code/models/

## FILE: code/models/blocks.py
Общие блоки для всех моделей:
- ConvBNReLU(in_ch, out_ch, kernel=3, padding=1, dilation=1)
- ResidualBlock(in_ch, out_ch)
- ASPP(in_channels, out_channels, rates=[6,12,18]) — для DeepLab
- AttentionGate(g_ch, x_ch, inter_ch) — для Attention U-Net
  Комментировать каждое изменение формы тензора
- MultiHeadSelfAttention(channels, num_heads=8, dropout=0.1)
  (B,C,H,W) → (B,HW,C) → MHA → (B,C,H,W)
  Комментировать каждый reshape

## FILE: code/models/fcn.py
FCN — простейший baseline:
- VGG-style encoder (5 блоков conv)
- Decoder через bilinear upsampling
- forward(x) → (B, num_classes, H, W)

## FILE: code/models/unet.py
U-Net с ResNet50-style encoder:
- EncoderBlock: ConvBNReLU x2 + MaxPool → возвращает (pooled, skip)
- DecoderBlock: ConvTranspose2d + concat skip + ConvBNReLU x2
- 4 уровня: [64, 128, 256, 512], bottleneck 1024
- forward(x) → (B, num_classes, H, W)

## FILE: code/models/deeplab.py
DeepLabV3+:
- ResNet50 encoder с atrous convolutions (dilation в последних блоках)
- ASPP модуль (rates=[6,12,18] + global avg pool)
- Lightweight decoder (low-level features + upsampling)
- forward(x) → (B, num_classes, H, W)

## FILE: code/models/attention_unet.py
U-Net + Attention:
- Тот же encoder что в unet.py
- Bottleneck: ConvBNReLU → MultiHeadSelfAttention → ConvBNReLU
- AttentionDecoderBlock: AttentionGate на skip перед concat
- forward(x) → (B, num_classes, H, W)

## FILE: code/models/segformer.py
Упрощённый SegFormer:
- Mix Transformer encoder (4 стадии, patch embedding + efficient self-attention)
- MLP decoder (concat multi-scale features → linear)
- forward(x) → (B, num_classes, H, W)

## FILE: code/models/__init__.py
get_model(name, num_classes=6, in_channels=3) → model
  name: "fcn" | "unet" | "deeplab" | "attention" | "segformer"

## Правила
- Все shape-трансформации прокомментированы
- Нет хардкода num_classes — берётся из аргумента
- Все модели принимают (B, in_channels, H, W) и возвращают (B, num_classes, H, W)

## После завершения
Обнови status.json: wave4.* = "done"
