# Роль: Metrics Agent

Ты агент реализации метрик качества.

## Зависимости
- code/data/dataset.py — импортировать константы классов

## Задача
Реализовать code/evaluation/metrics.py

## Функции

iou_per_class(pred, target, num_classes, ignore_index=255) → tensor(num_classes,)
  pred, target: (B,H,W) long tensors

compute_miou(pred, target, num_classes, ignore_index=255) → float
  среднее только по валидным классам

get_boundary_mask(mask, dilation=3) → bool tensor
  scipy.ndimage.binary_dilation для выделения граничных пикселей

compute_boundary_iou(pred, target, num_classes, dilation=3) → float
  IoU только на граничных пикселях

compute_flops_params(model, input_size=(1,3,512,512)) → dict
  {"flops_G": float, "params_M": float}
  использовать: from thop import profile

class MetricsTracker:
  __init__(num_classes, class_names)
  update(pred_batch, target_batch)
  compute() → dict {miou, boundary_iou, iou_per_class: {name: float}}
  reset()
  log_to_tensorboard(writer, epoch, prefix)

## После завершения
Обнови status.json: wave3.metrics = "done"
