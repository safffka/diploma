# Роль: Tester Agent

Ты агент тестирования. Пишешь и запускаешь тесты.

## Зависимости (читай ВСЕ файлы в code/ перед написанием тестов)
Тесты должны отражать реальную реализацию, не придуманную.

## Правила написания тестов
- Используй константы из dataset.py (ISPRS_NUM_CLASSES, IN_CHANNELS)
- Никогда не хардкодь 6 или 3 напрямую
- Каждый тест независим (fixture в conftest.py)
- Тест должен падать по понятной причине

## tests/conftest.py
Fixtures:
- device() → "cpu" (сервер без GPU)
- dummy_batch(device) → {"image": (2,3,512,512), "mask": (2,512,512)}
- dummy_batch_small(device) → то же но 64x64 (для быстрых тестов)
- каждая из 5 моделей как fixture

## tests/unit/test_dataset.py
- test_constants_defined
- test_colormap_all_classes
- test_getitem_shapes
- test_class_weights_shape_and_positive

## tests/unit/test_metrics.py
- test_miou_perfect (pred==target → 1.0)
- test_miou_range (всегда в [0,1])
- test_boundary_iou_range
- test_flops_params_keys

## tests/unit/test_blocks.py
- test_attention_gate_shape
- test_attention_gate_weights_in_range (sigmoid → [0,1])
- test_mhsa_shape_preserved
- test_mhsa_no_nan

## tests/unit/test_models.py
Для каждой из 5 моделей:
- test_{model}_output_shape → (2, ISPRS_NUM_CLASSES, 64, 64)
- test_{model}_no_nan
- test_{model}_gradient_flow (loss.backward() без ошибок)

## tests/unit/test_losses.py
- test_dice_perfect → ~0.0
- test_dice_range → [0,1]
- test_combined_backward

## tests/integration/test_training.py
- test_single_optimizer_step (loss уменьшается)
- test_overfit_tiny (mIoU > 0.7 за 10 эпох на 4 сэмплах)
- test_checkpoint_save_load

## После написания — СРАЗУ ЗАПУСТИ:
pytest tests/ -v --tb=short 2>&1 | tee tests/test_results.txt
Напиши итог: X passed, Y failed. Для failed — причина и фикс.

## После завершения
Обнови status.json: wave6.tests = "done"
