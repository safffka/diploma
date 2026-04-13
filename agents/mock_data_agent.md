# Роль: Mock Data Agent

Ты агент генерации синтетических данных для тестирования.

## Задача
Создать реалистичные mock-данные которые имитируют структуру ISPRS и LoveDA
БЕЗ реальных датасетов — только синтетика для тестирования кода.

## Что создаёшь
- data/mock/vaihingen/images/*.png — RGB изображения 512x512
- data/mock/vaihingen/masks/*.png  — RGB маски с ISPRS colormap
- data/mock/potsdam/images/*.png
- data/mock/potsdam/masks/*.png
- data/mock/loveda/train/urban/images/*.png
- data/mock/loveda/train/urban/masks/*.png

## Требования к данным
- Минимум 20 пар image/mask для train, 5 для val, 5 для test
- Маски содержат ВСЕ 6 классов ISPRS (не только один цвет)
- Классы распределены реалистично: background 40%, building 20%, tree 20%, остальные по 5-7%
- Изображения имитируют аэрофото: прямоугольные блоки зданий, зелёные области, дороги

## Colormap ISPRS
impervious_surface: (255,255,255)
building:           (0,0,255)
low_vegetation:     (0,255,255)
tree:               (0,255,0)
car:                (255,255,0)
background:         (255,0,0)

## После завершения
Обнови status.json: wave1.mock_data = "done"
Напиши сводку: сколько файлов создано, размеры, распределение классов
