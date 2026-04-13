# Роль: Colab Agent

Ты агент создания Google Colab notebook для обучения на GPU.

## Зависимости
- Все файлы в code/ должны быть готовы и протестированы
- tests/test_results.txt должен показывать все тесты пройдены

## Задача
Создать output/train_colab.ipynb

## Структура ноутбука

### Cell 1: Setup
!git clone {repo_url} diploma
%cd diploma
!pip install -r requirements_colab.txt

### Cell 2: Mount Google Drive (для сохранения чекпоинтов)
from google.colab import drive
drive.mount('/content/drive')

### Cell 3: Загрузка датасетов
Инструкции по загрузке ISPRS Vaihingen и Potsdam
(ссылки на официальные страницы, команды распаковки)

### Cell 4: Preprocessing
!python -c "from code.data.preprocessor import DataPreprocessor; ..."
Запуск EDA и препроцессинга

### Cell 5-9: Обучение каждой из 5 моделей
for model_name in ["fcn", "unet", "deeplab", "attention", "segformer"]:
    !python code/training/train.py \
        --model {model_name} \
        --dataset vaihingen \
        --dataset_path data/processed \
        --epochs 100 \
        --save_dir experiments/{model_name}

### Cell 10: Сравнение всех моделей
!python code/evaluation/evaluate.py --compare_all \
    --results_dir experiments/ \
    --save_dir experiments/comparison

### Cell 11: Визуализация результатов
Графики training curves, confusion matrix, примеры сегментации

### Cell 12: Сохранение на Drive
!cp -r experiments/comparison /content/drive/MyDrive/diploma_results/

## После завершения
Обнови status.json: wave7.colab_notebook = "done"
