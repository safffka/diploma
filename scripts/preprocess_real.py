"""Preprocessor for real ISPRS Vaihingen data.
Tiles large images to 512x512, converts RGB masks to class indices.
Usage: python3 scripts/preprocess_real.py [--raw_dir PATH] [--out_dir PATH]
"""
import numpy as np
from PIL import Image
import os, json, argparse
from pathlib import Path
from tqdm import tqdm

TILE_SIZE = 512
OVERLAP   = 64
STEP      = TILE_SIZE - OVERLAP
SEED      = 42

ISPRS_PALETTE = {
    (255,255,255): 0,  # impervious_surface
    (0,  0,  255): 1,  # building
    (0,  255,255): 2,  # low_vegetation
    (0,  255,  0): 3,  # tree
    (255,255,  0): 4,  # car
    (255,  0,  0): 5,  # background
}
BACKGROUND_IDX = 5
NUM_CLASSES    = 6
ISPRS_CLASSES  = ['impervious_surface','building','low_vegetation','tree','car','background']

def rgb_to_mask(rgb):
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    for color, idx in ISPRS_PALETTE.items():
        match = np.all(rgb == np.array(color, dtype=np.uint8), axis=2)
        out[match] = idx
    return out

def tile_pair(img, mask):
    H, W = img.shape[:2]
    tiles = []
    for y in range(0, H - TILE_SIZE + 1, STEP):
        for x in range(0, W - TILE_SIZE + 1, STEP):
            tiles.append((img[y:y+TILE_SIZE, x:x+TILE_SIZE],
                          mask[y:y+TILE_SIZE, x:x+TILE_SIZE]))
    return tiles

def should_keep(mask_tile):
    return (mask_tile == BACKGROUND_IDX).mean() < 0.90

def compute_stats(train_pairs):
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq     = np.zeros(3, dtype=np.float64)
    pixel_count  = 0
    class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for img_path, mask_path in tqdm(train_pairs, desc='Computing stats'):
        img  = np.array(Image.open(img_path).convert('RGB')).astype(np.float64) / 255.0
        mask = rgb_to_mask(np.array(Image.open(mask_path).convert('RGB')))
        pixel_sum   += img.reshape(-1, 3).sum(axis=0)
        pixel_sq    += (img**2).reshape(-1, 3).sum(axis=0)
        pixel_count += img.shape[0] * img.shape[1]
        for c in range(NUM_CLASSES):
            class_counts[c] += (mask == c).sum()
    mean    = pixel_sum / pixel_count
    std     = np.sqrt(np.maximum(pixel_sq / pixel_count - mean**2, 0))
    freq    = class_counts / class_counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    class_dist = {ISPRS_CLASSES[i]: float(freq[i]) for i in range(NUM_CLASSES)}
    return mean.tolist(), std.tolist(), weights.tolist(), class_dist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='data/raw/ISPRS_semantic_labeling_Vaihingen')
    parser.add_argument('--out_dir', default='data/processed/vaihingen')
    parser.add_argument('--stats',   default='experiments/eda/dataset_stats.json')
    args = parser.parse_args()

    raw_dir    = Path(args.raw_dir)
    out_dir    = Path(args.out_dir)
    stats_path = Path(args.stats)
    img_dir    = raw_dir / 'top'
    mask_dir   = raw_dir / 'gts_for_participants'

    # Создаём папки
    for split in ['train','val','test']:
        (out_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (out_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    # Находим пары
    imgs  = set(os.listdir(img_dir))
    masks = set(os.listdir(mask_dir))
    pairs = sorted([(img_dir/f, mask_dir/f) for f in (imgs & masks)])
    print(f'Найдено пар image+mask: {len(pairs)}')

    # Сплит по исходным изображениям
    np.random.seed(SEED)
    idx     = np.random.permutation(len(pairs))
    n_train = int(len(pairs) * 0.7)
    n_val   = int(len(pairs) * 0.15)
    split_pairs = {
        'train': [pairs[i] for i in idx[:n_train]],
        'val':   [pairs[i] for i in idx[n_train:n_train+n_val]],
        'test':  [pairs[i] for i in idx[n_train+n_val:]]
    }
    for s, p in split_pairs.items():
        print(f'  {s}: {len(p)} изображений')

    # Статистики только по train
    print()
    mean, std, weights, class_dist = compute_stats(split_pairs['train'])
    print(f'mean = {[round(x,4) for x in mean]}')
    print(f'std  = {[round(x,4) for x in std]}')

    # Тайлинг
    counts = {'train':0, 'val':0, 'test':0, 'filtered':0}
    for split, data in split_pairs.items():
        counter = 0
        for img_path, mask_path in tqdm(data, desc=f'Tiling {split}'):
            img  = np.array(Image.open(img_path).convert('RGB'))
            mask = rgb_to_mask(np.array(Image.open(mask_path).convert('RGB')))
            for img_t, mask_t in tile_pair(img, mask):
                fname = f'vaihingen_{split}_{counter:04d}.png'
                if should_keep(mask_t):
                    Image.fromarray(img_t).save(out_dir / split / 'images' / fname)
                    Image.fromarray(mask_t).save(out_dir / split / 'masks'  / fname)
                    counts[split] += 1
                    counter += 1
                else:
                    counts['filtered'] += 1

    # Итог
    print(f'\n{"Split":<8} {"Images":>7} {"Tiles":>7}')
    print('-' * 24)
    for s in ['train','val','test']:
        print(f'{s:<8} {len(split_pairs[s]):>7} {counts[s]:>7}')
    print(f'Отфильтровано: {counts["filtered"]} тайлов')

    # Обновляем dataset_stats.json
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
    stats['vaihingen'] = {
        'num_classes':        NUM_CLASSES,
        'in_channels':        3,
        'class_names':        ISPRS_CLASSES,
        'class_weights':      weights,
        'pixel_mean':         mean,
        'pixel_std':          std,
        'tile_size':          TILE_SIZE,
        'actual_train_tiles': counts['train'],
        'actual_val_tiles':   counts['val'],
        'actual_test_tiles':  counts['test'],
        'filtered_tiles':     counts['filtered'],
        'class_distribution': class_dist
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f'\nСохранено: {stats_path}')

main()
