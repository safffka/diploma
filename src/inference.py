import argparse, sys, os
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import get_model
from src.data.dataset import ISPRS_NUM_CLASSES, ISPRS_COLORMAP
from src.data.augmentation import get_transforms

COLORMAP_INV = {v: k for k, v in ISPRS_COLORMAP.items()}

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in COLORMAP_INV.items():
        rgb[mask == idx] = color
    return rgb

def main():
    parser = argparse.ArgumentParser(description='Segmentation inference')
    parser.add_argument('--model', required=True,
        choices=['fcn','unet','deeplab','attention','segformer','unet_pt','attention_pt'])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', default='output/mask.png')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stats', default='experiments/eda/dataset_stats.json')
    args = parser.parse_args()

    device = torch.device(args.device)

    model = get_model(args.model, num_classes=ISPRS_NUM_CLASSES).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f'Model: {args.model}')

    img = Image.open(args.image).convert('RGB')
    print(f'Image: {img.size[0]}x{img.size[1]}')

    import json
    stats = {}
    if os.path.exists(args.stats):
        stats = json.load(open(args.stats)).get('vaihingen', {})

    transform = get_transforms(stats, train=False)
    img_np = np.array(img)
    tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).argmax(1).squeeze().cpu().numpy().astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    Image.fromarray(mask_to_rgb(pred)).save(args.output)

    class_names = ['impervious','building','low_veg','tree','car','background']
    found = [class_names[c] for c in range(6) if (pred == c).any()]
    print(f'Classes: {found}')
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()
