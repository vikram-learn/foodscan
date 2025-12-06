# backend/data/generate_synthetic.py
"""
Generate a tiny synthetic "food" dataset for fast iteration.

Outputs (all under backend/data/synthetic):
 - images/<class>/*.jpg
 - labels.csv           (image_path,class_name)
 - features.npy         (N x F color-hist features)
 - labels.npy           (N ints)
 - label_map.json       (int -> class name)

This script uses the same color-histogram feature used by preprocess.py,
so downstream scripts (train_classical.py, prepare_quantum_data.py, quantum stubs)
can run against this dataset without needing the full Food-101 download.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv, json, os, random
from tqdm import tqdm

OUT_ROOT = Path(__file__).resolve().parent
SYN_ROOT = OUT_ROOT / "synthetic"
IMG_DIR = SYN_ROOT / "images"
LABELS_CSV = SYN_ROOT / "labels.csv"
FEATURES_NPY = SYN_ROOT / "features.npy"
LABELS_NPY = SYN_ROOT / "labels.npy"
LABEL_MAP = SYN_ROOT / "label_map.json"

IMG_SIZE = (128, 128)
HIST_BINS_PER_CHANNEL = 8
RANDOM_SEED = 42

CLASSES = [
    "rice",
    "fruit_bowl",
    "fried_potato",
    "salad",
    "dessert"
]

IMAGES_PER_CLASS = 40  # total ~200 images -> small but enough to test flows

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def ensure_dirs():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (IMG_DIR / c).mkdir(parents=True, exist_ok=True)

def _make_image_for_class(cls_name, idx):
    """
    Create a simple synthetic image for a class.
    Strategy:
     - Base color per class (gives different histograms)
     - Add simple shapes / noise / text so images are not identical
    """
    base_colors = {
        "rice": (245, 245, 240),
        "fruit_bowl": (220, 60, 80),
        "fried_potato": (230, 180, 70),
        "salad": (80, 170, 70),
        "dessert": (200, 120, 200),
    }
    bg = base_colors.get(cls_name, (180,180,180))
    img = Image.new("RGB", IMG_SIZE, bg)
    draw = ImageDraw.Draw(img)

    # Add a circle or rectangle with slight randomization
    w,h = IMG_SIZE
    # random rectangle
    rx = int(w * 0.1 + random.random() * w * 0.6)
    ry = int(h * 0.1 + random.random() * h * 0.6)
    x0 = random.randint(5, w//3)
    y0 = random.randint(5, h//3)
    shape_bbox = (x0, y0, x0+rx, y0+ry)
    shape_color = tuple(max(0, min(255, c + random.randint(-40, 40))) for c in bg)
    if random.random() > 0.5:
        draw.ellipse(shape_bbox, fill=shape_color, outline=None)
    else:
        draw.rectangle(shape_bbox, fill=shape_color)

    # Add small speckle noise
    for _ in range(200):
        px = random.randint(0, w-1)
        py = random.randint(0, h-1)
        img.putpixel((px,py), tuple(random.randint(0,255) for _ in range(3)))

    # Add class short text (optional)
    try:
        # PIL default font exists on most systems; if not, ignore
        draw.text((5, h-18), f"{cls_name[:6]}-{idx}", fill=(10,10,10))
    except Exception:
        pass

    return img

def compute_color_hist(img: Image.Image, bins=HIST_BINS_PER_CHANNEL):
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:,:,ch].ravel(), bins=bins, range=(0,255))
        feats.append(hist.astype(np.float32))
    feats = np.concatenate(feats)
    s = feats.sum()
    if s > 0:
        feats = feats / s
    return feats

def main():
    ensure_dirs()
    rows = []
    features = []
    labels = []
    class_to_idx = {c:i for i,c in enumerate(CLASSES)}

    print("Generating synthetic images...")
    for c in CLASSES:
        for i in range(IMAGES_PER_CLASS):
            img = _make_image_for_class(c, i)
            fname = f"{c}_{i:03d}.jpg"
            outp = IMG_DIR / c / fname
            img.save(outp, quality=80)
            rows.append((str(outp), c))
            feats = compute_color_hist(img)
            features.append(feats)
            labels.append(class_to_idx[c])

    features = np.stack(features).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)
    SYN_ROOT.mkdir(parents=True, exist_ok=True)

    # write labels.csv (image_path, class_name)
    with open(LABELS_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "class_name"])
        writer.writerows(rows)

    np.save(FEATURES_NPY, features)
    np.save(LABELS_NPY, labels)
    with open(LABEL_MAP, "w") as fh:
        json.dump({i: c for i,c in enumerate(CLASSES)}, fh, indent=2)

    print("Done.")
    print("Synthetic dataset path:", SYN_ROOT)
    print("images:", (IMG_DIR).resolve())
    print("labels.csv rows:", len(rows))
    print("features.npy shape:", features.shape)
    print("labels.npy shape:", labels.shape)
    print("label_map.json:", LABEL_MAP)

if __name__ == "__main__":
    main()
