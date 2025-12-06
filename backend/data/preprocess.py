# backend/data/preprocess.py
"""
Create labels.csv and simple image features from Food-101 images.

Outputs:
 - backend/data/labels.csv       (image_path,class_name)
 - backend/data/features.npy     (N x F numpy array)
 - backend/data/labels.npy       (N array of integer labels)
 - backend/data/label_map.json   (int->class mapping)
"""
import os, sys, json
from pathlib import Path
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

# config
ROOT_ENV = "FOOD101_ROOT"  # should point to images root (folder containing class subfolders)
OUT_DIR = Path(__file__).resolve().parent
IMG_SIZE = (128, 128)   # small size to speed up feature extraction
HIST_BINS_PER_CHANNEL = 8

def get_root():
    val = os.environ.get(ROOT_ENV)
    if not val:
        raise SystemExit(f"Error: set {ROOT_ENV} to images root (e.g. ~/datasets/food-101/images)")
    p = Path(val).expanduser()
    if not p.exists():
        raise SystemExit(f"Error: path {p} does not exist")
    return p

def compute_color_hist(img: Image.Image, bins=8):
    # returns normalized histogram concatenated for RGB (bins*3)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:,:,ch].ravel(), bins=bins, range=(0,255))
        feats.append(hist.astype(np.float32))
    feats = np.concatenate(feats)
    # L1 normalize
    s = feats.sum()
    if s > 0:
        feats = feats / s
    return feats

def main():
    root = get_root()
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit("No class folders found under FOOD101_ROOT")
    label_map = {i: c for i,c in enumerate(classes)}
    class_to_idx = {c:i for i,c in label_map.items()}

    rows = []
    features = []
    labels = []

    print("Scanning classes:", len(classes))
    for cls in classes:
        cls_dir = root / cls
        imgs = list(cls_dir.glob("*.jpg"))
        for img_p in tqdm(imgs, desc=cls, leave=False):
            rows.append((str(img_p), cls))
            try:
                with Image.open(img_p) as im:
                    im = im.resize(IMG_SIZE)
                    feat = compute_color_hist(im, bins=HIST_BINS_PER_CHANNEL)
                    features.append(feat)
                    labels.append(class_to_idx[cls])
            except Exception as e:
                # skip unreadable images
                print("skip", img_p, "err", e)

    features = np.stack(features) if features else np.zeros((0, HIST_BINS_PER_CHANNEL*3), dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # make outputs dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_p = OUT_DIR / "labels.csv"
    print("Writing", csv_p)
    with open(csv_p, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "class_name"])
        writer.writerows(rows)

    np.save(OUT_DIR / "features.npy", features)
    np.save(OUT_DIR / "labels.npy", labels)
    with open(OUT_DIR / "label_map.json", "w") as fh:
        json.dump(label_map, fh, indent=2)

    print("Done. features:", features.shape, "labels:", labels.shape)
    print("Files written to", OUT_DIR)

if __name__ == "__main__":
    main()
