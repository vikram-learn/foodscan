# backend/data/download_and_map.py
"""
Simple script to prepare a small training dataset:
 - Assumes you have downloaded Food-101 images folder (or will use a local subset).
 - Builds data/labels.csv mapping image -> class -> nutrition (json string)
 - If USDA API key is provided in env USDA_API_KEY, it will try to fetch nutrition facts for a class.
Instructions:
 - Put this file at backend/data/download_and_map.py
 - Edit DATA_ROOT to point to your downloaded Food-101 "images" folder (or point to frontend/assets/test_images)
 - Run: python backend/data/download_and_map.py
"""
import os, csv, json, time, requests
from pathlib import Path

# CONFIG: point this to where Food-101 images are (images/<class>/<img.jpg>)
DATA_ROOT = Path(os.environ.get("FOOD101_ROOT", "/home/vikram/datasets/food-101/images"))
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_CSV = OUT_DIR / "labels.csv"

# Small local fallback nutrition lookup (example values, expand later)
FALLBACK_NUTRITION = {
    "pizza": {"calories": 266, "carbs": 33, "protein": 11, "fat": 10},
    "rice": {"calories": 130, "carbs": 28, "protein": 2.7, "fat": 0.3},
    "ice_cream": {"calories": 207, "carbs": 24, "protein": 3.5, "fat": 11},
    # add more representative defaults...
}

USDA_API_KEY = os.environ.get("USDA_API_KEY")  # optional; set if you want real USDA lookups
USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

def usda_lookup(query):
    if not USDA_API_KEY:
        return None
    params = {"api_key": USDA_API_KEY, "query": query, "pageSize": 1}
    try:
        r = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        foods = data.get("foods") or []
        if not foods:
            return None
        # try to extract basic macros
        nutrients = {n["nutrientName"].lower(): n.get("value") for n in foods[0].get("foodNutrients", [])}
        return {
            "calories": nutrients.get("energy", nutrients.get("energy (kcal)")) or None,
            "protein": nutrients.get("protein") or None,
            "fat": nutrients.get("total lipid (fat)") or None,
            "carbs": nutrients.get("carbohydrate, by difference") or None,
        }
    except Exception as e:
        print("USDA lookup failed for", query, ":", e)
        return None

def guess_class_from_folder(name):
    # basic normalization: use lower and underscores
    return name.lower().replace(" ", "_")

def main():
    if not DATA_ROOT.exists():
        print("DATA_ROOT does not exist:", DATA_ROOT)
        print("Please download Food-101 (or point FOOD101_ROOT env) and retry.")
        return

    rows = []
    classes = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
    # For quick experiments, pick a small subset of classes
    classes_to_use = classes[:10]  # change to None or larger set to scale up
    print("Using classes:", classes_to_use)

    for cls in classes_to_use:
        cls_path = DATA_ROOT / cls
        imgs = list(cls_path.glob("*.jpg"))[:500]  # up to 500 per class
        nutrition = None
        # try USDA lookup
        nutrition = usda_lookup(cls.replace("_", " ")) or FALLBACK_NUTRITION.get(guess_class_from_folder(cls))
        for img in imgs:
            rows.append({
                "image_path": str(img),
                "class_name": cls,
                "nutrition_json": json.dumps(nutrition or {})
            })

    # write CSV
    with open(LABEL_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "class_name", "nutrition_json"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Wrote", LABEL_CSV, "with", len(rows), "rows")
    print("Next: create preprocessing script to resize & extract features (STEP 2).")

if __name__ == "__main__":
    main()
