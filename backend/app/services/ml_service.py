# backend/app/services/ml_service.py
"""
ML service (inference wrapper) for FoodScan-X.

This is a deterministic, easily-reproducible stub to:
 - accept a local image path (string)
 - "analyze" it and return a structured `scan_result` dict:
   {
     "detections": [{ "label": "rice", "confidence": 0.98, "portion_grams": 150 }, ...],
     "nutrition_estimate": {"calories": 320, "carbs": 60, "protein": 8, "fat": 4},
     "portion_grams": 150,
     "ayurveda": {
        "rasa": ["sweet"],
        "dosha_effect": {"kapha": "increase", "pitta": "neutral", "vata": "decrease"}
     },
     "meta": {...}
   }

Replace internals with a real model call (TF/PyTorch) later.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, List, Tuple
from pathlib import Path

# If you want to use image libs later, import here:
# from PIL import Image
# import numpy as np

# ---- Helpers ----
def _hash_to_seed(s: str) -> int:
    """Return a small integer seed derived from a string (filename, content hash)."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    # take last 6 hex digits -> int
    return int(h[-6:], 16) % 1000

def _choose_food_by_seed(seed: int) -> Tuple[str, Dict[str, Any]]:
    """
    A deterministic mapping from seed -> (label, nutrition)
    This gives repeatable mock outputs for a given filename.
    """
    foods = [
        ("rice", {"calories_per_100g": 130, "carbs": 28, "protein": 2.7, "fat": 0.3, "rasa": ["sweet"], "dosha": {"kapha": "increase"}}),
        ("salad", {"calories_per_100g": 25, "carbs": 3.6, "protein": 1.4, "fat": 0.2, "rasa": ["bitter"], "dosha": {"kapha": "decrease"}}),
        ("chicken", {"calories_per_100g": 239, "carbs": 0, "protein": 27, "fat": 14, "rasa": ["pungent"], "dosha": {"pitta": "increase"}}),
        ("dal", {"calories_per_100g": 116, "carbs": 20, "protein": 9, "fat": 0.4, "rasa": ["sweet"], "dosha": {"kapha": "increase"}}),
        ("fried_potato", {"calories_per_100g": 320, "carbs": 41, "protein": 3.4, "fat": 15, "rasa": ["salty"], "dosha": {"kapha": "increase", "pitta": "increase"}}),
        ("fruit_bowl", {"calories_per_100g": 60, "carbs": 15, "protein": 0.6, "fat": 0.2, "rasa": ["sweet"], "dosha": {"kapha": "increase"}}),
    ]
    idx = seed % len(foods)
    return foods[idx]

def _estimate_portion_and_nutrition(seed: int, base_nutrition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build portion_grams and nutrition_estimate from base_nutrition and seed.
    We add some deterministic variance so different images show different values.
    """
    # portion grams between 80 and 300 (deterministic from seed)
    portion = 80 + (seed % 221)  # 80..300
    calories = int((base_nutrition["calories_per_100g"] * portion) / 100)
    # approximate macros from per-100g
    carbs = round((base_nutrition.get("carbs", 0) * portion) / 100, 1)
    protein = round((base_nutrition.get("protein", 0) * portion) / 100, 1)
    fat = round((base_nutrition.get("fat", 0) * portion) / 100, 1)

    return {
        "portion_grams": portion,
        "nutrition_estimate": {
            "calories": calories,
            "carbs": carbs,
            "protein": protein,
            "fat": fat,
        },
    }

def _map_to_ayurveda(base_nutrition: Dict[str, Any]) -> Dict[str, Any]:
    """Return a small ayurveda impact object derived from base_nutrition."""
    rasa = base_nutrition.get("rasa", ["sweet"])
    dosha_effect = base_nutrition.get("dosha", {})
    # keep it simple: return fields provided plus a short rationale
    return {
        "rasa": rasa,
        "dosha_effect": dosha_effect,
        "rationale": f"Based on rasa={rasa} and macro profile, predicted dosha changes.",
    }

# ---- Public API ----

def analyze_image(local_image_path: str) -> Dict[str, Any]:
    """
    Analyze an image from local path and return structured scan_result.
    - local_image_path: file path (string) accessible to the server
    """
    # Basic validations
    if not local_image_path or not os.path.exists(local_image_path):
        raise FileNotFoundError(f"Image not found: {local_image_path}")

    # Use file path (or contents) to produce deterministic seed
    # For real model, open image and run inference here
    file_info = Path(local_image_path)
    seed = _hash_to_seed(str(file_info.resolve()))

    # pick a food label and base nutrition
    label, base = _choose_food_by_seed(seed)

    # estimate portion and nutrition
    portion_info = _estimate_portion_and_nutrition(seed, base)

    # build detections (single top detection in stub)
    detections: List[Dict[str, Any]] = [
        {
            "label": label,
            "confidence": round(0.8 + ((seed % 20) / 100.0), 2),  # 0.80 .. 0.99 deterministic
            "portion_grams": portion_info["portion_grams"],
        }
    ]

    ayurveda = _map_to_ayurveda(base)

    # metadata
    meta = {
        "analyzer": "stub_v1",
        "seed": seed,
        "filename": file_info.name,
        "processed_at": int(time.time()),
    }

    scan_result = {
        "detections": detections,
        "portion_grams": portion_info["portion_grams"],
        "nutrition_estimate": portion_info["nutrition_estimate"],
        "ayurveda": ayurveda,
        "meta": meta,
    }

    return scan_result

# small convenience wrapper used by routes to accept file-like objects
def analyze_fileobj(fileobj, dest_dir: str = "/tmp/foodscan_uploads") -> Dict[str, Any]:
    """
    Save incoming file-like object to a deterministic temporary path and analyze it.
    fileobj: file-like object with .read() or bytes (as provided by FastAPI UploadFile)
    dest_dir: local directory where we persist temporarily
    """
    os.makedirs(dest_dir, exist_ok=True)
    # create a deterministic filename from content hash
    content = fileobj.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    file_hash = hashlib.sha256(content).hexdigest()[:16]
    filename = f"scan_{file_hash}.jpg"
    path = os.path.join(dest_dir, filename)
    # write only if not already present (faster iterative dev)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(content)
    # rewind if original fileobj supports it
    try:
        fileobj.seek(0)
    except Exception:
        pass

    # call analyze_image
    return analyze_image(path)
# compatibility wrapper: some routers expect `infer_image`
def infer_image(local_image_path: str) -> Dict[str, Any]:
    """
    Backward-compatible wrapper for older router code expecting infer_image.
    Calls analyze_image and returns the scan_result dict.
    """
    return analyze_image(local_image_path)

# compatibility wrapper for file-like objects if router uses infer_fileobj
def infer_fileobj(fileobj, dest_dir: str = "/tmp/foodscan_uploads") -> Dict[str, Any]:
    """
    Backward-compatible wrapper that accepts a file-like object and returns scan_result.
    """
    return analyze_fileobj(fileobj, dest_dir=dest_dir)
