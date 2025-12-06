# backend/app/services/classical_service.py
"""
Classical model loader + prediction helpers.

Functions:
 - load_model() -> loads joblib model + metadata (cached)
 - prepare_features(scan_result) -> vector suitable for the model (np.array)
 - predict_from_scan_result(scan_result) -> dict with 'pred' and 'probs'
"""

from pathlib import Path
import json
import threading
import logging

logger = logging.getLogger("classical_service")
logger.setLevel(logging.INFO)

# lazy imports so file imports don't fail if numpy/joblib missing
_np = None
_joblib = None

_MODEL_LOCK = threading.Lock()
_MODEL = None
_META = None

# default paths (relative to backend/)
BASE = Path(__file__).resolve().parents[2]  # backend/
MODEL_PATH = BASE / "models" / "classical_model.joblib"
META_PATH = BASE / "models" / "classical_model_meta.json"


def _ensure_deps():
    global _np, _joblib
    if _np is None:
        import numpy as _np  # type: ignore
    if _joblib is None:
        import joblib as _joblib  # type: ignore


def load_model():
    """
    Load model + metadata once and cache in module globals.
    Returns (model, meta) or (None, None) if not available.
    """
    global _MODEL, _META
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL, _META

        try:
            _ensure_deps()
            if not MODEL_PATH.exists():
                logger.warning("Classical model file missing: %s", MODEL_PATH)
                return None, None

            _MODEL = _joblib.load(MODEL_PATH)
            if META_PATH.exists():
                try:
                    _META = json.loads(META_PATH.read_text(encoding="utf-8"))
                except Exception:
                    _META = None
            logger.info("Classical model loaded from %s", MODEL_PATH)
            return _MODEL, _META
        except Exception as e:
            logger.exception("Failed to load classical model: %s", e)
            return None, None


def prepare_features(scan_result: dict):
    """
    Convert your scan_result to the same feature vector used in training.
    This must match train_classical.py -> features layout.

    This implementation:
     - builds the canonical 7-item feature list we used originally, then
     - loads model metadata (if any) and pads / trims to the model's expected length.
     - falls back to a sensible DEFAULT_FEATURE_LENGTH when metadata missing.

    Returns: numpy array shaped (1, n_features) or raises on missing deps.
    """
    _ensure_deps()
    np = _np

    nut = scan_result.get("nutrition_estimate", {}) or {}
    calories = float(nut.get("calories", 0.0))
    protein = float(nut.get("protein", 0.0))
    carbs = float(nut.get("carbs", 0.0))
    fat = float(nut.get("fat", 0.0))

    ay = scan_result.get("ayurveda", {}) or {}
    dosha = ay.get("dosha_effect", {}) or {}

    def dosha_val(k):
        v = dosha.get(k)
        if v is None:
            return 0.0
        if isinstance(v, str):
            s = v.lower()
            if "increase" in s:
                return 1.0
            if "decrease" in s:
                return -1.0
        return 0.0

    kapha = dosha_val("kapha")
    pitta = dosha_val("pitta")
    vata = dosha_val("vata")

    # Base features (the 7 we currently derive)
    features = [
        calories / 1000.0,   # calories_norm
        protein / 100.0,     # protein_norm
        carbs / 200.0,       # carbs_norm
        fat / 100.0,         # fat_norm
        kapha,
        pitta,
        vata,
    ]

    # Load model metadata to know expected feature length (if present)
    _, meta = load_model()

    # Default target length (match your training: 24)
    DEFAULT_FEATURE_LENGTH = 24

    if meta and isinstance(meta, dict) and "feature_length" in meta:
        try:
            target_len = int(meta["feature_length"])
        except Exception:
            target_len = DEFAULT_FEATURE_LENGTH
    else:
        target_len = DEFAULT_FEATURE_LENGTH

    # pad with zeros if shorter, or trim if longer
    if len(features) < target_len:
        features = features + [0.0] * (target_len - len(features))
    elif len(features) > target_len:
        features = features[:target_len]

    return np.array(features, dtype=float).reshape(1, -1)


def predict_from_scan_result(scan_result: dict):
    """
    Top-level helper used by the API.
    Returns a dict: { "model_available": bool, "pred": <label or None>,
                     "probs": [probabilities] or None, "meta": ... }
    """
    try:
        _ensure_deps()
    except Exception as e:
        logger.exception("Missing numpy/joblib: %s", e)
        return {"model_available": False, "error": "missing_deps"}

    model, meta = load_model()
    if model is None:
        return {"model_available": False, "error": "model_not_found"}

    try:
        X = prepare_features(scan_result)
        # many scikit models have predict_proba; guard if not present
        pred = None
        probs = None
        try:
            pred_idx = model.predict(X)
            # convert to simple Python types
            pred = int(pred_idx[0]) if hasattr(pred_idx[0], "__int__") else pred_idx[0]
        except Exception:
            pred = str(model.predict(X)[0])

        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X)[0]
                probs = [float(x) for x in p]
            except Exception:
                probs = None

        return {
            "model_available": True,
            "pred": pred,
            "probs": probs,
            "meta": meta or {}
        }
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return {"model_available": False, "error": "prediction_failed", "exception": str(e)}
