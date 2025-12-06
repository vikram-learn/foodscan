# backend/ml/train_classical.py
"""
Train a small classical classifier on the synthetic Food dataset.

What it does:
 - loads features.npy and labels.npy from backend/data/synthetic
 - splits into train / val (stratified)
 - trains a LogisticRegression (fast, deterministic) + a RandomForest (stronger)
 - evaluates accuracy, classification report
 - saves the best model (by validation accuracy) to backend/models/classical_model.joblib
 - prints short instructions about next steps

Why:
 - gives a baseline for comparison with quantum-inspired or quantum pipelines
 - outputs a saved model file you can load from FastAPI or from frontend tests

How to run:
 - open this file in VS Code and run it inside your Python venv for the project.
 - it will print metrics and save the model file.

Notes:
 - This script uses scikit-learn and joblib. If you don't have them installed in the project's venv,
   install scikit-learn and joblib in the environment before running.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

# --- config
ROOT = Path(__file__).resolve().parents[1]     # backend/ml -> go up to backend
DATA_DIR = ROOT / "data" / "synthetic"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = OUT_DIR / "classical_model.joblib"
LABEL_MAP_FILE = DATA_DIR / "label_map.json"

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.2  # fraction of train used as validation

# --- load data
def load_data():
    feats = np.load(DATA_DIR / "features.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    # optional: load label_map
    label_map = {}
    if LABEL_MAP_FILE.exists():
        label_map = json.loads(LABEL_MAP_FILE.read_text())
    return feats, labels, label_map

# --- training
def train_and_evaluate(X_train, y_train, X_val, y_val):
    models = {
        "logreg": LogisticRegression(max_iter=200, random_state=RANDOM_SEED),
        "rf": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    best_model = None
    best_acc = -1.0
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        results[name] = {
            "accuracy": float(acc),
            "report": classification_report(y_val, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_val, preds).tolist(),
        }
        print(f"Model {name} — val accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model

    return best_model, results

def main():
    X, y, label_map = load_data()
    print("Data shapes:", X.shape, y.shape)
    # First split: test vs rest
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # Second split: train vs val (from rest)
    val_frac_of_rest = VALID_SIZE
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac_of_rest, random_state=RANDOM_SEED, stratify=y_rest
    )

    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    best_model, results = train_and_evaluate(X_train, y_train, X_val, y_val)

    # final eval on test
    test_preds = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nBest model test accuracy: {test_acc:.4f}")
    print("Classification report (test):")
    print(classification_report(y_test, test_preds))

    # save model + metadata
    save_payload = {
        "model_path": str(MODEL_OUT),
        "scikit_model": type(best_model).__name__,
        "test_accuracy": float(test_acc),
        "label_map": label_map,
        "notes": "Trained on synthetic dataset for quick iteration"
    }
    joblib.dump(best_model, MODEL_OUT)
    (OUT_DIR / "classical_model_meta.json").write_text(json.dumps(save_payload, indent=2))
    print("\nSaved model to:", MODEL_OUT)
    print("Saved metadata to:", OUT_DIR / "classical_model_meta.json")
    print("\nNext steps:")
    print(" - (a) load this model from your backend API to serve predictions")
    print(" - (b) run prepare_quantum_data.py (I can provide it) to convert features -> quantum-ready inputs")
    print(" - (c) later, retrain on full Food-101 once download finishes")

if __name__ == "__main__":
    main()
