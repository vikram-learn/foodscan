# prepare_quantum_data.py
# Convert classical features.npy / labels.npy -> quantum-ready inputs (angles, normalized)
import os, json
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "quantum")
os.makedirs(OUT_DIR, exist_ok=True)

def load_features():
    f = os.path.join(DATA_DIR, "features.npy")
    y = os.path.join(DATA_DIR, "labels.npy")
    X = np.load(f)
    y = np.load(y)
    return X, y

def normalize_features(X):
    # simple min-max per feature to [-pi, pi] for angle encoding
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    Xn = (X - mins) / rng
    angles = Xn * (2 * np.pi) - np.pi
    return angles

def save_quantum_data(angles, labels):
    np.save(os.path.join(OUT_DIR, "angles.npy"), angles)
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels)
    meta = {"angles_shape": list(angles.shape), "labels_shape": list(labels.shape)}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved quantum-ready data to", OUT_DIR)

if __name__ == "__main__":
    X, y = load_features()
    angles = normalize_features(X)
    save_quantum_data(angles, y)
