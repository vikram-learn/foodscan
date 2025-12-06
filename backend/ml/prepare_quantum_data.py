#!/usr/bin/env python3
"""
prepare_quantum_data.py

Reads classical features (features.npy) and labels (labels.npy) from:
  backend/data/synthetic/  (and optionally backend/data/food101/* when ready)

Produces:
  backend/data/quantum/qc_inputs.npy   -- shape (N, n_qubits) or (N, n_angles)
  backend/data/quantum/qc_labels.npy   -- integer labels (N,)
  backend/data/quantum/quantum_meta.json

Encoding:
 - angle encoding: map normalized feature vectors -> rotation angles in [0, pi]
 - If features > qubits*angles, we use PCA to reduce to n_qubits dims (default 8)
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # backend/
DATA_QUANTUM = ROOT / "data" / "quantum"
DATA_QUANTUM.mkdir(parents=True, exist_ok=True)

# Candidates: synthetic path and food101 path
CANDIDATES = [
    ROOT / "data" / "synthetic",
    ROOT / "data" / "food101",
]

# Parameters
N_QUBITS = int(os.environ.get("QC_N_QUBITS", 8))   # default 8 qubits
ANGLE_RANGE = np.pi  # map normalized features into [0, pi]

def load_classical_features():
    # Look for features.npy and labels.npy in candidates
    for p in CANDIDATES:
        if (p / "features.npy").exists() and (p / "labels.npy").exists():
            print("Using classical features from:", p)
            X = np.load(p / "features.npy")
            y = np.load(p / "labels.npy")
            return X, y, str(p)
    raise FileNotFoundError("No classical features found in expected locations. Run train_classical.py or place features.npy + labels.npy in backend/data/*")

def prepare_angles(X, n_qubits=N_QUBITS):
    """
    Normalize -> reduce (PCA if needed) -> scale to [0,1] -> multiply by ANGLE_RANGE
    Returns shape (N, n_qubits)
    """
    N, D = X.shape
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if D > n_qubits:
        pca = PCA(n_components=n_qubits)
        Xr = pca.fit_transform(Xs)
        used_method = f"PCA {D}->{n_qubits}"
    else:
        # pad with zeros if D < n_qubits
        Xr = np.zeros((N, n_qubits), dtype=float)
        Xr[:, :D] = Xs
        used_method = f"pad {D}->{n_qubits}"

    # Normalize each feature column to [0,1] (min-max)
    Xmin = Xr.min(axis=0, keepdims=True)
    Xmax = Xr.max(axis=0, keepdims=True)
    denom = (Xmax - Xmin)
    denom[denom == 0] = 1.0
    Xnorm = (Xr - Xmin) / denom

    angles = Xnorm * ANGLE_RANGE
    return angles, {"method": used_method, "orig_shape": (N, D)}

def main():
    X, y, src = load_classical_features()
    angles, meta_reduce = prepare_angles(X, N_QUBITS)

    np.save(DATA_QUANTUM / "qc_inputs.npy", angles)
    np.save(DATA_QUANTUM / "qc_labels.npy", y)

    meta = {
        "source": src,
        "n_qubits": N_QUBITS,
        "n_samples": int(angles.shape[0]),
        "angle_range": float(ANGLE_RANGE),
        "reduction": meta_reduce,
        "created_by": "prepare_quantum_data.py",
    }
    with open(DATA_QUANTUM / "quantum_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved qc_inputs.npy ->", DATA_QUANTUM / "qc_inputs.npy")
    print("Saved qc_labels.npy ->", DATA_QUANTUM / "qc_labels.npy")
    print("Saved quantum_meta.json ->", DATA_QUANTUM / "quantum_meta.json")
    print("Done.")

if __name__ == "__main__":
    main()
