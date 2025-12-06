#!/usr/bin/env python3
"""
train_quantum_vqc.py

Train a tiny Variational Quantum Circuit (VQC) using qiskit AerSimulator if available,
otherwise perform a simple classical "random search" fallback. Save best params & metadata.

Outputs:
  backend/models/quantum/best_vqc.json
  backend/models/quantum/training_history.json
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]  # backend/
DATA_QUANTUM = ROOT / "data" / "quantum"
OUT_DIR = ROOT / "models" / "quantum"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# hyperparams
N_QUBITS = int(os.environ.get("QC_N_QUBITS", 8))
N_LAYERS = int(os.environ.get("QC_N_LAYERS", 2))
N_ITER = int(os.environ.get("QC_N_ITER", 32))
SEED = int(os.environ.get("QC_SEED", 42))

np.random.seed(SEED)

# Try qiskit imports
_QISKIT_OK = False
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.providers.aer import AerSimulator
    from qiskit.quantum_info import Statevector
    from qiskit import transpile
    _QISKIT_OK = True
    print("Qiskit AerSimulator available -- will use simulator.")
except Exception as e:
    print("Qiskit not available or Aer missing; will do classical fallback.", e)
    _QISKIT_OK = False

def load_qc_data():
    X = np.load(DATA_QUANTUM / "qc_inputs.npy")
    y = np.load(DATA_QUANTUM / "qc_labels.npy")
    return X, y

def build_vqc(n_qubits=N_QUBITS, n_layers=N_LAYERS):
    params = ParameterVector("theta", length=n_qubits * n_layers)
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
    # measure implicitly via statevector expectation on Z of first qubit (we'll compute)
    return qc, params

def vqc_forward_counts(qc, param_values, angles, backend):
    """Build param-bound circuit by adding angle-encoding rotations then evaluate."""
    # angles: shape (n_qubits,)
    # Build full circuit: encode angles then append vqc
    from qiskit import QuantumCircuit
    n = len(angles)
    enc = QuantumCircuit(n)
    for i, a in enumerate(angles):
        enc.ry(a, i)
    full = enc.compose(qc)
    # bind params
    bound = full.bind_parameters(param_values)
    # run statevector or counts
    sv = Statevector(bound)
    # use expectation of Z on first qubit: <Z0> in [-1,1]
    z_expect = sv.expectation_value("Z", [0])
    # map expectation to a scalar score in [0,1]
    score = float((1 - z_expect) / 2.0)
    return score

def evaluate_params_on_set(qc, params, theta, X, y, backend=None):
    # For simplicity do a per-sample eval returning class by threshold on score,
    # this is a toy classifier: map score->class index by binning [0,1] into C bins.
    C = len(np.unique(y))
    preds = []
    for angles in X:
        score = vqc_forward_counts(qc, theta, angles, backend)
        # assign class by dividing into C bins
        cls = int(np.floor(score * C))
        if cls >= C:
            cls = C - 1
        preds.append(cls)
    return np.array(preds)

def classical_random_search(X_train, y_train, X_val, y_val, n_iter=N_ITER):
    best = None
    history = []
    C = len(np.unique(y_train))
    n_params = N_QUBITS * N_LAYERS
    for i in range(n_iter):
        theta = np.random.uniform(-np.pi, np.pi, size=(n_params,))
        # naive scoring: sum of dot products between theta and mean angles per class
        # This is a quantum-inspired heuristic — keep it simple
        preds = []
        for angles in X_val:
            score = (np.tanh(np.dot(theta[:N_QUBITS], angles)))  # in (-1,1)
            cls = int(np.floor(((score + 1) / 2.0) * C))
            cls = max(0, min(C - 1, cls))
            preds.append(cls)
        acc = accuracy_score(y_val, preds)
        history.append({"iter": i, "acc": float(acc)})
        if best is None or acc > best["acc"]:
            best = {"acc": float(acc), "theta": theta.tolist(), "iter": i}
    return best, history

def main():
    X, y = load_qc_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    meta = {"n_qubits": N_QUBITS, "n_layers": N_LAYERS, "n_iter": N_ITER, "seed": SEED, "n_classes": len(np.unique(y))}
    if _QISKIT_OK:
        qc, params = build_vqc(N_QUBITS, N_LAYERS)
        # Simple random search over parameter space (COBYLA could be used but keep simple)
        n_params = len(params)
        best = None
        hist = []
        for it in range(N_ITER):
            theta = np.random.uniform(-np.pi, np.pi, size=(n_params,))
            try:
                preds = evaluate_params_on_set(qc, params, theta, X_val, y_val, backend=None)
                acc = accuracy_score(y_val, preds)
            except Exception as e:
                acc = 0.0
            hist.append({"iter": it, "acc": float(acc)})
            if best is None or acc > best["acc"]:
                best = {"acc": float(acc), "theta": theta.tolist(), "iter": it}
        best_file = OUT_DIR / "best_vqc.json"
        with open(OUT_DIR / "training_history.json", "w") as f:
            json.dump(hist, f, indent=2)
        with open(best_file, "w") as f:
            json.dump({"meta": meta, "best": best}, f, indent=2)
        print("Saved best_vqc.json ->", best_file)
    else:
        print("Qiskit not available, running classical random-search fallback")
        best, hist = classical_random_search(X_train, y_train, X_val, y_val, n_iter=N_ITER)
        with open(OUT_DIR / "training_history.json", "w") as f:
            json.dump(hist, f, indent=2)
        with open(OUT_DIR / "best_vqc.json", "w") as f:
            json.dump({"meta": meta, "best": best}, f, indent=2)
        print("Saved classical-best_vqc.json ->", OUT_DIR / "best_vqc.json")

if __name__ == "__main__":
    main()
