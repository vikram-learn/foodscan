#!/usr/bin/env python3
"""
train_quantum_vqc.py

Small VQC trainer that tries to use Qiskit + Aer where available.
This version handles Qiskit API differences for parameter binding by trying:
 - QuantumCircuit.bind_parameters
 - QuantumCircuit.assign_parameters

If quantum primitives fail it falls back to a deterministic classical
random-search optimizer and saves a JSON result to models/quantum/best_vqc.json.
"""

import os
import json
import time
import math
import random
import traceback
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "quantum"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "best_vqc.json"

# small logging helper
def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# --------------------------
# try import Qiskit / Aer (robust)
# --------------------------
_QISKIT_AVAILABLE = False
_AER_AVAILABLE = False
_qiskit = None
_AerCompat = None

try:
    import qiskit as _qiskit
    _QISKIT_AVAILABLE = True
    # check possible Aer import paths
    try:
        # preferred: qiskit.providers.aer (may be missing in some installs)
        from qiskit.providers.aer import AerSimulator as _AerSim  # type: ignore
        _AER_AVAILABLE = True
    except Exception:
        try:
            # fallback package name
            from qiskit_aer import AerSimulator as _AerSim  # type: ignore
            _AER_AVAILABLE = True
        except Exception:
            _AerSim = None
            _AER_AVAILABLE = False
    _qiskit = _qiskit
except Exception:
    _QISKIT_AVAILABLE = False
    _AER_AVAILABLE = False
    _qiskit = None
    _AerSim = None

log("MODULE FLAGS:", {"_QISKIT_AVAILABLE": _QISKIT_AVAILABLE, "_AER_AVAILABLE": _AER_AVAILABLE})

# --------------------------
# helpers
# --------------------------
def save_result(res: dict):
    res_with_ts = res.copy()
    res_with_ts.setdefault("raw", {})
    res_with_ts["raw"].setdefault("timestamp", int(time.time()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(res_with_ts, fh, indent=2)
    log(f"Saved classical-best_vqc.json -> {OUT_FILE}")

def _classical_fallback(features, seed=0):
    random.seed(seed or int(time.time()))
    # simple deterministic heuristic result for prototyping
    risk = round(max(0.0, min(1.0, 1.0 - (features.get("calories_norm", 0) * 2.0))), 3)
    rec_pct = round(max(0.0, min(1.0, 1.0 - 0.1 * features.get("kapha", 0))), 4)
    return {
        "engine": "classical_fallback",
        "risk_score": risk,
        "recommended_portion_pct": rec_pct,
        "safe_frequency": "regular" if risk < 0.5 else "1-2 times/week",
        "notes": "Deterministic classical fallback optimizer.",
        "raw": {"engine": "classical_fallback", "notes": "fast heuristic", "timestamp": int(time.time())},
    }

def _encode_features(scan_result):
    nut = scan_result.get("nutrition_estimate", {})
    calories = float(nut.get("calories", 0))
    protein = float(nut.get("protein", 0))
    carbs = float(nut.get("carbs", 0))
    fat = float(nut.get("fat", 0))
    ay = scan_result.get("ayurveda", {}) or {}
    dosha = ay.get("dosha_effect", {}) if isinstance(ay, dict) else {}
    def dval(k):
        v = dosha.get(k)
        if isinstance(v, str):
            s = v.lower()
            if "increase" in s: return 1.0
            if "decrease" in s: return -1.0
        return 0.0
    return {
        "calories_norm": calories / 1000.0,
        "protein_norm": protein / 100.0,
        "carbs_norm": carbs / 200.0,
        "fat_norm": fat / 100.0,
        "kapha": dval("kapha"),
        "pitta": dval("pitta"),
        "vata": dval("vata"),
    }

# robust param binder for various qiskit versions
def bind_parameters_robust(qc, mapping):
    """
    Try multiple ways to bind parameter values into a QuantumCircuit.
    Returns a bound circuit (may be a new object) or raises RuntimeError.
    """
    # prefer bind_parameters if present
    if hasattr(qc, "bind_parameters"):
        try:
            return qc.bind_parameters(mapping)
        except Exception as e:
            # continue to other methods
            log("bind_parameters exists but failed:", e)
    # try assign_parameters (returns new circuit on many Qiskit versions)
    if hasattr(qc, "assign_parameters"):
        try:
            return qc.assign_parameters(mapping)
        except Exception as e:
            log("assign_parameters exists but failed:", e)
    # try assigning parameters by name with qiskit.quantum_info.Statevector usage is not supported here
    raise RuntimeError("QuantumCircuit parameter binding not supported in this Qiskit version")

# --------------------------
# Qiskit VQC implementation (best-effort)
# --------------------------
def _qiskit_optimize(features, maxiter=32, shots=1024):
    """
    Try to run a small VQC using Qiskit/Aer. If any step fails due to API variance,
    raise RuntimeError so caller can fallback.
    """
    if not (_QISKIT_AVAILABLE and _AER_AVAILABLE):
        raise RuntimeError("Qiskit/Aer not available")
    # local imports
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        # prefer AerSimulator class already discovered
        try:
            from qiskit.providers.aer import AerSimulator as QiskitAerSim
        except Exception:
            from qiskit_aer import AerSimulator as QiskitAerSim
    except Exception as e:
        raise RuntimeError("Qiskit primitives not available: " + str(e))

    # very small VQC: 3 qubits, 2 layers (adjustable)
    num_qubits = 3
    num_layers = 2
    params = []
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        for q in range(num_qubits):
            p = Parameter(f"th_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    # Add a measurement-ish primitive by returning expectation of Z on qubit 0 via counts mapping
    # We'll run shots and interpret counts to approximate expectation.
    # Build transpiled base circuit (no measurement instructions); binding will produce circuits executable.
    qc_template = qc

    # create a simulator instance
    try:
        sim = QiskitAerSim()
    except Exception as e:
        raise RuntimeError("Failed to create AerSimulator instance: " + str(e))

    # evaluation function: given theta (list) -> estimated objective (minimize)
    def eval_theta(theta_values):
        try:
            mapping = {p: float(v) for p, v in zip(params, theta_values)}
            bound_qc = bind_parameters_robust(qc_template, mapping)
            # if bound_qc has no measurements, append measurement for counts
            run_qc = bound_qc.copy()
            run_qc.measure_all()
            # execute
            job = sim.run(run_qc, shots=shots)
            res = job.result()
            # try counts -> expectation of Z on qubit 0
            try:
                counts = res.get_counts(run_qc)
            except Exception:
                # some Qiskit versions may return dict keyed differently
                counts = {}
                try:
                    counts = res.get_counts()
                except Exception:
                    raise RuntimeError("Failed to extract counts from simulator result")
            # compute expectation of Z on qubit 0 from counts
            total = sum(counts.values())
            if total == 0:
                raise RuntimeError("No counts returned from simulator")
            exp = 0.0
            for bitstr, c in counts.items():
                # bit order in qiskit counts is little-endian by default (rightmost is qubit 0)
                if len(bitstr) == 0:
                    continue
                # rightmost character corresponds to qubit 0
                b0 = bitstr[-1]
                z = 1 if b0 == "0" else -1
                exp += z * (c / total)
            # objective: we want to minimize risk -> derive a synthetic risk from expectation
            # map exp in [-1,1] to risk in [0,1]
            risk = float((1.0 - exp) / 2.0)
            return risk
        except Exception as e:
            raise RuntimeError("Quantum run failed during objective evaluation: " + str(e))

    # simple optimizer: try random restarts + keep best
    best_theta = None
    best_score = float("inf")
    tries = max(8, maxiter)
    for it in range(tries):
        # random init in [-pi, pi]
        theta0 = [random.uniform(-math.pi, math.pi) for _ in params]
        try:
            score = eval_theta(theta0)
            if score < best_score:
                best_score = score
                best_theta = list(theta0)
        except Exception as e:
            # allow some runs to fail
            log("Quantum eval iteration failed:", e)
            # continue searching
            continue
    if best_theta is None:
        raise RuntimeError("Quantum evaluation failed (random-search): no successful run")
    # package result
    return {
        "engine": "qiskit_simulator",
        "risk_score": round(best_score, 6),
        "recommended_portion_pct": round(max(0.0, 1.0 - best_score), 6),
        "safe_frequency": "regular" if best_score < 0.5 else "1-2 times/week",
        "notes": "Qiskit AerSimulator counts-based random-search evaluation",
        "raw": {"best_theta": best_theta, "iterations": tries},
    }

# --------------------------
# Public train runner
# --------------------------
def train_vqc(maxiter=32):
    log("Starting VQC trainer")
    # check prepared data location (for your project this may be created by prepare_quantum_data)
    qdir = Path(__file__).resolve().parents[1] / "models" / "quantum"
    qdir.mkdir(parents=True, exist_ok=True)

    # simple synthetic features if no prepared dataset
    # features shape: list of dicts with numeric keys used by _classical_fallback/_qiskit_optimize
    # For quick prototyping, create a few synthetic feature vectors:
    features = [{"calories_norm": random.uniform(0.1, 0.6), "protein_norm": random.uniform(0.01, 0.2),
                 "carbs_norm": random.uniform(0.05, 0.5), "fat_norm": random.uniform(0.01, 0.2),
                 "kapha": random.choice([0.0, 1.0, -1.0])} for _ in range(32)]

    # try quantum path
    if _QISKIT_AVAILABLE and _AER_AVAILABLE:
        log("Attempting Qiskit/Aer optimization...")
        try:
            # encode a single representative feature for the call
            example = features[0]
            qal = _qiskit if _qiskit is not None else {}
            res = _qiskit_optimize(example, maxiter=maxiter)
            log("Qiskit optimization finished.")
            save_result(res)
            return res
        except Exception as e:
            log("Qiskit optimization failed — falling back to classical. Reason:")
            traceback.print_exc()
            res = _classical_fallback(features[0], seed=42)
            save_result(res)
            return res
    else:
        log("Qiskit not available, running classical random-search fallback")
        res = _classical_fallback(features[0], seed=42)
        save_result(res)
        return res

if __name__ == "__main__":
    result = train_vqc(maxiter=32)
    log("Result summary:", json.dumps(result, indent=2))
