# backend/app/services/quantum_service.py
"""
Quantum service using Qiskit (with classical fallback).

Provides:
 - submit_quantum_job(input_payload) -> job_id
 - get_job_status(job_id) -> {job_id, status, created_at, result}
 - run_quantum_optimize_sync(features) -> dict (helper for synchronous calls)

Behavior:
 - If qiskit is installed and Aer simulator available, runs a small VQC optimization.
 - Otherwise falls back to a classical, quantum-inspired optimizer (fast numpy-based).
 - Keeps an in-memory job store for prototyping; replace with DB/queue for prod.

IMPORTANT:
 - To use real IBM quantum backends, install qiskit and configure IBMQ account.
"""

from typing import Dict, Any, Optional
import time
import threading
import uuid
import math
import os
import json
import logging

logger = logging.getLogger("quantum_service")
logger.setLevel(logging.INFO)

_jobs: Dict[str, Dict[str, Any]] = {}

# Try import Qiskit; if not available, we will fallback
# Try import Qiskit; if not available, we will fallback (robust across qiskit versions)
# Qiskit 2.x import logic
_QISKIT_AVAILABLE = False
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter
    from qiskit_aer import AerSimulator
    _QISKIT_AVAILABLE = True
    logger.info("Qiskit 2.x + AerSimulator loaded successfully.")
except Exception as e:
    _QISKIT_AVAILABLE = False
    logger.info(f"Qiskit unavailable, fallback enabled: {e}")


# -------------------------
# Feature encoder
# -------------------------
def _encode_features(scan_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert scan_result (nutrition + ayurveda) into a small numeric feature vector dict.
    Keep features small and interpretable:
      - calories_norm: calories / 1000
      - protein_norm: protein / 100
      - carbs_norm: carbs / 200
      - fat_norm: fat / 100
      - kapha_score, pitta_score, vata_score: derived from ayurveda dosha_effect mapping
    """
    nut = scan_result.get("nutrition_estimate", {})
    calories = float(nut.get("calories", 0))
    protein = float(nut.get("protein", 0))
    carbs = float(nut.get("carbs", 0))
    fat = float(nut.get("fat", 0))

    ay = scan_result.get("ayurveda", {})
    dosha = ay.get("dosha_effect", {}) if isinstance(ay, dict) else {}

    def _dosha_val(key):
        v = dosha.get(key)
        if v is None:
            return 0.0
        # map increase->+1, decrease->-1, neutral/other->0
        if isinstance(v, str):
            s = v.lower()
            if "increase" in s:
                return 1.0
            if "decrease" in s:
                return -1.0
        return 0.0

    kapha = _dosha_val("kapha")
    pitta = _dosha_val("pitta")
    vata = _dosha_val("vata")

    features = {
        "calories_norm": calories / 1000.0,
        "protein_norm": protein / 100.0,
        "carbs_norm": carbs / 200.0,
        "fat_norm": fat / 100.0,
        "kapha": kapha,
        "pitta": pitta,
        "vata": vata,
    }
    return features


# -------------------------
# Small VQC (if qiskit available)
# -------------------------
def _build_simple_vqc(num_qubits: int, num_layers: int):
    """
    Build a small parameterized variational circuit.
    We will use Parameter objects; optimizer will evaluate expectation value of Z on first qubit.
    """
    params = []
    qc = QuantumCircuit(num_qubits)
    # create parameterized layers
    for layer in range(num_layers):
        for q in range(num_qubits):
            p = Parameter(f"theta_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        # entangle
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    # measurement will be added by the execution layer (we will compute expectation via statevector or counts)
    return qc, params


def _qiskit_optimize(features: Dict[str, float], maxiter: int = 16):
    """
    Qiskit 2.x execution path using AerSimulator and counts-based expectation.
    Uses a simple random-search optimizer for robustness on varied qiskit installs.

    Returns a dict:
      { "engine": "qiskit_simulator", "risk_score": float, ... }
    """
    if not _QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit not available")

    try:
        # Lazy import of AerSimulator for Qiskit 2.x
        try:
            from qiskit_aer import AerSimulator
        except Exception:
            from qiskit.providers.aer import AerSimulator

        # circuit sizing
        num_features = len(features)
        num_qubits = min(6, max(1, num_features))
        num_layers = 2

        # Build parameterized circuit
        qc = QuantumCircuit(num_qubits)
        params = []
        for layer in range(num_layers):
            for q in range(num_qubits):
                p = Parameter(f"theta_{layer}_{q}")
                params.append(p)
                qc.ry(p, q)
            for q in range(num_qubits - 1):
                qc.cz(q, q + 1)
        qc.measure_all()

        # Prepare simulator backend
        backend = AerSimulator()
        # transpile once (parameterized circuits remain parameterized)
        transpiled = transpile(qc, backend=backend, optimization_level=1)

        # Helper: evaluate a parameter vector -> risk score using counts expectation on qubit-0
        def eval_params(theta_values):
            # map params -> values and assign them to the transpiled circuit
            bind_map = {p: v for p, v in zip(params, theta_values)}
            try:
                bound_circ = transpiled.assign_parameters(bind_map)
            except Exception:
                # fallback: try assign on original qc
                bound_circ = qc.assign_parameters(bind_map)

            # run job on simulator
            job = backend.run(bound_circ, shots=1024)
            result = job.result()
            # counts-based expectation
            try:
                counts = result.get_counts()
            except Exception:
                # sometimes result.get_counts() expects circuit argument
                try:
                    counts = result.get_counts(bound_circ)
                except Exception:
                    counts = {}

            total = sum(counts.values()) if isinstance(counts, dict) else 0
            if total == 0:
                # no counts returned, assume neutral expectation
                return 0.5  # mid-risk

            exp = 0.0
            for bitstr, c in counts.items():
                # Qiskit bitstring order: qubit_n ... qubit_0 (rightmost is qubit-0)
                bit0 = int(bitstr[::-1][0])
                zval = 1.0 if bit0 == 0 else -1.0
                exp += (c / total) * zval
            # map expectation (+1 -> low risk) to risk [0,1]
            risk = (1 - float(exp)) / 2.0
            return max(0.0, min(1.0, risk))

        # Random-search optimizer
        import random
        best_score = float("inf")
        best_theta = None
        dim = len(params) or 1
        for it in range(maxiter):
            theta = [random.uniform(-3.14159, 3.14159) for _ in range(dim)]
            try:
                score = eval_params(theta)
            except Exception:
                score = 0.5  # penalize if evaluation failed
            if score < best_score:
                best_score = score
                best_theta = theta

        recommended_portion_pct = max(0.1, 1.0 - best_score * 0.6)
        safe_frequency = "limit" if best_score > 0.8 else ("1-2 times/week" if best_score > 0.4 else "regular")

        return {
            "engine": "qiskit_simulator",
            "risk_score": float(best_score),
            "recommended_portion_pct": float(recommended_portion_pct),
            "safe_frequency": safe_frequency,
            "notes": "Qiskit AerSimulator counts-based random-search evaluation",
            "raw": {"best_theta": best_theta, "iterations": maxiter},
        }

    except Exception:
        # bubble up so caller logs and falls back
        raise

# -------------------------
# Classical fallback optimizer
# -------------------------
def _classical_optimize(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Fast deterministic 'quantum-inspired' optimizer used when Qiskit is unavailable.
    Produces a risk score and recommended portion by simple rules and small optimization.
    """
    calories = features.get("calories_norm", 0.0) * 1000.0
    kapha = features.get("kapha", 0.0)
    pitta = features.get("pitta", 0.0)
    vata = features.get("vata", 0.0)

    # base risk proportional to calories and kapha; normalize to 0..1 with logistic-like mapping
    base = (calories / 1000.0) * 0.5 + max(0.0, kapha) * 0.4
    risk = 1.0 / (1.0 + math.exp(- (base - 0.2) * 6.0))
    # recommended portion: try to reduce portion proportionally if risk > 0.5
    portion_adj_pct = 1.0 - 0.25 * max(0.0, risk - 0.5)
    # safe frequency heuristic
    if risk < 0.3:
        safe_freq = "daily"
    elif risk < 0.6:
        safe_freq = "3-4 times/week"
    else:
        safe_freq = "1-2 times/week"

    result = {
        "engine": "classical_fallback",
        "risk_score": float(max(0.0, min(1.0, risk))),
        "recommended_portion_pct": float(portion_adj_pct),
        "safe_frequency": safe_freq,
        "notes": "Deterministic classical fallback optimizer.",
        "timestamp": int(time.time()),
    }
    return result


# -------------------------
# Public API: run optimize once (sync)
# -------------------------
def run_quantum_optimize_sync(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the quantum (or fallback) optimizer synchronously and return a result dict.
    """
    features = _encode_features(scan_result)
    if _QISKIT_AVAILABLE:
        try:
            return _qiskit_optimize(features)
        except Exception as e:
            logger.exception("Qiskit run failed, using classical fallback: %s", e)
            return _classical_optimize(features)
    else:
        return _classical_optimize(features)


# -------------------------
# Async job API (keeps compatibility with earlier router)
# -------------------------
def submit_quantum_job(input_payload: Dict[str, Any]) -> str:
    """
    Create an async job entry and process it in background thread.
    input_payload should contain 'scan_result' or 'classical_result' etc.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "payload": input_payload,
        "result": None,
        "created_at": int(time.time()),
    }

    def _worker(jid: str):
        try:
            _jobs[jid]["status"] = "running"
            payload = _jobs[jid]["payload"] or {}
            scan_result = payload.get("scan_result") or payload.get("classical_result") or {}
            # run the sync optimizer (qiskit or fallback)
            res = run_quantum_optimize_sync(scan_result)
            # attach some contextual recommendations (eg: recommended_portion_grams)
            # if classical_result present and has portion_grams, apply adjustment
            classical = payload.get("classical_result") or {}
            portion_grams = classical.get("portion_grams") or scan_result.get("portion_grams") or None
            if portion_grams and res.get("risk_score") is not None:
                # naive mapping: reduce portion by risk_score*15% (example)
                recommended_portion = int(max(20, portion_grams * (1.0 - res["risk_score"] * 0.15)))
            else:
                recommended_portion = portion_grams

            # build final enriched result
            final = {
                "job_id": jid,
                "quantum_engine_result": res,
                "recommended_portion_grams": recommended_portion,
                "explanation": res.get("notes", ""),
                "timestamp": int(time.time()),
            }
            _jobs[jid]["result"] = final
            _jobs[jid]["status"] = "done"
        except Exception as e:
            logger.exception("Quantum job %s failed: %s", jid, e)
            _jobs[jid]["status"] = "error"
            _jobs[jid]["result"] = {"error": str(e)}

    t = threading.Thread(target=_worker, args=(job_id,))
    t.daemon = True
    t.start()
    return job_id


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    job = _jobs.get(job_id)
    if not job:
        return None
    # shallow copy for safety
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "result": job["result"],
    }


# -------------------------
# Utilities: load/save small model params (placeholder)
# -------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "quantum_model.json")


def save_model_params(params: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        with open(_MODEL_PATH, "w") as fh:
            json.dump(params, fh, indent=2)
    except Exception:
        logger.exception("Failed to save quantum model params")


def load_model_params() -> Dict[str, Any]:
    try:
        if os.path.exists(_MODEL_PATH):
            with open(_MODEL_PATH, "r") as fh:
                return json.load(fh)
    except Exception:
        logger.exception("Failed to load quantum model params")
    return {}
