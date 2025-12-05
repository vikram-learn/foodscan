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
_QISKIT_AVAILABLE = False
try:
    # qiskit-terra and qiskit-aer are the main dependencies for simulation
    from qiskit import QuantumCircuit, Aer, transpile, execute
    from qiskit.circuit import Parameter
    from qiskit.utils import QuantumInstance
    from qiskit.algorithms.optimizers import COBYLA
    _QISKIT_AVAILABLE = True
    logger.info("Qiskit imported: using quantum simulator where possible.")
except Exception:
    _QISKIT_AVAILABLE = False
    logger.info("Qiskit not available — using classical fallback optimizer.")


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


def _qiskit_optimize(features: Dict[str, float], maxiter: int = 30) -> Dict[str, Any]:
    """
    Run a simple variational loop using Qiskit's Aer simulator.
    Objective: produce a scalar "risk" we minimize. For demo we construct a small Hamiltonian
    where higher calories and kapha increase risk.
    """
    # pull a compact numeric vector from features
    v = [
        features.get("calories_norm", 0.0),
        features.get("protein_norm", 0.0),
        features.get("carbs_norm", 0.0),
        features.get("fat_norm", 0.0),
        features.get("kapha", 0.0),
        features.get("pitta", 0.0),
        features.get("vata", 0.0),
    ]
    # choose number of qubits = min(4, len(v))
    num_qubits = min(4, len(v))
    num_layers = 2

    qc, params = _build_simple_vqc(num_qubits, num_layers)

    # Prepare quantum instance
    try:
        backend = Aer.get_backend("aer_simulator_statevector")
    except Exception:
        backend = Aer.get_backend("aer_simulator")
    qi = QuantumInstance(backend=backend, shots=512)

    # map initial parameters deterministically from features
    import numpy as np

    init = np.array([(sum(v) % 1.0) + 0.1 * i for i in range(len(params))], dtype=float)

    # define objective function: run circuit with given parameters, get expectation on Z of qubit 0,
    # then combine with features to produce "risk" scalar.
    def objective(x):
        # set parameter values
        bind_map = {p: float(xi) for p, xi in zip(params, x)}
        bound_qc = qc.bind_parameters(bind_map)
        # get statevector and compute expectation of Z on qubit 0
        try:
            # prefer statevector for exact expectation if available
            sv_backend = Aer.get_backend("aer_simulator_statevector")
            job = execute(bound_qc, backend=sv_backend)
            result = job.result()
            sv = result.get_statevector(bound_qc)
            # compute expectation of Z on qubit 0
            # statevector is complex amplitudes; compute expectation directly
            exp_z = 0.0
            for idx, amp in enumerate(sv):
                prob = abs(amp) ** 2
                # bit 0 is least-significant bit in index ordering
                bit0 = (idx >> 0) & 1
                z = 1 if bit0 == 0 else -1
                exp_z += prob * z
        except Exception:
            # fallback: run shots and compute z expectation from counts
            shots_backend = Aer.get_backend("aer_simulator")
            bound_qc_measure = bound_qc.copy()
            bound_qc_measure.save_statevector()
            job = execute(bound_qc_measure, backend=shots_backend, shots=256)
            res = job.result()
            try:
                sv = res.get_statevector(bound_qc_measure)
                exp_z = 0.0
                for idx, amp in enumerate(sv):
                    prob = abs(amp) ** 2
                    bit0 = (idx >> 0) & 1
                    z = 1 if bit0 == 0 else -1
                    exp_z += prob * z
            except Exception:
                # last fallback: random-ish estimate
                exp_z = 0.0

        # combine with simple classical linear model: high calories & kapha -> higher risk
        calories = features.get("calories_norm", 0.0)
        kapha = features.get("kapha", 0.0)
        # map exp_z (-1..1) to 0..1
        qscore = (exp_z + 1.0) / 2.0
        # risk estimate in 0..1
        risk = 0.4 * calories + 0.3 * kapha + 0.3 * qscore
        # objective: we return risk to minimize
        return float(risk)

    # Run a small classical optimizer (COBYLA) over parameters
    try:
        optimizer = COBYLA(maxiter=maxiter)
        # initial guess
        x0 = init
        res = optimizer.optimize(num_vars=len(params), objective_function=objective, initial_point=x0)
        params_opt = res[0] if isinstance(res, tuple) else res
        risk_val = objective(params_opt)
    except Exception as e:
        logger.exception("Qiskit optimization failed, falling back to quick eval: %s", e)
        # fallback behavior: compute objective at init
        import numpy as np
        risk_val = objective(init)
        params_opt = init.tolist() if hasattr(init, "tolist") else list(init)

    # Build result dict
    result = {
        "engine": "qiskit_simulator",
        "risk_score": float(max(0.0, min(1.0, risk_val))),
        "params": [float(p) for p in params_opt] if hasattr(params_opt, "__iter__") else [],
        "notes": "Variational circuit optimized on simulator (small).",
        "timestamp": int(time.time()),
    }
    return result


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
