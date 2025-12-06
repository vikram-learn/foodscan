# backend/app/services/quantum_service.py
"""
Quantum service for FoodScan-X.

Features:
 - try to use Qiskit + Aer (AerSimulator) if available
 - fall back to a deterministic classical optimizer when quantum engine is not usable
 - provide sync optimizer (run_quantum_optimize_sync) and async submit/status primitives
 - defensive: adapt to qiskit import path differences and API differences (bind/assign, AerSimulator location)

Public functions:
 - run_quantum_optimize_sync(input_payload_or_features) -> dict
 - submit_quantum_job(input_payload) -> dict (job_id + immediate fallback)
 - get_job_status(job_id) -> dict
"""

from typing import Dict, Any, Optional, Tuple
import time
import uuid
import logging
import math
import random
import threading

logger = logging.getLogger("quantum_service")
logger.setLevel(logging.INFO)

# in-memory job store for demo/prototype
_jobs: Dict[str, Dict[str, Any]] = {}

# Flags that other code may inspect
_QISKIT_AVAILABLE = False
_AER_AVAILABLE = False

# We'll try to import Qiskit / Aer in the most compatible way.
# Some installations expose Aer under 'qiskit.providers.aer', others expose it as 'qiskit_aer'.
_qiskit = None
_AerCompat = None
_AerSimulatorCls = None
_QuantumCircuit = None
_Parameter = None
try:
    import numpy as _np  # used when qiskit works or fallback math
except Exception:
    # numpy is helpful but fallback to python math if not present
    _np = None

try:
    import qiskit as _qiskit_mod
    _qiskit = _qiskit_mod
    # try canonical Aer path
    try:
        from qiskit.providers.aer import Aer as _Aer  # newer qiskit exposes Aer provider
        _AerCompat = _Aer
        # prefer AerSimulator from the Aer provider if present
        try:
            from qiskit.providers.aer import AerSimulator as _AerSimCls
            _AerSimulatorCls = _AerSimCls
        except Exception:
            _AerSimulatorCls = None
    except Exception:
        # fallback to qiskit_aer package
        try:
            import qiskit_aer as _qiskit_aer
            # qiskit_aer provides AerSimulator class
            try:
                from qiskit_aer import AerSimulator as _AerSimCls
                _AerSimulatorCls = _AerSimCls
                _AerCompat = _qiskit_aer
            except Exception:
                _AerSimulatorCls = None
        except Exception:
            _AerCompat = None
            _AerSimulatorCls = None

    # core terra pieces
    try:
        from qiskit import QuantumCircuit as _QC
        from qiskit.circuit import Parameter as _Param
        _QuantumCircuit = _QC
        _Parameter = _Param
    except Exception:
        _QuantumCircuit = None
        _Parameter = None

    # mark qiskit available if we imported at least the qiskit package
    _QISKIT_AVAILABLE = True
    # mark AER available if we found a simulator class
    _AER_AVAILABLE = _AerSimulatorCls is not None

    logger.info("qiskit imported (ok). AerSimulator available: %s", _AER_AVAILABLE)
except Exception:
    _qiskit = None
    _AerCompat = None
    _AerSimulatorCls = None
    _QuantumCircuit = None
    _Parameter = None
    _QISKIT_AVAILABLE = False
    _AER_AVAILABLE = False
    logger.info("qiskit not importable from this environment; will use classical fallback.")


# Helpers
def _encode_features(scan_result: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert a scan_result (nutrition + ayurveda) into a small numeric feature dict.
    Keeps the same feature names used in other code.
    """
    nut = scan_result.get("nutrition_estimate", {}) if isinstance(scan_result, dict) else {}
    calories = float(nut.get("calories", 0) or 0)
    protein = float(nut.get("protein", 0) or 0)
    carbs = float(nut.get("carbs", 0) or 0)
    fat = float(nut.get("fat", 0) or 0)

    ay = scan_result.get("ayurveda", {}) if isinstance(scan_result, dict) else {}
    dosha = ay.get("dosha_effect", {}) if isinstance(ay, dict) else {}

    def _dosha_val(key):
        v = dosha.get(key)
        if v is None:
            return 0.0
        if isinstance(v, str):
            s = v.lower()
            if "increase" in s:
                return 1.0
            if "decrease" in s:
                return -1.0
        return 0.0

    features = {
        "calories_norm": calories / 1000.0,
        "protein_norm": protein / 100.0,
        "carbs_norm": carbs / 200.0,
        "fat_norm": fat / 100.0,
        "kapha": _dosha_val("kapha"),
        "pitta": _dosha_val("pitta"),
        "vata": _dosha_val("vata"),
    }
    return features


# -------------------------
# Qiskit-backed optimizer
# -------------------------
def _build_vqc(num_qubits: int = 3, num_layers: int = 2):
    """
    Build a small parameterized variational circuit.
    Returns (qc, params_list)
    """
    if _QuantumCircuit is None or _Parameter is None:
        raise RuntimeError("Quantum circuit primitives not available in this environment")

    params = []
    qc = _QuantumCircuit(num_qubits)
    # create parameterized layers
    for layer in range(num_layers):
        for q in range(num_qubits):
            p = _Parameter(f"theta_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        # simple entangling
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    # measurement not appended here; we will use statevector/counts to evaluate
    return qc, params


def _eval_on_backend_state_expectation(sim, circuit, params, theta_values):
    """
    Bind params, run circuit on sim (AerSimulator) and compute expectation of Z on first qubit.
    Returns a float in [-1,1] (higher -> better for our toy optimization objective).
    """
    # bind parameters (adapt to qiskit version)
    try:
        # newer API
        bound = circuit.bind_parameters({p: v for p, v in zip(params, theta_values)})
    except AttributeError:
        # older API (assign_parameters)
        try:
            bound = circuit.assign_parameters({p: v for p, v in zip(params, theta_values)})
        except Exception:
            # if neither works, raise and caller will fallback
            raise RuntimeError("Parameter binding not supported by this qiskit installation")

    # run on simulator
    try:
        # AerSimulator.run() preferred in newer qiskit-aer
        job = sim.run(bound)
        res = job.result()
    except Exception:
        # older style execute/transpile
        try:
            from qiskit import transpile, execute  # type: ignore
            backend = sim
            tqc = transpile(bound, backend=backend)
            job = execute(tqc, backend=backend, shots=1024)
            res = job.result()
        except Exception as e:
            raise RuntimeError(f"Failed to run circuit on Aer: {e}")

    # try get statevector or counts and compute expectation of Z on qubit 0
    try:
        # first try statevector extraction
        sv = None
        try:
            sv = res.get_statevector(bound)
        except Exception:
            # some Aer versions expose get_statevector differently
            try:
                sv = res.data(bound).get("statevector")  # sometimes available
            except Exception:
                sv = None

        if sv is not None:
            # sv might be qiskit.quantum_info.Statevector or numpy array
            if hasattr(sv, "data"):
                arr = _np.asarray(sv.data) if _np is not None else None
            else:
                arr = _np.asarray(sv) if _np is not None else None

            if arr is None:
                # fallback: can't compute expectation
                raise RuntimeError("Can't convert statevector to numpy array")
            # compute probability of first qubit = 0 and 1
            n = arr.size
            probs = (arr * _np.conj(arr)).real
            # bit indexing: least significant bit is qubit 0 in Qiskit ordering (rightmost)
            p0 = 0.0
            p1 = 0.0
            num_qubits = int(round(_np.log2(n)))
            for idx, pr in enumerate(probs):
                # check bit 0 of idx
                if (idx & 1) == 0:
                    p0 += pr
                else:
                    p1 += pr
            expectation = p0 - p1  # in [-1,1]
            return float(expectation)
        else:
            # fallback to counts
            counts = res.get_counts(bound)
            total = sum(counts.values()) if counts else 1
            p0 = 0.0
            p1 = 0.0
            for bitstring, c in counts.items():
                # qiskit bitstring is e.g. '010' with qubit0 as rightmost character
                if bitstring and bitstring[-1] == "0":
                    p0 += c
                else:
                    p1 += c
            p0 /= total
            p1 /= total
            expectation = p0 - p1
            return float(expectation)
    except Exception as e:
        raise RuntimeError(f"Failed to extract expectation from simulator result: {e}")


def _qiskit_optimize(features: Dict[str, float], maxiter: int = 32) -> Dict[str, Any]:
    """
    Try to run a small VQC-based optimization using AerSimulator.
    Returns a dictionary with engine='qiskit_simulator' and other details.
    May raise RuntimeError if qiskit primitives missing or runtime fails.
    """
    if not _QISKIT_AVAILABLE or not _AER_AVAILABLE:
        raise RuntimeError("Qiskit/Aer not available")

    # build circuit
    num_qubits = 3
    num_layers = 2
    qc, params = _build_vqc(num_qubits=num_qubits, num_layers=num_layers)

    # --- NEW FIX ---
    # AerSimulator requires explicit save instruction or measurement.
    # We add a statevector save so statevector extraction always works.
    try:
        qc.save_statevector()
    except Exception:
        # For very old Aer that doesn't support save_statevector:
        # Add measurement fallback
        try:
            qc.measure_all()
        except Exception:
            raise RuntimeError("Unable to attach statevector or measurement to circuit")
    # --- END FIX ---


    # build simulator instance (adapt to AerSimulator class)
    try:
        sim = _AerSimulatorCls()
    except Exception as e:
        # try to instantiate via Aer provider wrapper if present
        try:
            if _AerCompat is not None and hasattr(_AerCompat, "get_backend"):
                backend_cls = _AerCompat.get_backend("aer_simulator")
                sim = backend_cls()
            else:
                raise
        except Exception as e2:
            raise RuntimeError(f"AerSimulator instantiation failed: {e2 or e}")

    # tiny objective: we convert features to a target expectation we like and attempt to maximize
    # map features to a target value in [-1, 1] to be approximated by expectation
    # simple mapping: desired = 1.0 - normalized calories - kapha penalty
    calories_norm = float(features.get("calories_norm", 0.0))
    kapha = float(features.get("kapha", 0.0))
    desired = 1.0 - (calories_norm * 0.5) - (0.2 * kapha)
    desired = max(-1.0, min(1.0, desired))

    def objective(theta: Tuple[float, ...]) -> float:
        # returns scalar to minimize (we'll minimize squared error from desired)
        try:
            exp = _eval_on_backend_state_expectation(sim, qc, params, theta)
        except Exception as e:
            # bubble up a runtime error to trigger fallback
            raise RuntimeError(f"Quantum run failed during objective evaluation: {e}")
        # we want exp close to `desired`
        return float((exp - desired) ** 2)

    # Try to use scipy.optimize.minimize (COBYLA) if available for small dimension
    dim = len(params)
    best_theta = None
    best_val = float("inf")
    used_scipy = False
    try:
        import scipy.optimize as _sopt  # type: ignore
        used_scipy = True
        # random init
        init = [random.uniform(-math.pi, math.pi) for _ in range(dim)]
        res = _sopt.minimize(lambda x: objective(tuple(x)), x0=init, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5})
        theta_best = list(res.x)
        best_val = float(res.fun)
        best_theta = theta_best
    except Exception as e:
        # fallback to simple random-search over a few iterations
        logger.info("Qiskit optimize: scipy not usable or optimization failed (%s) — random search", e)
        for i in range(max(8, maxiter)):
            cand = [random.uniform(-math.pi, math.pi) for _ in range(dim)]
            try:
                v = objective(tuple(cand))
            except Exception as e2:
                raise RuntimeError(f"Quantum evaluation failed (random-search): {e2}")
            if v < best_val:
                best_val = v
                best_theta = cand

    # map best_val -> recommended portion percent (toy mapping)
    # smaller best_val means expectation matched desired => safer (higher allowed portion)
    recommended_portion_pct = max(0.1, min(1.0, 1.0 - best_val))
    risk_score = float(min(1.0, max(0.0, best_val)))  # approximate
    return {
        "engine": "qiskit_simulator",
        "risk_score": risk_score,
        "recommended_portion_pct": recommended_portion_pct,
        "safe_frequency": "regular" if recommended_portion_pct > 0.7 else "1-2 times/week",
        "notes": "Qiskit AerSimulator optimized (COBYLA if available, else random-search)",
        "raw": {"best_theta": best_theta, "iterations": maxiter, "used_scipy": used_scipy},
    }


# -------------------------
# Classical fallback optimizer
# -------------------------
def _classical_fallback(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Fast deterministic classical heuristic for a recommendation.
    """
    cal = float(features.get("calories_norm", 0.0))
    kapha = float(features.get("kapha", 0.0))
    # heuristic: higher calories or kapha => reduce portion
    base = 1.0 - (cal * 0.6) - (0.2 * (kapha if kapha > 0 else 0.0))
    recommended_portion_pct = max(0.1, min(1.0, round(base, 4)))
    risk_score = max(0.0, min(1.0, 1.0 - recommended_portion_pct))
    safe_frequency = "regular" if recommended_portion_pct >= 0.8 else "1-2 times/week"
    return {
        "engine": "classical_fallback",
        "risk_score": float(risk_score),
        "recommended_portion_pct": float(recommended_portion_pct),
        "safe_frequency": safe_frequency,
        "notes": "Deterministic classical fallback optimizer.",
        "raw": {"engine": "classical_fallback", "notes": "fast heuristic", "timestamp": int(time.time())},
    }


# -------------------------
# Public API
# -------------------------
def _normalize_input_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either:
     - full scan_result dict (has nutrition_estimate),
     - or a feature dict already (contains features like calories_norm)
    Return (features_dict, original_payload)
    """
    if not isinstance(payload, dict):
        raise ValueError("input must be a dict")
    # detection of full scan result
    if "nutrition_estimate" in payload or "ayurveda" in payload:
        features = _encode_features(payload)
        return features, payload
    # else assume it's already feature dict
    # copy safety
    return dict(payload), payload


def run_quantum_optimize_sync(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous optimize call:
      - input_payload may be full scan_result or a precomputed features dict
      - tries qiskit optimize; if any error occurs, falls back to classical optimizer
    Returns a dict with engine, risk_score, recommended_portion_pct, safe_frequency, notes, raw
    """
    features, orig = _normalize_input_payload(input_payload)
    # try qiskit route
    if _QISKIT_AVAILABLE and _AER_AVAILABLE:
        try:
            return _qiskit_optimize(features)
        except Exception as e:
            logger.warning("Qiskit run failed, using classical fallback: %s", e)
            # fall through to classical
    # classical fallback
    return _classical_fallback(features)


# Async job API (lightweight in-memory worker)
def _worker_process_job(job_id: str):
    j = _jobs.get(job_id)
    if not j:
        return
    try:
        j["status"] = "running"
        payload = j.get("input_payload", {}) or {}
        features, _ = _normalize_input_payload(payload)
        # attempt qiskit sync optimize (may raise)
        if _QISKIT_AVAILABLE and _AER_AVAILABLE:
            try:
                res = _qiskit_optimize(features)
            except Exception as e:
                logger.warning("Async qiskit job failed (%s) — using classical fallback", e)
                res = _classical_fallback(features)
        else:
            res = _classical_fallback(features)
        j["status"] = "done"
        j["result"] = res
        j["completed_at"] = int(time.time())
        j["engine"] = res.get("engine", "classical_fallback")
    except Exception as e:
        logger.exception("Job worker failed")
        j["status"] = "error"
        j["error"] = str(e)


def submit_quantum_job(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit an asynchronous quantum job.
    Returns immediately with job_id and an immediate classical fallback.
    The job runs in a background thread (in-memory).
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "created_at": int(time.time()),
        "status": "queued",
        "input_payload": input_payload,
        "result": None,
    }
    # start worker thread
    t = threading.Thread(target=_worker_process_job, args=(job_id,), daemon=True)
    t.start()
    # provide immediate classical fallback result
    features, _ = _normalize_input_payload(input_payload)
    classical = _classical_fallback(features)
    return {"job_id": job_id, "status": "queued", "classical_fallback": classical}


def get_job_status(job_id: str) -> Dict[str, Any]:
    j = _jobs.get(job_id)
    if not j:
        return {"job_id": job_id, "status": "not_found"}
    return j.copy()


# export some names for other modules that inspect flags
__all__ = [
    "run_quantum_optimize_sync",
    "submit_quantum_job",
    "get_job_status",
    "_encode_features",
    "_QISKIT_AVAILABLE",
    "_AER_AVAILABLE",
]
