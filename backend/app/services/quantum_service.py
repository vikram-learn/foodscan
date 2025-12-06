# backend/app/services/quantum_service.py
"""
Quantum service using Qiskit (with classical fallback).

This implementation:
 - Tries to detect Qiskit + Aer in multiple import layouts
   (qiskit.providers.aer vs qiskit_aer).
 - Uses AerSimulator.run(...) when available.
 - Uses assign_parameters(...) to bind parameter values (compatible with many qiskit versions).
 - Falls back to a fast deterministic classical optimizer if quantum primitives fail.
 - Provides a small in-memory job store for async demos.
"""
from typing import Dict, Any, Optional
import time
import uuid
import threading
import math
import random
import logging

logger = logging.getLogger("quantum_service")
logger.setLevel(logging.INFO)

# In-memory job store for prototyping
_jobs: Dict[str, Dict[str, Any]] = {}

# Module capability flags (updated at import time)
_QISKIT_AVAILABLE = False
_AER_AVAILABLE = False

# Placeholders for imported classes (if available)
_AerSimulatorCls = None
_QuantumCircuit = None
_Parameter = None
_transpile = None

# --- robust imports for qiskit / aer ---
try:
    # try import qiskit top-level
    import qiskit as _qiskit
    _QISKIT_AVAILABLE = True
    logger.info("qiskit import OK (%s)", getattr(_qiskit, "__version__", None))
except Exception:
    _qiskit = None
    logger.info("qiskit import failed")

# Try AerSimulator from qiskit.providers.aer (preferred)
if _QISKIT_AVAILABLE:
    try:
        from qiskit.providers.aer import AerSimulator as _AerSim1  # type: ignore
        _AerSimulatorCls = _AerSim1
        _AER_AVAILABLE = True
        logger.info("AerSimulator imported from qiskit.providers.aer")
    except Exception:
        # try separate qiskit_aer package
        try:
            import qiskit_aer as _qiskit_aer  # type: ignore
            # AerSimulator lives under qiskit_aer.backends.aer_simulator.AerSimulator
            try:
                from qiskit_aer.backends.aer_simulator import AerSimulator as _AerSim2  # type: ignore
                _AerSimulatorCls = _AerSim2
                _AER_AVAILABLE = True
                logger.info("AerSimulator imported from qiskit_aer.backends.aer_simulator")
            except Exception:
                _AER_AVAILABLE = False
                logger.info("qiskit_aer import present but AerSimulator not found")
        except Exception:
            _AER_AVAILABLE = False
            logger.info("qiskit.providers.aer import failed and qiskit_aer not present")

# Try to import circuit related primitives if qiskit available
if _QISKIT_AVAILABLE:
    try:
        # prefer qiskit.circuit objects
        from qiskit import QuantumCircuit as _QC, transpile as _tp  # type: ignore
        from qiskit.circuit import Parameter as _Param  # type: ignore
        _QuantumCircuit = _QC
        _Parameter = _Param
        _transpile = _tp
    except Exception:
        # fallback: try qiskit.circuit from qiskit package (older/newer variations)
        try:
            from qiskit.circuit import QuantumCircuit as _QC, Parameter as _Param  # type: ignore
            _QuantumCircuit = _QC
            _Parameter = _Param
            _transpile = None
        except Exception:
            _QuantumCircuit = None
            _Parameter = None

# Module state helper
def _module_flags() -> Dict[str, bool]:
    return {"_QISKIT_AVAILABLE": bool(_QISKIT_AVAILABLE), "_AER_AVAILABLE": bool(_AER_AVAILABLE)}

# -------------------------
# Feature encoder
# -------------------------
def _encode_features(scan_result: Dict[str, Any]) -> Dict[str, float]:
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
        if isinstance(v, str):
            s = v.lower()
            if "increase" in s:
                return 1.0
            if "decrease" in s:
                return -1.0
        return 0.0

    return {
        "calories_norm": calories / 1000.0,
        "protein_norm": protein / 100.0,
        "carbs_norm": carbs / 200.0,
        "fat_norm": fat / 100.0,
        "kapha": _dosha_val("kapha"),
        "pitta": _dosha_val("pitta"),
        "vata": _dosha_val("vata"),
    }

# -------------------------
# Small parameterized VQC builder
# -------------------------
def _build_simple_vqc(num_qubits: int, num_layers: int):
    """
    Returns (qc_template, params_list).
    Each layer: ry(param) on each qubit, then CNOT chain.
    The circuit includes final measurement in Z basis.
    """
    if _QuantumCircuit is None or _Parameter is None:
        raise RuntimeError("QuantumCircuit/Parameter not available")

    params = []
    qc = _QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        for q in range(num_qubits):
            p = _Parameter(f"theta_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        # entangle chain
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    # add measurements
    qc.measure_all()
    return qc, params

# -------------------------
# Qiskit-backed optimize (best-effort)
# -------------------------
def _qiskit_optimize(features: Dict[str, float], maxiter: int = 32) -> Dict[str, Any]:
    """
    Try to perform a tiny VQC optimization using AerSimulator.
    Returns dict with recommendation or raises if qiskit/Aer unavailable.
    """
    if not (_QISKIT_AVAILABLE and _AER_AVAILABLE and _QuantumCircuit):
        raise RuntimeError("Qiskit/Aer primitives not available")

    # simple feature -> target mapping: combine floats into scalar target (toy)
    # In real training you'd use labelled training data; here we create a toy objective
    target = float(features.get("calories_norm", 0.0))  # toy target

    # small circuit
    num_qubits = 2
    num_layers = 3
    qc_template, params = _build_simple_vqc(num_qubits, num_layers)

    # create simulator
    try:
        sim = _AerSimulatorCls()
    except Exception as e:
        # try constructing via qiskit Aer wrapper if present
        raise RuntimeError("Failed to create AerSimulator: %s" % e)

    # evaluation function: given theta vector, assign parameters, run and compute expectation
    def eval_theta(theta_values):
        # create mapping param->value
        mapping = {p: float(v) for p, v in zip(params, theta_values)}
        try:
            # assign_parameters is broadly supported and returns a new circuit
            qc_bound = qc_template.assign_parameters(mapping)
        except Exception:
            # try bind_parameters as older API
            try:
                qc_bound = qc_template.bind_parameters(mapping)
            except Exception as e:
                raise RuntimeError("Failed to bind parameters: %s" % e)

        # run the circuit; prefer counts via shot sampling
        try:
            # new AerSimulator uses .run(...) -> job
            job = sim.run(qc_bound, shots=1024)
            result = job.result()
            # try counts extraction
            try:
                # tries to get counts for the measurement
                counts = result.get_counts(0) if hasattr(result, "get_counts") else result.get_counts(qc_bound)
            except Exception:
                # try generic get_counts
                counts = result.get_counts()
            # compute simple expectation of first qubit (0 label counts vs others)
            # counts keys could be bitstrings like '00','01' -> first char is q(n-1) depending on endianness
            total = sum(counts.values()) if isinstance(counts, dict) else 0
            if total == 0:
                raise RuntimeError("No counts returned by simulator")
            # sum where first qubit measured '0'
            zeros = 0
            ones = 0
            for bits, c in counts.items():
                # ensure string
                bstr = str(bits)
                # qiskit returns bitstrings with qubit-0 as rightmost char -> take rightmost char
                first_bit = bstr[-1] if len(bstr) > 0 else "0"
                if first_bit == "0":
                    zeros += c
                else:
                    ones += c
            expectation = (zeros - ones) / float(total)
            # map expectation -> risk_score (toy mapping)
            risk_score = max(0.0, min(1.0, 1.0 - (expectation + 1.0) / 2.0))
            recommended_portion_pct = max(0.0, min(1.0, 1.0 - risk_score * 0.05))
            return {"engine": "qiskit_simulator", "risk_score": round(risk_score, 6), "recommended_portion_pct": round(recommended_portion_pct, 6), "raw": {"counts": counts}}
        except Exception as e:
            raise RuntimeError("Quantum run failed during objective evaluation: %s" % e)

    # Simple random-search + optional local optimize (toy)
    best = None
    best_theta = None
    for it in range(maxiter):
        theta0 = [random.uniform(-math.pi, math.pi) for _ in params]
        try:
            out = eval_theta(theta0)
            score = out["risk_score"]
            if best is None or score < best:
                best = score
                best_theta = theta0
        except Exception:
            # skip failing evaluations
            continue

    if best is None:
        raise RuntimeError("Quantum evaluation failed (random-search): no successful evaluations")

    return {"engine": "qiskit_simulator", "risk_score": best, "recommended_portion_pct": max(0.0, min(1.0, 1.0 - best * 0.05)), "raw": {"best_theta": best_theta, "iterations": maxiter}}

# -------------------------
# Classical fallback optimizer (deterministic, quick)
# -------------------------
def _classical_fallback(features: Dict[str, float]) -> Dict[str, Any]:
    # toy deterministic heuristic combining calories + kapha
    c = features.get("calories_norm", 0.0)
    kapha = features.get("kapha", 0.0)
    score = min(1.0, max(0.0, (c * 0.8 + (kapha + 1.0) * 0.1)))
    recommended = max(0.0, min(1.0, 1.0 - 0.1 * score))
    freq = "regular" if score < 0.2 else ("1-2 times/week" if score < 0.6 else "rarely")
    return {"engine": "classical_fallback", "risk_score": round(score, 6), "recommended_portion_pct": round(recommended, 6), "safe_frequency": freq, "notes": "Deterministic classical fallback optimizer.", "raw": {"engine": "classical_fallback", "notes": "fast heuristic", "timestamp": int(time.time())}}

# -------------------------
# Public API
# -------------------------
def run_quantum_optimize_sync(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous call: encodes features and tries qiskit optimize, else classical fallback.
    """
    features = _encode_features(scan_result)
    # prefer qiskit if primitives available
    if _QISKIT_AVAILABLE and _AER_AVAILABLE and _QuantumCircuit:
        try:
            return _qiskit_optimize(features)
        except Exception as e:
            logger.info("Qiskit run failed, using classical fallback: %s", e)
            return _classical_fallback(features)
    else:
        return _classical_fallback(features)

def submit_quantum_job(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit a job for asynchronous processing. Job worker will attempt qiskit and fall back.
    """
    job_id = str(uuid.uuid4())
    created = int(time.time())
    # store job with queued status
    _jobs[job_id] = {"job_id": job_id, "status": "queued", "created_at": created, "input_payload": scan_result, "result": None}
    # run worker thread
    def _worker(jid, payload):
        try:
            _jobs[jid]["status"] = "running"
            res = run_quantum_optimize_sync(payload)
            _jobs[jid]["result"] = res
            _jobs[jid]["status"] = "done"
            _jobs[jid]["completed_at"] = int(time.time())
            _jobs[jid]["engine"] = res.get("engine")
        except Exception as e:
            _jobs[jid]["status"] = "failed"
            _jobs[jid]["result"] = {"engine": "classical_fallback", "error": str(e)}
    t = threading.Thread(target=_worker, args=(job_id, scan_result), daemon=True)
    t.start()
    # return immediate classical fallback to show something useful quickly
    classical = _classical_fallback(_encode_features(scan_result))
    return {"job_id": job_id, "status": "queued", "classical_fallback": classical}

def get_job_status(job_id: str) -> Dict[str, Any]:
    return _jobs.get(job_id, {"job_id": job_id, "status": "not_found"})

# expose introspection helper
def _module_flags() -> Dict[str, bool]:
    # return runtime flags for quick diagnostics
    return _module_flags.__wrapped__() if hasattr(_module_flags, "__wrapped__") else {"_QISKIT_AVAILABLE": bool(_QISKIT_AVAILABLE), "_AER_AVAILABLE": bool(_AER_AVAILABLE)}

# ensure the above function name returns real flags (fallback if weird)
def _module_flags_actual() -> Dict[str, bool]:
    return {"_QISKIT_AVAILABLE": bool(_QISKIT_AVAILABLE), "_AER_AVAILABLE": bool(_AER_AVAILABLE)}

# alias correct function
_module_flags = _module_flags_actual

# provide a tiny convenience export for external diagnostics
def module_info() -> Dict[str, Any]:
    return {"qiskit_present": _QISKIT_AVAILABLE, "aer_present": _AER_AVAILABLE, "aer_cls": getattr(_AerSimulatorCls, "__name__", None)}
