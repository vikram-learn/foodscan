# backend/app/services/quantum_service.py
"""
Quantum service (Qiskit + Aer when available) with classical fallback.

Provides:
 - submit_quantum_job(input_payload) -> { job_id, status, classical_fallback }
 - get_job_status(job_id) -> job info
 - run_quantum_optimize_sync(scan_result) -> optimization dict

Compatibility:
 - Supports qiskit.providers.aer (preferred) or qiskit_aer (fallback package).
 - If Qiskit/Aer primitives needed for evaluation are missing, falls back gracefully.
"""

from typing import Dict, Any, Optional
import time
import uuid
import math
import random
import threading
import logging

logger = logging.getLogger("quantum_service")
logger.setLevel(logging.INFO)

# in-memory job store for prototyping
_jobs: Dict[str, Dict[str, Any]] = {}

# Feature encoder -----------------------------------------------------------
def _encode_features(scan_result: Dict[str, Any]) -> Dict[str, float]:
    nut = scan_result.get("nutrition_estimate", {}) or {}
    calories = float(nut.get("calories", 0))
    protein = float(nut.get("protein", 0))
    carbs = float(nut.get("carbs", 0))
    fat = float(nut.get("fat", 0))

    ay = scan_result.get("ayurveda", {}) or {}
    dosha = ay.get("dosha_effect", {}) or {}

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

# Try to import Qiskit and Aer with robust compatibility --------------------
_QISKIT_AVAILABLE = False
_AER_AVAILABLE = False
_AER_SIM_CLASS = None
try:
    import qiskit
    _QISKIT_AVAILABLE = True
    logger.info("qiskit import ok, version=%s", getattr(qiskit, "__version__", None))
except Exception as e:
    logger.info("qiskit import failed: %s", e)
    _QISKIT_AVAILABLE = False

# Prefer qiskit.providers.aer, fallback to qiskit_aer
try:
    # preferred import path used by most qiskit versions
    from qiskit.providers.aer import Aer  # type: ignore
    try:
        # If Aer exposes AerSimulator via providers.aer
        from qiskit.providers.aer import AerSimulator  # type: ignore
        _AER_SIM_CLASS = AerSimulator
    except Exception:
        _AER_SIM_CLASS = None
    _AER_AVAILABLE = True
    logger.info("qiskit.providers.aer available")
except Exception:
    # fallback: qiskit_aer package (separate distribution) often exposes AerSimulator
    try:
        from qiskit_aer import AerSimulator as _AerSim  # type: ignore
        _AER_SIM_CLASS = _AerSim
        # provide a small Aer-compatible wrapper that exposes get_backend(name)
        class _AerCompat:
            @staticmethod
            def get_backend(name: str):
                # return the AerSimulator class; callers expecting an instance should instantiate it
                return _AER_SIM_CLASS
        Aer = _AerCompat()
        _AER_AVAILABLE = True
        logger.info("qiskit_aer fallback available (AerSimulator loaded)")
    except Exception as e:
        _AER_AVAILABLE = False
        logger.info("Aer not available (qiskit.providers.aer & qiskit_aer both failed): %s", e)

# Only import the Terra primitives when qiskit is present
if _QISKIT_AVAILABLE:
    try:
        from qiskit import QuantumCircuit, transpile  # type: ignore
        from qiskit.circuit import Parameter  # type: ignore
        # quantum_info for statevector manipulation when available
        try:
            from qiskit.quantum_info import Statevector  # type: ignore
        except Exception:
            Statevector = None
    except Exception:
        QuantumCircuit = None
        Parameter = None
        transpile = None
        Statevector = None

# Small utilities -----------------------------------------------------------
def _classical_fallback_optimizer(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Fast, deterministic heuristic for a recommendation.
    Returns a dictionary similar to quantum optimizer outputs.
    """
    # simple heuristic: more calories -> higher risk_score; kapha increase raises risk slightly
    calories = features.get("calories_norm", 0.0) * 1000.0
    kapha = features.get("kapha", 0.0)
    base = min(1.0, max(0.0, (calories - 200.0) / 800.0))  # normalized risk
    risk = round(min(1.0, base + 0.1 * kapha), 3)
    recommended_portion_pct = round(max(0.1, 1.0 - risk * 0.6), 4)
    safe_frequency = "regular" if risk < 0.3 else "1-2 times/week" if risk < 0.7 else "limit"
    result = {
        "engine": "classical_fallback",
        "risk_score": risk,
        "recommended_portion_pct": recommended_portion_pct,
        "safe_frequency": safe_frequency,
        "notes": "Deterministic classical fallback optimizer.",
        "timestamp": int(time.time()),
    }
    return result

# Qiskit-based small VQC optimizer -----------------------------------------
def _qiskit_optimize(features: Dict[str, float], maxiter: int = 32) -> Dict[str, Any]:
    """
    Try to run a tiny variational circuit and produce a result. If any Qiskit/Aer
    primitives needed are missing, raise RuntimeError.
    """
    if not (_QISKIT_AVAILABLE and _AER_AVAILABLE and QuantumCircuit is not None):
        raise RuntimeError("Quantum circuit primitives not available in this environment")

    # Map features to angles (simple deterministic mapping)
    theta_seed = [
        features.get("calories_norm", 0.0) * 2.0,
        features.get("protein_norm", 0.0) * 3.0,
        features.get("carbs_norm", 0.0) * 1.5,
        features.get("fat_norm", 0.0) * 2.5,
        features.get("kapha", 0.0) * 1.0,
        features.get("pitta", 0.0) * 1.0,
    ]
    num_qubits = 3
    num_layers = 2

    # build parametric circuit
    params = []
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        for q in range(num_qubits):
            p = Parameter(f"t_{layer}_{q}")
            params.append(p)
            qc.ry(p, q)
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)

    # We'll evaluate a simple objective: expectation of Z on qubit-0 (mapped to [0..1] risk)
    # Create a small evaluation function that runs the circuit with a given theta
    def eval_params(theta_values):
        # bind parameters (terra >= 0.20 uses bind_parameters or assign_parameters)
        try:
            # prefer bind_parameters if available
            base_qc = qc
            try:
                bound = base_qc.bind_parameters({p: v for p, v in zip(params, theta_values)})
            except AttributeError:
                # older/newer APIs might use assign_parameters
                bound = base_qc.assign_parameters({p: v for p, v in zip(params, theta_values)})
        except Exception:
            # fallback: do a quick numeric objective without running quantum
            return float(sum((v % math.pi) ** 2 for v in theta_values))

        # transpile for simulator backend
        try:
            backend_cls = Aer.get_backend("aer_simulator") if hasattr(Aer, "get_backend") else Aer
            # instantiate simulator if class returned
            sim = backend_cls() if callable(backend_cls) else backend_cls
        except Exception:
            # fallback: if AER_SIM_CLASS present, use it
            if _AER_SIM_CLASS is not None:
                try:
                    sim = _AER_SIM_CLASS()
                except Exception:
                    raise RuntimeError("Qiskit/Aer primitives not available")
            else:
                raise RuntimeError("Qiskit/Aer primitives not available")

        # make sure we request statevector if supported
        # Use save_statevector instruction (Aer will provide statevector) or run with many shots for counts
        qc_to_run = bound.copy()
        # For newer Aer, save_statevector is the API
        try:
            qc_to_run.save_statevector()
            job = sim.run(transpile(qc_to_run, sim))
            res = job.result()
            # try to get statevector
            try:
                sv = res.get_statevector(qc_to_run)
                # sv may be qiskit.quantum_info.Statevector or numpy array-like
                probs = (abs(s) ** 2 for s in sv)
                # compute expectation of Z on qubit-0: +1 for |0>, -1 for |1>
                # map basis states -> bit0 is LSB in many tools; we'll compute by index
                # compute p0 = sum probs of states where qubit0 = 0
                p0 = 0.0
                # convert to list to iterate
                p_list = list(abs(s) ** 2 for s in sv)
                n = len(p_list)
                for idx, p in enumerate(p_list):
                    # extract LSB (qubit-0 index) -> if using standard ordering, LSB corresponds to qubit 0
                    bit0 = (idx >> 0) & 1
                    if bit0 == 0:
                        p0 += p
                expectation_z = (p0 * 1.0 + (1 - p0) * -1.0)
                # map expectation [-1,1] -> risk [0,1]
                risk = round(max(0.0, min(1.0, (1.0 - expectation_z) / 2.0)), 6)
                return float(risk)
            except Exception:
                # try to get counts
                try:
                    counts = res.get_counts()
                    # compute p0 from counts
                    total = sum(counts.values()) if counts else 1
                    p0 = 0.0
                    for bitstr, c in counts.items():
                        # bitstr likely "010" with leftmost = qubitN-1; check both possibilities
                        # prefer last char as qubit-0
                        if bitstr[-1] == "0":
                            p0 += c
                    p0 = p0 / total
                    expectation_z = (p0 * 1.0 + (1 - p0) * -1.0)
                    risk = round(max(0.0, min(1.0, (1.0 - expectation_z) / 2.0)), 6)
                    return float(risk)
                except Exception:
                    raise RuntimeError("Failed to extract expectation from simulator result")
        except Exception:
            # If save_statevector not supported, try simple counts-based run
            try:
                transpiled = transpile(bound, sim)
                job = sim.run(transpiled, shots=512)
                res = job.result()
                counts = res.get_counts()
                total = sum(counts.values()) if counts else 1
                p0 = 0.0
                for bitstr, c in counts.items():
                    if bitstr[-1] == "0":
                        p0 += c
                p0 = p0 / total
                expectation_z = (p0 * 1.0 + (1 - p0) * -1.0)
                risk = round(max(0.0, min(1.0, (1.0 - expectation_z) / 2.0)), 6)
                return float(risk)
            except Exception:
                raise RuntimeError("Quantum run failed during objective evaluation")

    # simple optimizer: use COBYLA if available, otherwise random-search
    try:
        # try classical optimizer from qiskit or scipy if present
        from qiskit.algorithms.optimizers import COBYLA  # type: ignore
        use_cobyla = True
    except Exception:
        use_cobyla = False

    # initial theta from seed (wrap to small angles)
    init_theta = [(s % (2 * math.pi)) for s in theta_seed for _ in range(1)]
    # extend or truncate to params length
    param_len = len(params)
    # repeat pattern to match parameter count if needed
    theta0 = [init_theta[i % len(init_theta)] for i in range(param_len)]

    if use_cobyla:
        try:
            opt = COBYLA(maxiter=maxiter)
            # construct objective that minimises risk
            def obj(x):
                try:
                    return eval_params(x)
                except Exception:
                    return float(sum((v % math.pi) ** 2 for v in x))
            # run a simple optimization (we won't return the optimizer object)
            result = opt.optimize(num_vars=param_len, objective=obj, initial_point=theta0)
            best_theta = list(result[0])
            best_score = float(result[1])
            # assemble response mapping score->risk
            return {
                "engine": "qiskit_simulator",
                "risk_score": best_score,
                "recommended_portion_pct": round(max(0.05, 1.0 - best_score), 6),
                "safe_frequency": "regular" if best_score < 0.3 else "1-2 times/week",
                "notes": "Qiskit AerSimulator optimized (COBYLA)",
                "raw": {"best_theta": best_theta, "iterations": maxiter},
            }
        except Exception:
            # fallthrough to random search
            logger.info("COBYLA attempt failed; falling back to random-search")
    # random-search fallback: try N random samples and pick minimal risk
    best_theta = None
    best = 1.0
    for _ in range(maxiter):
        cand = [random.uniform(-math.pi, math.pi) for _ in range(param_len)]
        try:
            score = eval_params(cand)
            if score < best:
                best = score
                best_theta = cand
        except Exception as e:
            logger.debug("random-search eval failed: %s", e)
    if best_theta is None:
        # everything failed
        raise RuntimeError("Qiskit/Aer primitives not available")
    return {
        "engine": "qiskit_simulator",
        "risk_score": round(float(best), 6),
        "recommended_portion_pct": round(max(0.05, 1.0 - float(best)), 6),
        "safe_frequency": "regular" if best < 0.3 else "1-2 times/week",
        "notes": "Qiskit AerSimulator counts-based random-search evaluation",
        "raw": {"best_theta": best_theta, "iterations": maxiter},
    }

# Public API ---------------------------------------------------------------
def run_quantum_optimize_sync(scan_result: Dict[str, Any]) -> Dict[str, Any]:
    features = _encode_features(scan_result)
    # try qiskit path
    try:
        if _QISKIT_AVAILABLE and _AER_AVAILABLE:
            res = _qiskit_optimize(features, maxiter=32)
            return res
    except Exception as e:
        logger.info("Qiskit run failed, using classical fallback: %s", e)
    # classical fallback
    return _classical_fallback_optimizer(features)

def submit_quantum_job(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit an asynchronous job: store in-memory and spawn a worker that tries quantum run,
    but returns a classical fallback immediately to keep API responsive.
    """
    job_id = str(uuid.uuid4())
    created = int(time.time())
    # compute immediate classical fallback
    try:
        classical = run_quantum_optimize_sync(input_payload)
    except Exception as e:
        classical = {"engine": "classical_fallback", "notes": f"fallback_error: {e}", "timestamp": int(time.time())}

    _jobs[job_id] = {
        "job_id": job_id,
        "created_at": created,
        "status": "queued",
        "input_payload": input_payload,
        "classical_fallback": classical,
        "result": None,
        "completed_at": None,
        "engine": None,
    }

    def worker(jid):
        logger.info("Worker started for job %s", jid)
        try:
            r = run_quantum_optimize_sync(input_payload)
            _jobs[jid]["result"] = r
            _jobs[jid]["engine"] = r.get("engine")
            _jobs[jid]["status"] = "done"
            _jobs[jid]["completed_at"] = int(time.time())
            logger.info("Worker finished job %s", jid)
        except Exception as e:
            logger.info("Async qiskit job failed (%s) — using classical fallback", e)
            _jobs[jid]["result"] = _jobs[jid]["classical_fallback"]
            _jobs[jid]["engine"] = _jobs[jid]["result"].get("engine")
            _jobs[jid]["status"] = "done"
            _jobs[jid]["completed_at"] = int(time.time())

    t = threading.Thread(target=worker, args=(job_id,), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "queued", "classical_fallback": classical}

def get_job_status(job_id: str) -> Dict[str, Any]:
    j = _jobs.get(job_id)
    if not j:
        return {"job_id": job_id, "status": "not_found"}
    return j

# small helper to expose module flags (for debug)
def _module_flags() -> Dict[str, Any]:
    return {"_QISKIT_AVAILABLE": _QISKIT_AVAILABLE, "_AER_AVAILABLE": _AER_AVAILABLE}

# Expose names
__all__ = [
    "run_quantum_optimize_sync",
    "submit_quantum_job",
    "get_job_status",
    "_encode_features",
    "_module_flags",
]
