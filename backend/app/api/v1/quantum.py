# backend/app/api/v1/quantum.py
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional
from app.services import quantum_service
import traceback

router = APIRouter(prefix="/quantum", tags=["quantum"])

class OptimizeRequest(BaseModel):
    # Accept full scan_result or nutrition/ayurveda fragments
    scan_result: Optional[Dict[str, Any]] = None
    nutrition: Optional[Dict[str, Any]] = None
    ayurveda: Optional[Dict[str, Any]] = None
    # optionally provide a classical fallback to derive portions
    classical_result: Optional[Dict[str, Any]] = None

class OptimizeResponse(BaseModel):
    engine: str
    risk_score: float
    recommended_portion_pct: Optional[float] = None
    recommended_portion_grams: Optional[int] = None
    safe_frequency: Optional[str] = None
    explanation: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@router.post("/optimize", response_model=OptimizeResponse)
def optimize_sync(payload: OptimizeRequest = Body(...)):
    """
    Synchronous optimize: runs the quantum optimizer (or fallback) and returns a recommendation.
    Use this for immediate checks or small workloads.
    """
    try:
        # Build a scan_result dict to pass into the optimizer
        if payload.scan_result:
            scan = payload.scan_result
        else:
            scan = {
                "nutrition_estimate": payload.nutrition or {},
                "ayurveda": payload.ayurveda or {},
                # optionally include portion_grams from classical_result
                "portion_grams": (payload.classical_result or {}).get("portion_grams")
            }

        result = quantum_service.run_quantum_optimize_sync(scan)

        # map a few common fields into our response shape
        resp = {
            "engine": result.get("engine"),
            "risk_score": float(result.get("risk_score", 0.0)),
            "recommended_portion_pct": result.get("recommended_portion_pct"),
            "safe_frequency": result.get("safe_frequency"),
            "explanation": result.get("notes") or result.get("explanation"),
            "raw": result,
            # recommended_portion_grams will be null for pure sync call unless client passes classical portion
        }

        # if client passed classical portion, compute grams if possible
        c = payload.classical_result or {}
        portion_grams = c.get("portion_grams")
        if portion_grams and resp["recommended_portion_pct"]:
            try:
                resp["recommended_portion_grams"] = int(portion_grams * float(resp["recommended_portion_pct"]))
            except Exception:
                resp["recommended_portion_grams"] = None

        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Quantum optimize failed: {e}")


class SubmitRequest(BaseModel):
    # keeps same shape as earlier submit endpoint to maintain compatibility
    scan_result: Optional[Dict[str, Any]] = None
    classical_result: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None

class SubmitResponse(BaseModel):
    job_id: str
    status: str
    classical_fallback: Optional[Dict[str, Any]] = None

@router.post("/submit", response_model=SubmitResponse)
def submit_job(req: SubmitRequest):
    """
    Submit an async quantum job. Returns job id and immediate classical fallback if available.
    The job runs in background and can be polled via /status/{job_id}
    """
    try:
        payload = req.dict()
        # try to build a classical fallback if not provided
        classical = payload.get("classical_result")
        if not classical and payload.get("scan_result"):
            # simple fallback: copy portion and nutrition through
            s = payload["scan_result"]
            classical = {
                "portion_grams": s.get("portion_grams"),
                "nutrition_estimate": s.get("nutrition_estimate"),
                "notes": "Auto classical fallback (from scan_result)"
            }
            payload["classical_result"] = classical

        job_id = quantum_service.submit_quantum_job(payload)
        return {"job_id": job_id, "status": "queued", "classical_fallback": classical}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Submit failed: {e}")


@router.get("/status/{job_id}")
def job_status(job_id: str):
    """
    Poll job status. Response includes status and result when done.
    """
    j = quantum_service.get_job_status(job_id)
    if not j:
        raise HTTPException(status_code=404, detail="Job not found")
    return j
