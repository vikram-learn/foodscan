# backend/app/api/v1/scan.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Any, Dict
from app.services import ml_service
from app.db.session import get_session
import traceback
from io import BytesIO

router = APIRouter(prefix="/scan", tags=["scan"])

# Try to import persistence models if present; otherwise continue gracefully
try:
    from app.db.models import ScanRecord, UserTable  # may not exist in minimal scaffold
except Exception:
    ScanRecord = None
    UserTable = None

@router.post("/image")
async def scan_image(file: UploadFile = File(...), session=Depends(get_session)):
    """
    Accept an image upload, run ML analysis (via ml_service), save a ScanRecord
    if the model exists and return JSON { "result": scan_result, "saved": True/False, "scan_id": <id or null> }.
    """
    try:
        # Read file bytes
        contents = await file.read()
        fileobj = BytesIO(contents)

        # call the ml service to analyze and get a scan_result dict
        scan_result = ml_service.analyze_fileobj(fileobj)

        # Optionally persist to DB if ScanRecord model is available
        saved = False
        scan_id = None
        if ScanRecord is not None:
            try:
                rec = ScanRecord(
                    filename=file.filename,
                    result=scan_result,
                )
                session.add(rec)
                session.commit()
                session.refresh(rec)
                scan_id = getattr(rec, "id", None)
                saved = True
            except Exception:
                # Do not fail entire request if DB persistence has issues
                session.rollback()

        return {"result": scan_result, "saved": saved, "scan_id": scan_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scan failed: {e}")
