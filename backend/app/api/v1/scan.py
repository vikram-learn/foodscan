# backend/app/api/v1/scan.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, select
import shutil
import tempfile
import time
import os
import logging
from typing import Optional

from app.db.session import get_session
from app.db.models import ScanRecord, UserTable
from app.services import ml_service
from app.services import quantum_service

logger = logging.getLogger("api.scan")
router = APIRouter()

@router.post("/image", name="scan_image")
def scan_image(file: UploadFile = File(...), user_email: Optional[str] = Form(None), session: Session = Depends(get_session)):
    """
    Receive image file, analyze it, optionally attach to user by email,
    run the quantum optimizer (best-effort), save a ScanRecord and return the result.
    - file: multipart form file (image)
    - user_email (optional): if provided, tries to link scan to the user
    """
    # save upload to a temporary file
    try:
        suffix = os.path.splitext(file.filename or "upload")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="scan_") as tmp:
            temp_path = tmp.name
            # stream write
            file.file.seek(0)
            shutil.copyfileobj(file.file, tmp)
    except Exception as e:
        logger.exception("failed to save uploaded file")
        return JSONResponse(status_code=400, content={"error": "upload_failed", "detail": str(e)})

    try:
        # call your ML analysis entrypoint (ml_service.analyze_image) — returns dict result
        result = ml_service.analyze_image(temp_path)
    except Exception as e:
        logger.exception("analyze_image failed")
        # ensure temp file cleaned up
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": "analysis_failed", "detail": str(e)})

    # Optionally run quantum optimizer (best-effort)
    try:
        # quantum_service.run_quantum_optimize_sync expects scan-like dict
        qres = quantum_service.run_quantum_optimize_sync(result)
    except Exception as e:
        logger.exception("quantum optimizer failed, falling back to classical result")
        qres = {"engine": "error", "error": str(e)}

    # Save ScanRecord to DB (best-effort)
    scan_id = None
    saved = False
    try:
        with session:
            # if user_email provided, attempt to link user
            user = None
            if user_email:
                user = session.exec(select(UserTable).where(UserTable.email == user_email)).first()
            scan_payload = result  # should be JSON-serializable dict from ml_service

            rec = ScanRecord(
                user_id=(user.id if user else None),
                filename=os.path.basename(temp_path),
                scan_payload=scan_payload,
                created_at=int(time.time()),
            )
            session.add(rec)
            session.commit()
            session.refresh(rec)
            scan_id = rec.id
            saved = True
    except Exception as e:
        logger.exception("failed to save ScanRecord (non-fatal)")
        # do not fail the entire request for a DB save problem

    # cleanup temp file
    try:
        os.remove(temp_path)
    except Exception:
        pass

    # Build response (keeps previous shape)
    response = {
        "scan_id": scan_id,
        "saved": saved,
        "result": result,
    }
    # attach quantum result if available
    response["result"]["quantum"] = qres

    return JSONResponse(status_code=200, content=response)
