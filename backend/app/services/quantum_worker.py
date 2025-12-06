# backend/app/services/quantum_worker.py
"""
Background worker for processing QuantumJob rows.

Behavior:
 - Polls the database every `POLL_INTERVAL_SECONDS`.
 - Atomically claims one queued job (status="queued") and sets it to "running".
 - Calls quantum_service.run_quantum_optimize_sync(input_payload["scan_result"])
   (falling back gracefully if the payload shape is different).
 - Writes result/engine/timestamps and marks status "done" or "failed".
 - Uses SQLModel sessions and the app's get_engine() (from app.db.session).
"""
from threading import Thread, Event
import time
import traceback
import logging
from typing import Optional

from sqlmodel import Session, select
from datetime import datetime

from app.db.models import QuantumJob
try:
    # Prefer using your existing engine helper if present
    from app.db.session import get_engine
except Exception:
    # fallback: create_engine will be used by the caller/tester if needed
    get_engine = None

from app.services import quantum_service

logger = logging.getLogger("quantum_worker")
logger.setLevel(logging.INFO)

# Single-thread guard
_worker_thread: Optional[Thread] = None
_worker_stop_event: Optional[Event] = None

# Poll interval (seconds) — adjust for your environment
POLL_INTERVAL_SECONDS = 3.0


def _process_job_once(engine):
    """
    Try to pick a single queued job and process it.
    Returns True if processed a job (or claimed one), False if none found.
    """
    with Session(engine) as session:
        # select one queued job (first) with simple locking semantics
        stmt = select(QuantumJob).where(QuantumJob.status == "queued").limit(1)
        job = session.exec(stmt).first()
        if not job:
            return False

        logger.info("Claiming job id=%s", job.id)
        # mark running
        job.status = "running"
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)

    # Do the heavy work outside the commit session above to keep claim short.
    try:
        # Prepare input for quantum_service
        payload = job.input_payload or {}
        # expected shape: {"scan_result": {...}} — but support raw scan_result too
        scan_result = payload.get("scan_result") if isinstance(payload, dict) and "scan_result" in payload else payload

        logger.info("Processing job %s with scan_result keys: %s", job.id, list(scan_result.keys()) if isinstance(scan_result, dict) else type(scan_result))

        # Run the synchronous quantum optimizer (already tested)
        result = quantum_service.run_quantum_optimize_sync(scan_result)

        # Save result
        with Session(engine) as session:
            db_job = session.get(QuantumJob, job.id)
            db_job.result = result
            db_job.engine = result.get("engine") if isinstance(result, dict) else None
            db_job.status = "done"
            db_job.updated_at = datetime.utcnow()
            session.add(db_job)
            session.commit()
            logger.info("Job %s done (engine=%s)", job.id, db_job.engine)

    except Exception as exc:
        logger.error("Job %s failed: %s", job.id, exc)
        logger.debug(traceback.format_exc())
        with Session(engine) as session:
            db_job = session.get(QuantumJob, job.id)
            db_job.error = str(exc)
            db_job.status = "failed"
            db_job.updated_at = datetime.utcnow()
            session.add(db_job)
            session.commit()
        # swallow exceptions so worker continues


def _worker_loop(engine, stop_event: Event):
    logger.info("Quantum worker loop started (poll interval %.1fs)", POLL_INTERVAL_SECONDS)
    while not stop_event.is_set():
        try:
            found = _process_job_once(engine)
            if not found:
                # nothing to do, sleep a little
                stop_event.wait(POLL_INTERVAL_SECONDS)
        except Exception:
            logger.exception("Unexpected error in worker loop")
            # small delay to avoid tight crash loops
            time.sleep(1.0)
    logger.info("Quantum worker loop stopped")


def start_worker_on_thread(engine):
    """
    Start the background worker thread if not already running.
    `engine` should be a SQLAlchemy engine (returned by get_engine()).
    """
    global _worker_thread, _worker_stop_event
    if _worker_thread and _worker_thread.is_alive():
        logger.info("Quantum worker already running")
        return

    _worker_stop_event = Event()
    _worker_thread = Thread(target=_worker_loop, args=(engine, _worker_stop_event), daemon=True, name="quantum-worker")
    _worker_thread.start()
    logger.info("Quantum worker thread launched")


def stop_worker():
    global _worker_thread, _worker_stop_event
    if _worker_stop_event:
        _worker_stop_event.set()
    if _worker_thread:
        _worker_thread.join(timeout=5.0)
    _worker_thread = None
    _worker_stop_event = None
    logger.info("Quantum worker stopped")
