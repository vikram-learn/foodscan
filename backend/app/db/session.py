# backend/app/db/session.py
"""
Database engine / session helpers for FoodScan-X.

Provides:
 - get_engine() -> SQLAlchemy engine (used by worker/startup)
 - get_session()  -> FastAPI dependency yielding sqlmodel.Session
 - A default DB path (./foodscan.db) to match the rest of the repo

Also exports `engine` for modules that import it directly (legacy code).
"""

import os
from typing import Generator, Optional

from sqlmodel import create_engine, Session, SQLModel

# Read DB path from env if set; fall back to local file in backend/
DB_FILE = os.getenv("FOODSCAN_DB", os.path.abspath("./foodscan.db"))
DATABASE_URL = f"sqlite:///{DB_FILE}"

# Engine is module-level to allow reuse; created lazily
_engine = None

def get_engine() -> "Engine":
    """
    Return a SQLAlchemy Engine instance. Lazily creates it once.
    Other modules (worker, main) import this to use the same engine.
    """
    global _engine
    if _engine is None:
        # echo=True is helpful for dev (SQL logging). Set False in production.
        _engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
    return _engine

def init_db() -> None:
    """
    Create DB tables if missing. Call at app startup.
    """
    engine = get_engine()
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a session and closes it after the request.
    Usage in routes: `session: Session = Depends(get_session)`
    """
    engine = get_engine()
    with Session(engine) as session:
        yield session

# --- Legacy compatibility: export engine variable for modules that import it directly ---
# Create the engine at import time so `from app.db.session import engine` works.
# This keeps compatibility with older bits of the codebase (like app/db/init_db.py).
engine = get_engine()
