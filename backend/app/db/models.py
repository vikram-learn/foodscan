# backend/app/db/models.py
from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON

# --- User table (existing) ---
class UserTable(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str = Field(index=True, sa_column=Column("email", nullable=False, unique=True))
    abha_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

    # Ayurveda personalization
    dosha_type: Optional[str] = None   # vata/pitta/kapha
    conditions: Optional[List[str]] = Field(default_factory=list, sa_column=Column(JSON), nullable=False)

# --- Quantum job table (existing) ---
class QuantumJob(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    status: str = Field(default="queued", index=True)  # queued/processing/done/failed
    engine: Optional[str] = None
    input_payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON), nullable=True)
    result: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON), nullable=True)
    error: Optional[str] = None

# --- New: Scan record table ---
class ScanRecord(SQLModel, table=True):
    """
    Stores a single scan result returned by the ML/Quantum pipeline.

    Fields:
      - user_id: optional FK to UserTable.id (we keep it simple and not enforce foreign key constraints)
      - filename: stored filename if image saved
      - scan_payload: JSON blob with full detection, nutrition_estimate, ayurveda, meta etc.
      - created_at: integer timestamp (unix) — we let other layers set it
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, index=True)
    filename: Optional[str] = None
    scan_payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON), nullable=True)
    created_at: Optional[int] = None
