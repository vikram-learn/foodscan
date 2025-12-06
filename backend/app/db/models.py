from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, JSON, Text

# ===============================
# Database models for FoodScan-X
# ===============================

class UserTable(SQLModel, table=True):
    __tablename__ = "usertable"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(..., description="User name")

    # email as TEXT + unique
    email: Optional[str] = Field(
        default=None,
        index=True,
        sa_column=Column("email", Text, unique=True)
    )

    abha_id: Optional[str] = Field(default=None, index=True)
    age: Optional[int] = Field(default=None)
    gender: Optional[str] = Field(default=None)
    dosha_type: Optional[str] = Field(default=None)

    # medical history stored as JSON array
    conditions: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON)
    )


class QuantumJob(SQLModel, table=True):
    __tablename__ = "quantumjob"

    id: Optional[int] = Field(default=None, primary_key=True)
    status: str = Field(default="queued", index=True)

    input_payload: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON)
    )

    engine: Optional[str] = Field(default=None, index=True)
    created_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
    completed_at: Optional[int] = Field(default=None)


class ScanRecord(SQLModel, table=True):
    __tablename__ = "scanrecord"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(
        default=None,
        foreign_key="usertable.id",
        index=True
    )
    filename: Optional[str] = Field(default=None)

    scan_payload: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON)
    )

    created_at: int = Field(default_factory=lambda: int(datetime.utcnow().timestamp()))
