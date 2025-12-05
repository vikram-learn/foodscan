from typing import Optional, List
from sqlmodel import SQLModel, Field, Column, String, JSON

class UserTable(SQLModel, table=True):
    """
    SQLModel representation of a user.
    Keep a separate DB model (UserTable) from the Pydantic-only User model to
    allow DB-specific fields (id, created_at, etc).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column("name", String, nullable=False))
    email: Optional[str] = Field(default=None, sa_column=Column("email", String, unique=True, nullable=True))
    abha_id: Optional[str] = Field(default=None, sa_column=Column("abha_id", String, nullable=True))
    age: Optional[int] = Field(default=None)
    gender: Optional[str] = Field(default=None)

    # Ayurveda personalization
    dosha_type: Optional[str] = Field(default=None)  # vata / pitta / kapha

    # Store conditions as JSON array in the DB for simplicity
    # (SQLite supports storing JSON text; SQLModel will serialize Python list -> JSON)
    conditions: Optional[List[str]] = Field(default_factory=list, sa_column=Column("conditions", JSON, nullable=True))

    # You can add created_at, updated_at later (datetime fields)
