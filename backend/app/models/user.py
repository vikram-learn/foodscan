from pydantic import BaseModel
from typing import Optional, List

class User(BaseModel):
    id: Optional[int] = None
    name: str
    email: Optional[str] = None
    abha_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None

    # For Ayurveda personalization:
    dosha_type: Optional[str] = None      # vata / pitta / kapha
    conditions: List[str] = []            # diabetes, hypertension, etc.

    class Config:
        orm_mode = True
