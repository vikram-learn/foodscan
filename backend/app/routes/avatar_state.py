# backend/app/routes/avatar_state.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import time

router = APIRouter()

# In-memory store for demo (small projects). Replace with DB later.
_AVATAR_STATE = {
    "bodyScale": 1.0,
    "emotion": "neutral",
    "last_updated": None
}

class AvatarState(BaseModel):
    bodyScale: float
    emotion: Optional[str] = "neutral"

@router.get("/avatar_state")
async def get_avatar_state():
    return _AVATAR_STATE

@router.post("/avatar_state")
async def set_avatar_state(state: AvatarState):
    _AVATAR_STATE["bodyScale"] = max(0.6, min(1.6, float(state.bodyScale)))
    _AVATAR_STATE["emotion"] = state.emotion or "neutral"
    _AVATAR_STATE["last_updated"] = int(time.time())
    return {"ok": True, "state": _AVATAR_STATE}
