# /backend/routes/conversation_svg.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import re

router = APIRouter()

class ConvReq(BaseModel):
    text: Optional[str] = ""
    sessionId: Optional[str] = None

@router.post("/conversation_svg")
async def conversation_svg(payload: ConvReq):
    t = (payload.text or "").strip()
    reply = "I couldn't parse the scan. Try 'scan 320' or include a number."
    calories = None

    m = re.search(r"(\d{2,4})", t)
    if m:
        calories = int(m.group(1))

    bodyScale = 1.0
    animationHint = "Talk"
    emotion = "neutral"
    animationDuration = 1.8

    if calories is not None:
        if calories <= 150:
            bodyScale = 0.85
            reply = f"Low calories: {calories} kcal — the avatar looks thinner."
            emotion = "surprised"
            animationHint = "Think"
            animationDuration = 1.4
        elif calories <= 350:
            bodyScale = 1.0
            reply = f"Moderate calories: {calories} kcal — normal body."
            emotion = "neutral"
            animationHint = "Talk"
            animationDuration = 1.6
        elif calories <= 700:
            bodyScale = 1.15
            reply = f"High calories: {calories} kcal — avatar gets a bit chubby."
            emotion = "happy"
            animationHint = "Happy"
            animationDuration = 1.6
        else:
            bodyScale = 1.32
            reply = f"Very high: {calories} kcal — avatar gets noticeably larger."
            emotion = "surprised"
            animationHint = "Wave"
            animationDuration = 1.7
    else:
        if re.search(r"\b(good|great|nice)\b", t, re.I):
            reply = "Nice to hear that!"
            animationHint = "Happy"
            emotion = "happy"
            animationDuration = 1.2
        elif re.search(r"\b(bad|sad|not)\b", t, re.I):
            reply = "Oh, sorry to hear that."
            animationHint = "Sad"
            emotion = "sad"
            animationDuration = 1.4
        elif re.search(r"\b(hello|hi)\b", t, re.I):
            reply = "Hello! Send me the scan or calorie number."
            animationHint = "Talk"
            animationDuration = 1.4
        else:
            reply = "Send a scan result (e.g., 'scan 320') or say hello."
            animationHint = "Listen"
            emotion = "neutral"
            animationDuration = 1.1

    return {
        "reply": reply,
        "calories": calories,
        "bodyScale": bodyScale,
        "animationHint": animationHint,
        "emotion": emotion,
        "animationDuration": animationDuration
    }
