# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Relative imports (correct when running uvicorn backend.app.main:app)
from .routes.conversation_svg import router as conversation_svg_router

# v1 routers (relative imports)
from .api.v1 import scan, auth, user, quantum

# DB init + engine + services (relative imports)
from .db.init_db import init_db
from .db.session import get_engine
from .services import quantum_worker

# ---------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------
app = FastAPI(
    title="FoodScan-X API",
    version="0.1.0",
)

# ---------------------------------------------------
# CORS Middleware (required for mobile app)
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # later: restrict to app domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Startup: initialize DB and start background worker
# ---------------------------------------------------
@app.on_event("startup")
def on_startup():
    # initialize DB (creates tables / seed as configured)
    init_db()

    # start background quantum worker in a separate thread (non-blocking)
    try:
        engine = get_engine()
        quantum_worker.start_worker_on_thread(engine)
    except Exception:
        logging.getLogger("quantum_worker").exception("Failed to start quantum worker")


# ---------------------------------------------------
# Include Routers
# conversation_svg router exposes POST /conversation_svg
# ---------------------------------------------------
app.include_router(conversation_svg_router)

# Note: scan.router uses prefix "/scan" so we mount it under "/api/v1"
app.include_router(scan.router, prefix="/api/v1", tags=["scan"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(user.router, prefix="/api/v1/user", tags=["user"])
app.include_router(quantum.router, prefix="/api/v1", tags=["quantum"])

# ---------------------------------------------------
# Root route
# ---------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "foodscan-x"}
