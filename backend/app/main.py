# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import v1 routers (we import modules here; includes are below after app is created)
from app.api.v1 import scan
from app.api.v1 import auth
from app.api.v1 import user
from app.api.v1 import quantum

# DB init
from app.db.init_db import init_db
# backend/app/main.py  (edit the on_startup function)
from app.db.session import get_engine
from app.services import quantum_worker


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
# Startup: initialize DB
# ---------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()

# ---------------------------------------------------
# Include Routers
# Note: scan.router uses prefix "/scan" so we mount it under "/api/v1"
#       resulting endpoints: /api/v1/scan/...
# ---------------------------------------------------
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
@app.on_event("startup")
def on_startup():
    init_db()
    # start background worker (non-blocking)
    try:
        engine = get_engine()
        quantum_worker.start_worker_on_thread(engine)
    except Exception:
        # if get_engine isn't available for some reason, worker won't start — safe fallback
        import logging
        logging.getLogger("quantum_worker").exception("Failed to start quantum worker")
