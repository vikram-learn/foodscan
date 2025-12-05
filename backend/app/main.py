# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.api.v1 import scan
from app.api.v1 import auth
from app.api.v1 import user

# DB init
from app.db.init_db import init_db
# after the other includes
from app.api.v1 import quantum
app.include_router(quantum.router, prefix="/api/v1", tags=["quantum"])


# ---------------------------------------------------
#  Create FastAPI app
# ---------------------------------------------------
app = FastAPI(
    title="FoodScan-X API",
    version="0.1.0",
)


# ---------------------------------------------------
#  CORS Middleware (required for mobile app)
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # later: restrict to app domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
#  Startup: initialize DB
# ---------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------
#  Include Routers
# ---------------------------------------------------

# IMPORTANT:
# scan.router already has prefix="/scan"
# so we mount it on "/api/v1" to produce:
#    /api/v1/scan/image
app.include_router(scan.router, prefix="/api/v1", tags=["scan"])

# auth endpoints → /api/v1/auth/...
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])

# user endpoints → /api/v1/user/...
app.include_router(user.router, prefix="/api/v1/user", tags=["user"])


# ---------------------------------------------------
#  Root route
# ---------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "service": "foodscan-x"}
