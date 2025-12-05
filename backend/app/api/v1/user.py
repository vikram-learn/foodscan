from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from sqlmodel import Session, select

from app.models.user import User               # Pydantic model
from app.db.models import UserTable            # DB model
from app.db.session import get_session
from app.api.v1 import auth as auth_module

router = APIRouter()

# Temporary in-memory storage — still used until we migrate link-abha + get-user
fake_user_db = []

# ---------------------------------------
# REGISTER USER → USE DATABASE NOW
# ---------------------------------------
@router.post("/register")
def register_user(user: User, session: Session = Depends(get_session)):
    # Check whether user already exists in DB
    statement = select(UserTable).where(UserTable.email == user.email)
    existing_user = session.exec(statement).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    # Create DB user entry
    db_user = UserTable(
        name=user.name,
        email=user.email,
        abha_id=user.abha_id,
        age=user.age,
        gender=user.gender,
        dosha_type=user.dosha_type,
        conditions=user.conditions,
    )

    session.add(db_user)
    session.commit()
    session.refresh(db_user)

    return {"message": "User registered successfully", "user": db_user}


# ---------------------------------------
# LINK ABHA — STILL USING IN-MEMORY STORAGE
# (Will migrate to DB after we update get-user)
# ---------------------------------------
@router.post("/link-abha")
def link_abha(email: str, abha_id: str, session: Session = Depends(get_session)):
    """
    Link an ABHA ID to the actual database user.
    Also merges demo ABHA health data (conditions).
    """

    # Look up user in DB
    statement = select(UserTable).where(UserTable.email == email)
    db_user = session.exec(statement).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update ABHA ID
    db_user.abha_id = abha_id

    # Fetch demo ABHA health data and merge conditions
    try:
        demo = auth_module.abha_profile()
        abha_profile = demo.get("abha_profile", {})

        conditions = abha_profile.get("conditions", [])
        for cond in conditions:
            if cond not in db_user.conditions:
                db_user.conditions.append(cond)

    except Exception:
        # If demo API fails, still save ABHA ID
        pass

    # Save updated user
    session.add(db_user)
    session.commit()
    session.refresh(db_user)

    return {"message": "ABHA ID linked successfully", "user": db_user}
# ---------------------------------------
# GET USER — STILL USING IN-MEMORY STORAGE
# (Next step: migrate to database)
# ---------------------------------------
@router.get("/get")
def get_user(email: Optional[str] = None, session: Session = Depends(get_session)):
    """
    Retrieve a user by email from the REAL database.
    If no email provided, returns all users from the database.
    """

    if email:
        statement = select(UserTable).where(UserTable.email == email)
        db_user = session.exec(statement).first()

        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        return {"user": db_user}

    # Return all users from DB
    all_users = session.exec(select(UserTable)).all()
    return {"users": all_users}
