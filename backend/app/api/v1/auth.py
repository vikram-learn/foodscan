from fastapi import APIRouter

router = APIRouter()

@router.get("/abha/authorize")
def abha_authorize():
    """
    Placeholder: In a real app, this would redirect to ABHA consent manager.
    """
    return {"message": "ABHA consent flow placeholder"}

@router.get("/abha/profile")
def abha_profile():
    """
    Returns a demo ABHA-like medical record.
    Later will be replaced by actual ABHA API integration.
    """
    demo_record = {
        "name": "Test User",
        "age": 24,
        "conditions": ["prediabetes"],
        "latest_hba1c": 6.1,
        "latest_bp": "130/85",
        "notes": "This is demo medical data until ABHA integration is added."
    }
    return {"abha_profile": demo_record}
