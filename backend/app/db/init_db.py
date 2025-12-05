from sqlmodel import SQLModel
from .session import engine
from .models import UserTable

def init_db():
    """
    Create tables if they do not exist.
    This runs at FastAPI startup.
    """
    SQLModel.metadata.create_all(engine)