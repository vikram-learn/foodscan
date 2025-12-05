from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = "sqlite:///./foodscan.db"

# Create database engine
engine = create_engine(
    DATABASE_URL,
    echo=True  # shows SQL logs in terminal — useful for debugging; turn off later
)

# Dependency to get DB session
def get_session():
    with Session(engine) as session:
        yield session
