import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

BASE_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DB_FILE = os.path.join(BASE_BACKEND_DIR, "marine_debris.db")
DEFAULT_SQLITE_URL = f"sqlite:///{CANONICAL_DB_FILE.replace(os.sep, '/')}"

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL or DATABASE_URL == "sqlite:///./marine_debris.db":
    DATABASE_URL = DEFAULT_SQLITE_URL

try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        pass
except Exception as exc:
    print(f"Database connection to {DATABASE_URL} failed ({exc}). Falling back to SQLite development database.")
    DATABASE_URL = DEFAULT_SQLITE_URL
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})



# Create a session factory - each request gets its own session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that our models will inherit from
Base = declarative_base()


# Dependency function used by FastAPI routes to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
