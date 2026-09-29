import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load variables from .env file (only matters locally)
load_dotenv()

BASE_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_DB_FILE = os.path.join(BASE_BACKEND_DIR, "marine_debris.db")
DEFAULT_SQLITE_URL = f"sqlite:///{CANONICAL_DB_FILE.replace(os.sep, '/')}"

# Render sets the RENDER environment variable automatically
ON_RENDER = os.getenv("RENDER") is not None

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL or DATABASE_URL == "sqlite:///./marine_debris.db":
    DATABASE_URL = DEFAULT_SQLITE_URL

# Render gives postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _make_engine(url: str):
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


try:
    engine = _make_engine(DATABASE_URL)
    with engine.connect() as conn:
        pass
except Exception as exc:
    if ON_RENDER:
        # On Render, do NOT fall back to SQLite: fail loudly so you see the problem in logs
        raise
    print(f"Database connection failed ({exc}). Falling back to SQLite development database.")
    DATABASE_URL = DEFAULT_SQLITE_URL
    engine = _make_engine(DATABASE_URL)


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