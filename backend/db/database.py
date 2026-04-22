"""
backend/db/database.py
=======================
SQLAlchemy database engine and session management.

Supports:
  - SQLite (development — zero config, file-based)
  - PostgreSQL (production — set DATABASE_URL env var)

Usage:
  from backend.db.database import get_db, init_db
  
  # In FastAPI route:
  async def route(db: Session = Depends(get_db)):
      users = db.query(User).all()
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base

# ── Database URL ──────────────────────────────────────────────────────────
# SQLite for development (no setup required)
# PostgreSQL for production: postgresql://user:pass@host:5432/quantail
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "sqlite:///./quantail.db"   # stored in current directory
)

# ── Engine config ─────────────────────────────────────────────────────────
if DATABASE_URL.startswith("sqlite"):
    # SQLite: single-threaded, no pool
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=os.environ.get("DB_ECHO", "false").lower() == "true",
    )
    # Enable WAL mode for better concurrent reads
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL: connection pool
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=os.environ.get("DB_ECHO", "false").lower() == "true",
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Call on startup."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {DATABASE_URL.split('?')[0]}")


def get_db():
    """FastAPI dependency — yields a DB session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
