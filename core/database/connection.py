"""
Database Connection — Engine & Session
=========================================
Konfigurasi koneksi database menggunakan SQLAlchemy.

Mendukung 2 mode:
  - SQLite (development): Default
  - MySQL (production): Set DB_USE_MYSQL=true

Seluruh timestamp menggunakan WIB (UTC+7).
"""

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from core.config import (
    BASE_DIR, DATA_DIR,
    DB_USE_MYSQL, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME,
)
from core.database.models import Base

# ==============================================================================
# Engine
# ==============================================================================
if DB_USE_MYSQL:
    DATABASE_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False,
    )
    DB_DISPLAY_NAME = f"MySQL @ {DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    SQLITE_PATH = DATA_DIR / "air_quality.db"
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )
    DB_DISPLAY_NAME = f"SQLite @ {SQLITE_PATH.name}"

# ==============================================================================
# Session
# ==============================================================================
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ==============================================================================
# Helpers
# ==============================================================================
def get_db():
    """Dependency injection untuk FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Buat semua tabel di database."""
    Base.metadata.create_all(bind=engine)
    mode = "MySQL" if DB_USE_MYSQL else "SQLite (dev)"
    print(f"[OK] Database initialized — Mode: {mode} | {DB_DISPLAY_NAME}")
