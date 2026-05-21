"""
DentaScan — Conexión y gestión de la base de datos.
Usa SQLAlchemy con SQLite por defecto (reemplazable con PostgreSQL via DATABASE_URL).
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import get_settings

settings = get_settings()

# SQLite requiere este pragma para activar claves foráneas
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {},
    echo=settings.DEBUG,
)

# Activar claves foráneas en SQLite
if "sqlite" in SQLALCHEMY_DATABASE_URL:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency de FastAPI — provee una sesión de base de datos por request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea todas las tablas definidas en los modelos."""
    # Importar modelos para que SQLAlchemy los registre
    from app.models import user, analysis  # noqa: F401
    Base.metadata.create_all(bind=engine)
