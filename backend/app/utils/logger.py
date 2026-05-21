"""
DentaScan — Configuración del logger.
Escribe a consola y archivo con rotación automática.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import get_settings

settings = get_settings()


def get_logger(name: str) -> logging.Logger:
    """Retorna un logger configurado para el módulo indicado."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Evitar duplicar handlers

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler de archivo con rotación (10 MB, mantiene 5 archivos)
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
