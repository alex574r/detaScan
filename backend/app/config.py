"""
DentaScan — Configuración centralizada por entorno.
Lee variables del archivo .env y provee valores por defecto seguros.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── Aplicación ────────────────────────────────────────────────────────────
    APP_NAME: str = "DentaScan"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Sistema de análisis de radiografías dentales mediante visión artificial. "
        "Herramienta de apoyo diagnóstico — no reemplaza el criterio del odontólogo."
    )
    DEBUG: bool = False
    ENVIRONMENT: str = "production"  # development | staging | production

    # ─── Servidor ──────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:5500"]

    # ─── Seguridad / JWT ────────────────────────────────────────────────────────
    SECRET_KEY: str = "CAMBIA_ESTO_EN_PRODUCCION_usa_openssl_rand_hex_32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 horas

    # ─── Base de datos ─────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./dentascan.db"

    # ─── Almacenamiento de archivos ─────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    INPUT_DIR: Path = BASE_DIR / "input"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    MODELS_DIR: Path = BASE_DIR / "models_ml"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ─── Procesamiento de imágenes ─────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: set[str] = {".png", ".tif", ".tiff", ".dcm", ".jpg", ".jpeg"}
    PERIAPICAL_SIZE: tuple[int, int] = (512, 512)
    PANORAMIC_SIZE: tuple[int, int] = (1024, 512)

    # CLAHE — parámetros estándar para radiografías dentales
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_GRID_SIZE: tuple[int, int] = (8, 8)

    # Filtros de ruido
    GAUSSIAN_KERNEL_SIZE: int = 5
    MEDIAN_KERNEL_SIZE: int = 3

    # Detección de bordes Canny
    CANNY_LOW_THRESHOLD: int = 50
    CANNY_HIGH_THRESHOLD: int = 150

    # ─── Modelos ML ────────────────────────────────────────────────────────────
    RF_MODEL_PATH: str = "models_ml/rf_model.pkl"
    SVM_MODEL_PATH: str = "models_ml/svm_model.pkl"
    SCALER_PATH: str = "models_ml/scaler.pkl"

    # Clases del clasificador
    CLASS_LABELS: dict[int, str] = {
        0: "Diente Sano",
        1: "Caries Incipiente",
        2: "Caries Avanzada",
        3: "Absceso Periapical",
        4: "Lesión Ósea",
    }

    # ─── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/dentascan.log"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "case_sensitive": True}

    def create_directories(self) -> None:
        """Crea los directorios necesarios si no existen."""
        for directory in [self.INPUT_DIR, self.OUTPUT_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Retorna la instancia singleton de configuración."""
    settings = Settings()
    settings.create_directories()
    return settings
