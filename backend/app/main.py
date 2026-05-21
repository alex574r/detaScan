"""
DentaScan — Punto de entrada de la aplicación FastAPI.
Registra routers, middleware CORS, manejo de excepciones y archivos estáticos.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import init_db
from app.api import auth, analysis
from app.exceptions.custom import (
    DentaScanException, ImageLoadError, UnsupportedFormatError,
    FileTooLargeError, ProcessingError, ModelNotFoundError,
)
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS + ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(analysis.router, prefix="/api/v1")

    # ── Archivos estáticos — imágenes de output ───────────────────────────────
    output_dir = settings.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    # ── Manejo de excepciones del dominio ─────────────────────────────────────
    @app.exception_handler(UnsupportedFormatError)
    async def unsupported_format_handler(request: Request, exc: UnsupportedFormatError):
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(FileTooLargeError)
    async def file_too_large_handler(request: Request, exc: FileTooLargeError):
        return JSONResponse(status_code=413, content={"detail": str(exc)})

    @app.exception_handler(ProcessingError)
    async def processing_error_handler(request: Request, exc: ProcessingError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(DentaScanException)
    async def dentascan_exception_handler(request: Request, exc: DentaScanException):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # ── Eventos de ciclo de vida ──────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        logger.info("Iniciando DentaScan v%s...", settings.APP_VERSION)
        init_db()
        settings.create_directories()
        logger.info("Base de datos inicializada. Servidor listo.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("DentaScan apagándose.")

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/api/health", tags=["Sistema"])
    async def health_check():
        from app.core.classifier import dental_classifier
        return {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "ml_models_loaded": dental_classifier.is_trained(),
        }

    return app


app = create_app()
