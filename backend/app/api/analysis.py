"""
DentaScan — Router de Análisis.
Endpoints para subir radiografías y consultar resultados del pipeline.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.analysis import Analysis, AnalysisStatus, XRayType
from app.models.user import User
from app.schemas.analysis import AnalysisResponse, AnalysisListItem, AnalysisSummary
from app.services.image_service import image_processing_service
from app.core.dental_validator import dental_validator
from app.core.loader import image_loader
from app.utils.security import get_current_active_user
from app.utils.logger import get_logger
from app.exceptions.custom import (
    UnsupportedFormatError, FileTooLargeError, ProcessingError
)

router = APIRouter(prefix="/analyses", tags=["Análisis"])
settings = get_settings()
logger = get_logger(__name__)


@router.post("/validate", status_code=status.HTTP_200_OK)
async def validate_image(
    file: UploadFile = File(..., description="Imagen a validar antes del análisis"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Valida si una imagen es una radiografía/foto dental aceptable.
    Útil para pre-validación en el cliente antes de enviar el análisis completo.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Formato no soportado: '{ext}'",
        )
    tmp_name = f"_validate_{uuid.uuid4().hex}{ext}"
    tmp_path = settings.INPUT_DIR / tmp_name
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img, _ = image_loader.load(tmp_path)
        result = dental_validator.validate(img)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.post("/", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Radiografía dental (PNG, TIFF, DICOM, JPEG)"),
    xray_type: str = Form(default="unknown"),
    model: str = Form(default="random_forest"),
    analysis_mode: str = Form(default="all", description="'all' | 'targeted'"),
    min_confidence: float = Form(default=0.0, ge=0.0, le=1.0),
    lesion_types: str = Form(default="", description="CSV de tipos de lesión a buscar"),
    sensitivity: str = Form(default="normal", description="'screening' | 'normal' | 'high' | 'research'"),
    use_yolo: bool = Form(default=False, description="Activa detección YOLOv8 adicional"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Sube una radiografía dental e inicia el pipeline de análisis.
    El procesamiento ocurre en background; consulta GET /analyses/{id} para el resultado.
    """
    # Validar extensión antes de guardar
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Formato no soportado: '{ext}'. Formatos válidos: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Guardar en disco con nombre UUID (no sobreescribe originales)
    stored_name = f"{uuid.uuid4().hex}{ext}"
    input_path = settings.INPUT_DIR / stored_name
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error guardando el archivo: {exc}")

    # ── Validación dental pre-análisis ───────────────────────────────────────
    # Verifica que la imagen sea realmente una radiografía / foto dental.
    # Si no, eliminar el archivo y devolver 422 con mensaje técnico claro.
    try:
        loaded_img, _ = image_loader.load(input_path)
    except Exception as exc:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No se pudo decodificar la imagen: {exc}",
        )

    validation = dental_validator.validate(loaded_img)
    if not validation.accepted:
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        logger.warning(
            "Imagen rechazada por validador dental | usuario=%s | razones=%s",
            current_user.email, validation.reasons,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": validation.rejection_message,
                "reasons": validation.reasons,
                "metrics": validation.metrics,
                "dental_likelihood": validation.dental_likelihood,
                "quality_score":     validation.quality_score,
            },
        )
    logger.info(
        "Validación dental OK | usuario=%s | dental=%.2f | calidad=%.2f",
        current_user.email, validation.dental_likelihood, validation.quality_score,
    )

    # Crear registro en BD con estado pendiente
    analysis = Analysis(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_filename=stored_name,
        file_format=ext.upper().strip("."),
        xray_type=XRayType(xray_type) if xray_type in XRayType.__members__.values() else XRayType.UNKNOWN,
        status=AnalysisStatus.PENDING,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Ejecutar pipeline en background
    config = {
        "analysis_mode":  analysis_mode if analysis_mode in ("all", "targeted") else "all",
        "min_confidence": float(min_confidence),
        "lesion_types":   [s.strip() for s in lesion_types.split(",") if s.strip()],
        "sensitivity":    sensitivity if sensitivity in ("screening", "normal", "high", "research") else "normal",
        "use_yolo":       bool(use_yolo),
    }

    background_tasks.add_task(
        _run_pipeline,
        analysis_id=analysis.id,
        file_path=input_path,
        xray_type=xray_type,
        model=model,
        config=config,
    )

    logger.info("Análisis creado: id=%d | archivo=%s | usuario=%s",
                analysis.id, file.filename, current_user.email)

    return analysis


@router.get("/", response_model=list[AnalysisListItem])
def list_analyses(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Lista los análisis del usuario autenticado, ordenados por fecha."""
    return (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/summary", response_model=AnalysisSummary)
def get_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Resumen estadístico de los análisis del usuario."""
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    class_dist: dict[str, int] = {}
    for a in analyses:
        if a.predicted_label:
            class_dist[a.predicted_label] = class_dist.get(a.predicted_label, 0) + 1

    return AnalysisSummary(
        total=len(analyses),
        completed=sum(1 for a in analyses if a.status == AnalysisStatus.COMPLETED),
        failed=sum(1 for a in analyses if a.status == AnalysisStatus.FAILED),
        pending=sum(1 for a in analyses if a.status in (AnalysisStatus.PENDING, AnalysisStatus.PROCESSING)),
        class_distribution=class_dist,
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Retorna el detalle completo de un análisis, incluyendo resultados y rutas de imágenes."""
    analysis = _get_analysis_or_404(analysis_id, current_user.id, db)
    return analysis


@router.delete("/", status_code=status.HTTP_200_OK)
def delete_all_analyses(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Elimina TODOS los análisis del usuario actual junto con sus archivos
    de salida. Operación destructiva — el frontend debe confirmar antes.

    Registrado ANTES de /{analysis_id} para evitar conflictos de ruta en Starlette.

    Returns:
        dict: { deleted: int } con la cantidad de análisis eliminados
    """
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    if not analyses:
        return {"deleted": 0}

    output_fields = [
        "output_preprocessed", "output_edges", "output_mask",
        "output_annotated", "output_heatmap", "output_histogram",
    ]
    files_removed = 0
    for a in analyses:
        for field in output_fields:
            path_str = getattr(a, field, None)
            if path_str:
                try:
                    Path(path_str).unlink(missing_ok=True)
                    files_removed += 1
                except Exception:
                    pass
        db.delete(a)
    db.commit()

    logger.info(
        "Historial completo eliminado | user_id=%d | %d análisis | %d archivos",
        current_user.id, len(analyses), files_removed,
    )
    return {"deleted": len(analyses), "files_removed": files_removed}


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Elimina un análisis y sus archivos de salida asociados."""
    analysis = _get_analysis_or_404(analysis_id, current_user.id, db)

    # Limpiar archivos de output
    for field in ["output_preprocessed", "output_edges", "output_mask",
                  "output_annotated", "output_heatmap", "output_histogram"]:
        path_str = getattr(analysis, field, None)
        if path_str:
            try:
                Path(path_str).unlink(missing_ok=True)
            except Exception:
                pass

    db.delete(analysis)
    db.commit()
    logger.info("Análisis eliminado: id=%d", analysis_id)


# ─── Helper privado ───────────────────────────────────────────────────────────

def _get_analysis_or_404(analysis_id: int, user_id: int, db: Session) -> Analysis:
    analysis = db.query(Analysis).filter(
        Analysis.id == analysis_id,
        Analysis.user_id == user_id,
    ).first()
    if not analysis:
        raise HTTPException(status_code=404, detail=f"Análisis {analysis_id} no encontrado.")
    return analysis


def _run_pipeline(analysis_id: int, file_path: Path, xray_type: str, model: str, config: dict | None = None) -> None:
    """Función de background: ejecuta el pipeline y actualiza la BD."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return

        analysis.status = AnalysisStatus.PROCESSING
        db.commit()

        result = image_processing_service.process(
            file_path=file_path,
            analysis_id=analysis_id,
            xray_type=xray_type,
            model=model,
            config=config or {},
        )

        clf = result["classification"]
        output = result["output_paths"]

        analysis.status = AnalysisStatus.COMPLETED
        analysis.features = result["features"]
        analysis.predicted_class = clf["predicted_class"]
        analysis.predicted_label = clf["predicted_label"]
        analysis.confidence_score = clf["confidence_score"]
        analysis.class_probabilities = clf["class_probabilities"]
        analysis.model_used = clf["model_used"]
        analysis.processing_time_ms = result["processing_time_ms"]
        analysis.lesion_findings = result.get("lesion_findings")
        analysis.dicom_metadata = result["metadata"].get("dicom")
        analysis.output_preprocessed = output.get("preprocessed")
        analysis.output_edges = output.get("edges")
        analysis.output_mask = output.get("mask")
        analysis.output_annotated = output.get("annotated")
        analysis.output_heatmap = output.get("heatmap")
        analysis.output_histogram = output.get("histogram")
        db.commit()
        logger.info("Pipeline completado para analysis_id=%d", analysis_id)

    except (UnsupportedFormatError, FileTooLargeError, ProcessingError) as exc:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = str(exc)
        db.commit()
        logger.error("Error en pipeline analysis_id=%d: %s", analysis_id, exc)
    except Exception as exc:
        analysis.status = AnalysisStatus.FAILED
        analysis.error_message = f"Error inesperado: {exc}"
        db.commit()
        logger.exception("Error inesperado en analysis_id=%d", analysis_id)
    finally:
        db.close()
