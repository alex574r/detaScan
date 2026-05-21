"""
DentaScan — Schemas Pydantic para Análisis.
"""

from datetime import datetime
from typing import Any
from pydantic import BaseModel
from app.models.analysis import XRayType, AnalysisStatus


class AnalysisCreate(BaseModel):
    xray_type: XRayType = XRayType.UNKNOWN


class FeatureVector(BaseModel):
    """Las 12 características radiométricas extraídas por OpenCV."""
    media: float
    std: float
    min_px: float
    max_px: float
    bordes_mean: float      # media de bordes Canny
    sobel_mean: float       # gradiente Sobel promedio
    zona_tl: float          # media del cuadrante superior-izquierdo
    zona_tr: float          # media del cuadrante superior-derecho
    zona_bl: float          # media del cuadrante inferior-izquierdo
    zona_br: float          # media del cuadrante inferior-derecho
    prop_oscuros: float     # proporción de píxeles oscuros (radiolucidez)
    asimetria: float        # diferencia entre cuadrantes izq/der


class ClassificationResult(BaseModel):
    predicted_class: int
    predicted_label: str
    confidence_score: float
    class_probabilities: dict[str, float]
    model_used: str


class AnalysisResponse(BaseModel):
    id: int
    user_id: int
    original_filename: str
    file_format: str | None
    xray_type: XRayType
    status: AnalysisStatus
    error_message: str | None

    # Rutas de imágenes procesadas (accesibles vía /static/)
    output_preprocessed: str | None
    output_edges: str | None
    output_mask: str | None
    output_annotated: str | None
    output_heatmap: str | None
    output_histogram: str | None

    # Resultados numéricos
    features: dict[str, Any] | None
    predicted_class: int | None
    predicted_label: str | None
    confidence_score: float | None
    class_probabilities: dict[str, Any] | None
    model_used: str | None
    processing_time_ms: float | None
    lesion_findings: list[Any] | None
    dicom_metadata: dict[str, Any] | None

    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class AnalysisListItem(BaseModel):
    id: int
    original_filename: str
    xray_type: XRayType
    status: AnalysisStatus
    predicted_label: str | None
    confidence_score: float | None
    output_annotated: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisSummary(BaseModel):
    total: int
    completed: int
    failed: int
    pending: int
    class_distribution: dict[str, int]
