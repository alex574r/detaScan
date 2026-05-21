"""
DentaScan — Modelo de Análisis (SQLAlchemy).
Cada registro representa un análisis completo de una radiografía dental.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Enum as SAEnum, JSON, Text
)
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class XRayType(str, enum.Enum):
    PERIAPICAL = "periapical"
    CORONAL = "coronal"
    PANORAMIC = "panoramic"
    BITEWING = "bitewing"
    UNKNOWN = "unknown"


class AnalysisStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Metadatos del archivo
    original_filename = Column(String(500), nullable=False)
    stored_filename = Column(String(500), nullable=False)  # UUID-based
    file_format = Column(String(20))  # PNG, DICOM, TIFF, JPG
    xray_type = Column(SAEnum(XRayType), default=XRayType.UNKNOWN)

    # Estado del pipeline
    status = Column(SAEnum(AnalysisStatus), default=AnalysisStatus.PENDING, nullable=False)
    error_message = Column(Text, nullable=True)

    # Resultados del procesamiento
    # Rutas a las imágenes procesadas (relativas a output/)
    output_preprocessed = Column(String(500), nullable=True)  # CLAHE + filtros
    output_edges = Column(String(500), nullable=True)          # Canny/Sobel
    output_mask = Column(String(500), nullable=True)           # Máscara de hallazgos
    output_annotated = Column(String(500), nullable=True)      # Imagen con bounding boxes
    output_histogram = Column(String(500), nullable=True)      # Histograma PNG

    # Features extraídas (12 características radiométricas)
    features = Column(JSON, nullable=True)

    # Resultados de clasificación ML
    predicted_class = Column(Integer, nullable=True)           # índice de clase
    predicted_label = Column(String(100), nullable=True)       # etiqueta legible
    confidence_score = Column(Float, nullable=True)            # confianza del modelo
    class_probabilities = Column(JSON, nullable=True)          # prob. por clase
    model_used = Column(String(50), nullable=True)             # "random_forest" | "svm"

    # Métricas de procesamiento
    processing_time_ms = Column(Float, nullable=True)

    # Hallazgos por región (lista de dicts con lesion_type, severity, coordenadas…)
    lesion_findings = Column(JSON, nullable=True)

    # Heatmap térmico de zonas de interés
    output_heatmap = Column(String(500), nullable=True)

    # Metadatos DICOM (si aplica)
    dicom_metadata = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    user = relationship("User", back_populates="analyses")

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} file={self.original_filename} status={self.status}>"
