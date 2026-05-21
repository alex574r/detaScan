"""
DentaScan — Detector dental con YOLOv8.

Usa ultralytics. Carga lazy del modelo desde:
  1. Path local en settings.MODELS_DIR / "yolo_dental.pt" si existe
  2. Hub de Ultralytics (descarga automática primera vez)

Sin un .pt entrenado en datos dentales, el modelo COCO genérico NO detecta
patologías clínicas — solo se usa como detector de regiones de interés
(bounding boxes con clases genéricas). El usuario puede dropear un modelo
dental específico (.pt) en `models_ml/yolo_dental.pt` para reemplazarlo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading

import numpy as np

from app.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

_yolo_available: Optional[bool] = None
_model = None
_lock = threading.Lock()


def yolo_available() -> bool:
    """Detecta si ultralytics está instalado (cacheado)."""
    global _yolo_available
    if _yolo_available is not None:
        return _yolo_available
    try:
        import ultralytics                    # noqa: F401
        _yolo_available = True
    except ImportError as exc:
        logger.warning("ultralytics no disponible: %s", exc)
        _yolo_available = False
    return _yolo_available


def _load_model():
    """Carga lazy del modelo YOLO. Thread-safe."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        from ultralytics import YOLO

        # Prioridad 1: modelo dental específico en models_ml/
        custom_path = settings.BASE_DIR / "models_ml" / "yolo_dental.pt"
        if custom_path.exists():
            logger.info("Cargando YOLO dental personalizado: %s", custom_path)
            _model = YOLO(str(custom_path))
        else:
            # Fallback: YOLOv8 nano genérico (COCO) — solo detección genérica
            logger.info(
                "Sin yolo_dental.pt en %s — usando YOLOv8n genérico (COCO). "
                "Para detección clínica real coloque un .pt entrenado en datos dentales.",
                custom_path.parent,
            )
            _model = YOLO("yolov8n.pt")  # descarga automática ~6MB

        return _model


def detect_regions(image: np.ndarray, conf_threshold: float = 0.20) -> list[dict]:
    """
    Ejecuta YOLO sobre la imagen y retorna detecciones.

    Args:
        image: ndarray 2D o 3D (uint8)
        conf_threshold: confianza mínima para incluir detección

    Returns:
        Lista de dicts con: x, y, width, height, confidence, class_name
    """
    if not yolo_available():
        return []

    if image.ndim == 2:
        img = np.stack([image, image, image], axis=-1).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    try:
        model = _load_model()
        results = model.predict(img, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names  # dict {class_id: name}
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().item())
                conf = float(box.conf[0].cpu().item())
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                detections.append({
                    "x":          int(x1),
                    "y":          int(y1),
                    "width":      int(x2 - x1),
                    "height":     int(y2 - y1),
                    "centroid_x": int((x1 + x2) / 2),
                    "centroid_y": int((y1 + y2) / 2),
                    "area_px":    float((x2 - x1) * (y2 - y1)),
                    "confidence": round(conf, 3),
                    "lesion_type": f"YOLO: {names.get(cls_id, str(cls_id))}",
                    "severity":   "moderada" if conf > 0.5 else "leve",
                    "circularity": 0.0,
                    "mean_intensity": 0.0,
                    "is_radiopaque": False,
                    "source": "yolov8",
                })
        return detections
    except Exception as exc:
        logger.exception("Error en YOLO: %s", exc)
        return []
