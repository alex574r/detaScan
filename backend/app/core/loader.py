"""
DentaScan — Módulo de Carga de Imágenes.
Requisitos: RF-01 (carga), RF-02 (compatibilidad), RF-03 (DICOM), RF-04 (matriz NumPy).

Soporta: PNG, TIFF, DICOM (.dcm), JPEG.
El módulo NO sobreescribe los archivos originales.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings
from app.exceptions.custom import (
    DicomReadError,
    ImageLoadError,
    UnsupportedFormatError,
    FileTooLargeError,
)
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImageLoader:
    """
    Carga radiografías dentales en distintos formatos y las convierte
    a matrices NumPy normalizadas a 8 bits para el pipeline de OpenCV.
    """

    SUPPORTED_EXTENSIONS = {".png", ".tif", ".tiff", ".dcm", ".jpg", ".jpeg"}

    def __init__(self) -> None:
        self._max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    # ─── Método principal ─────────────────────────────────────────────────────

    def load(self, file_path: str | Path) -> tuple[np.ndarray, dict]:
        """
        Carga una imagen desde disco y retorna:
            - imagen: array NumPy uint8 en escala de grises
            - metadata: dict con info del archivo y DICOM si aplica
        """
        path = Path(file_path)

        if not path.exists():
            raise ImageLoadError(f"Archivo no encontrado: {path}")

        file_size = path.stat().st_size
        if file_size > self._max_bytes:
            raise FileTooLargeError(
                f"Archivo {path.name} ({file_size / 1e6:.1f} MB) "
                f"supera el límite de {settings.MAX_FILE_SIZE_MB} MB"
            )

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Formato '{ext}' no soportado. "
                f"Formatos válidos: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        logger.info("Cargando imagen: %s (%.1f KB)", path.name, file_size / 1024)

        if ext == ".dcm":
            return self._load_dicom(path)
        else:
            return self._load_standard(path, ext)

    def load_from_bytes(self, data: bytes, filename: str) -> tuple[np.ndarray, dict]:
        """Carga una imagen desde bytes (útil para uploads HTTP)."""
        ext = Path(filename).suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"Formato '{ext}' no soportado.")

        if len(data) > self._max_bytes:
            raise FileTooLargeError(f"Archivo supera el límite de {settings.MAX_FILE_SIZE_MB} MB")

        if ext == ".dcm":
            return self._load_dicom_bytes(data)

        # Decodificar imagen estándar desde buffer
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ImageLoadError("No se pudo decodificar la imagen desde los bytes proporcionados.")

        img_8bit = self._normalize_to_8bit(img)
        metadata = {"filename": filename, "format": ext.upper().strip("."), "shape": img.shape}
        return img_8bit, metadata

    # ─── Loaders específicos ─────────────────────────────────────────────────

    def _load_standard(self, path: Path, ext: str) -> tuple[np.ndarray, dict]:
        """Carga PNG, TIFF, JPEG usando OpenCV con soporte de alta profundidad de bits."""
        # IMREAD_ANYDEPTH preserva 16-bit; IMREAD_GRAYSCALE convierte a grises
        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ImageLoadError(f"OpenCV no pudo leer el archivo: {path}")

        original_depth = img.dtype
        img_8bit = self._normalize_to_8bit(img)

        metadata = {
            "filename": path.name,
            "format": ext.upper().strip("."),
            "shape": img.shape,
            "original_depth": str(original_depth),
            "file_size_kb": path.stat().st_size / 1024,
        }

        logger.debug(
            "Imagen cargada: %s | shape=%s | depth=%s",
            path.name, img.shape, original_depth,
        )
        return img_8bit, metadata

    def _load_dicom(self, path: Path) -> tuple[np.ndarray, dict]:
        """Carga un archivo DICOM usando pydicom y extrae metadatos clínicos."""
        try:
            import pydicom
            from pydicom.pixels import apply_windowing
        except ImportError:
            raise DicomReadError(
                "pydicom no está instalado. Ejecute: pip install pydicom"
            )

        try:
            ds = pydicom.dcmread(str(path))
        except Exception as exc:
            raise DicomReadError(f"Error al leer DICOM {path.name}: {exc}") from exc

        # Extraer pixel array
        try:
            pixel_array = ds.pixel_array
        except AttributeError as exc:
            raise DicomReadError(f"El DICOM no contiene pixel_array: {exc}") from exc

        # Aplicar Window/Level si está disponible (estándar para radiografías dentales)
        # W=3000, L=500 son valores típicos en HU para radiografías dentales
        img = pixel_array.astype(np.float32)
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            wc = float(ds.WindowCenter) if not hasattr(ds.WindowCenter, "__iter__") else float(ds.WindowCenter[0])
            ww = float(ds.WindowWidth) if not hasattr(ds.WindowWidth, "__iter__") else float(ds.WindowWidth[0])
            low = wc - ww / 2
            high = wc + ww / 2
            img = np.clip(img, low, high)
            img = ((img - low) / (ww)) * 255.0
        else:
            img = self._normalize_to_8bit(img)

        img_8bit = np.clip(img, 0, 255).astype(np.uint8)

        # Extraer metadatos DICOM relevantes (sin datos del paciente por privacidad)
        dicom_metadata = self._extract_dicom_metadata(ds)
        metadata = {
            "filename": path.name,
            "format": "DICOM",
            "shape": img_8bit.shape,
            "dicom": dicom_metadata,
        }

        logger.info("DICOM cargado: %s | shape=%s", path.name, img_8bit.shape)
        return img_8bit, metadata

    def _load_dicom_bytes(self, data: bytes) -> tuple[np.ndarray, dict]:
        """Carga DICOM desde bytes en memoria."""
        try:
            import pydicom
        except ImportError:
            raise DicomReadError("pydicom no está instalado.")

        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            pixel_array = ds.pixel_array.astype(np.float32)
            img_8bit = self._normalize_to_8bit(pixel_array).astype(np.uint8)
            return img_8bit, {"format": "DICOM", "shape": img_8bit.shape, "dicom": self._extract_dicom_metadata(ds)}
        except Exception as exc:
            raise DicomReadError(f"Error procesando DICOM desde bytes: {exc}") from exc

    # ─── Utilidades ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_to_8bit(img: np.ndarray) -> np.ndarray:
        """
        Normalización Min-Max al rango [0, 255].
        Preserva la relación tonal: esmalte=brillante, lesiones=oscuro.
        """
        img_float = img.astype(np.float64)
        min_val, max_val = img_float.min(), img_float.max()
        if max_val == min_val:
            return np.zeros_like(img, dtype=np.uint8)
        normalized = (img_float - min_val) / (max_val - min_val) * 255.0
        return normalized.astype(np.uint8)

    @staticmethod
    def _extract_dicom_metadata(ds) -> dict:
        """Extrae metadatos técnicos del DICOM (sin PII del paciente)."""
        fields = {
            "Modality": "Modalidad",
            "Manufacturer": "Fabricante",
            "ManufacturerModelName": "Modelo",
            "KVP": "kVp",
            "ExposureTime": "Tiempo_exposicion_ms",
            "Rows": "Filas",
            "Columns": "Columnas",
            "BitsAllocated": "Bits_asignados",
            "BitsStored": "Bits_almacenados",
            "PixelSpacing": "Espaciado_pixel_mm",
            "ImageType": "Tipo_imagen",
        }
        metadata = {}
        for tag, label in fields.items():
            value = getattr(ds, tag, None)
            if value is not None:
                try:
                    metadata[label] = str(value)
                except Exception:
                    pass
        return metadata


# Instancia global reutilizable
image_loader = ImageLoader()
