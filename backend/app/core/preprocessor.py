"""
DentaScan — Módulo de Preprocesamiento clínico avanzado.

Pipeline clínico (RF-05/06/07 + mejoras):

  1. Normalización percentil (p1, p99)  → estandariza rango dinámico
  2. Corrección de iluminación          → resta fondo lento (top-hat blanco/negro)
  3. Filtro Bilateral                    → suaviza ruido preservando bordes
  4. Mediana (sal-y-pimienta)
  5. CLAHE multi-escala                  → fusiona tile-grids para textura + global
  6. Unsharp masking suave (opcional)    → realce de transiciones esmalte-dentina

Adicional:
  - data_augmentation(): generador de variaciones (rotación, brillo, contraste,
    flip, ruido, etc.) — usado por entrenamiento sintético.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import cv2

from app.config import get_settings
from app.exceptions.custom import ProcessingError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Pipeline avanzado para radiografías dentales.

    Diseñado para variaciones de iluminación, calidad, ángulo y anatomía.
    """

    def __init__(
        self,
        gaussian_ksize: int = None,
        median_ksize: int = None,
        clahe_clip_limit: float = None,
        clahe_tile_grid: tuple[int, int] = None,
    ) -> None:
        self.gaussian_ksize = gaussian_ksize or settings.GAUSSIAN_KERNEL_SIZE
        self.median_ksize = median_ksize or settings.MEDIAN_KERNEL_SIZE
        self.clahe_clip_limit = clahe_clip_limit or settings.CLAHE_CLIP_LIMIT
        self.clahe_tile_grid = clahe_tile_grid or settings.CLAHE_TILE_GRID_SIZE

        if self.gaussian_ksize % 2 == 0:
            self.gaussian_ksize += 1

        # CLAHE a múltiples escalas (textura local fina + contraste global)
        self._clahe_fine    = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,    tileGridSize=(16, 16))
        self._clahe_medium  = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,    tileGridSize=self.clahe_tile_grid)
        self._clahe_coarse  = cv2.createCLAHE(clipLimit=self.clahe_clip_limit*1.2, tileGridSize=(4, 4))

    # ─── API principal ────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline clínico completo.

        Devuelve uint8. Resistente a:
          - Imágenes muy oscuras o muy brillantes (normalización percentil)
          - Iluminación no uniforme (corrección de fondo)
          - Ruido cuántico y film grain (bilateral)
          - Diferencias de exposición entre adquisiciones (CLAHE multi-escala)
        """
        self._validate_input(image)

        # 1. Normalización percentil — fija el rango dinámico a [p1, p99]
        normalized = self.normalize_percentile(image)

        # 2. Corrección de iluminación — quita gradiente de fondo lento
        corrected = self.correct_illumination(normalized)

        # 3. Filtro bilateral — preserva bordes mientras suaviza superficies
        bilateral = self.apply_bilateral(corrected)

        # 4. Mediana — quita sal-y-pimienta sin tocar bordes
        denoised = self.apply_median(bilateral)

        # 5. CLAHE multi-escala — equilibra contraste local + global
        enhanced = self.apply_multiscale_clahe(denoised)

        logger.debug("Preprocesamiento clínico ok | shape=%s", enhanced.shape)
        return enhanced

    # ─── Etapas individuales ──────────────────────────────────────────────────

    def normalize_percentile(
        self,
        image: np.ndarray,
        low_pct: float = 1.0,
        high_pct: float = 99.0,
    ) -> np.ndarray:
        """
        Estira el rango dinámico al intervalo [p1, p99] → robusto a outliers
        (puntos brillantes de metal o esquinas oscuras vacías).
        """
        self._validate_input(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        lo = float(np.percentile(image, low_pct))
        hi = float(np.percentile(image, high_pct))
        if hi - lo < 1e-3:
            return image.copy()
        stretched = np.clip((image.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        return stretched.astype(np.uint8)

    def correct_illumination(
        self,
        image: np.ndarray,
        kernel_size: int = 51,
    ) -> np.ndarray:
        """
        Corrige iluminación no uniforme estimando el fondo con un blur
        muy grande y restándolo (rolling-ball aproximado).

        Después re-centra la imagen al rango uint8.
        """
        self._validate_input(image)
        # Estimar fondo: blur grande captura la iluminación de baja frecuencia
        background = cv2.medianBlur(image, kernel_size | 1)  # asegurar impar
        background = cv2.GaussianBlur(background, (kernel_size | 1, kernel_size | 1), 0)

        # Diferencia con el fondo (puede ser negativa)
        diff = image.astype(np.int16) - background.astype(np.int16)
        # Re-centrar al rango medio (128)
        corrected = np.clip(diff + int(np.mean(image)), 0, 255).astype(np.uint8)
        return corrected

    def apply_bilateral(
        self,
        image: np.ndarray,
        d: int = 7,
        sigma_color: float = 35.0,
        sigma_space: float = 7.0,
    ) -> np.ndarray:
        """
        Filtro bilateral — suaviza zonas homogéneas pero preserva bordes
        (esmalte-dentina, márgenes de lesión).
        """
        self._validate_input(image)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def apply_gaussian(self, image: np.ndarray) -> np.ndarray:
        """RF-05: Filtro gaussiano (compatibilidad con tests)."""
        self._validate_input(image)
        return cv2.GaussianBlur(image, (self.gaussian_ksize, self.gaussian_ksize), 0)

    def apply_median(self, image: np.ndarray) -> np.ndarray:
        """RF-06: Mediana — robusta a sal-y-pimienta."""
        self._validate_input(image)
        return cv2.medianBlur(image, self.median_ksize)

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """RF-07: CLAHE single-scale (compatibilidad)."""
        self._validate_input(image)
        return self._clahe_medium.apply(image)

    def apply_multiscale_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE a 3 escalas y fusión ponderada — mejora simultáneamente
        contraste fino (lesiones sutiles) y global (estructura mandibular).
        """
        self._validate_input(image)
        fine   = self._clahe_fine.apply(image)
        medium = self._clahe_medium.apply(image)
        coarse = self._clahe_coarse.apply(image)
        # Mezcla: privilegiar 'medium' (estándar clínico), añadir un poco de
        # fine para textura y coarse para luminosidad de fondo
        fused = (0.25 * fine.astype(np.float32) +
                 0.55 * medium.astype(np.float32) +
                 0.20 * coarse.astype(np.float32))
        return np.clip(fused, 0, 255).astype(np.uint8)

    def resize_standard(
        self,
        image: np.ndarray,
        xray_type: str = "periapical",
    ) -> np.ndarray:
        """
        Redimensiona según tipo de rx:
          - Periapical / coronal / bitewing → 512×512
          - Panorámica                       → 1024×512
        """
        size_map = {
            "periapical": settings.PERIAPICAL_SIZE,
            "bitewing":   settings.PERIAPICAL_SIZE,
            "coronal":    settings.PERIAPICAL_SIZE,
            "panoramic":  settings.PANORAMIC_SIZE,
        }
        target = size_map.get(xray_type.lower(), settings.PERIAPICAL_SIZE)
        return cv2.resize(image, target, interpolation=cv2.INTER_LINEAR)

    def apply_unsharp_mask(
        self, image: np.ndarray, strength: float = 1.2
    ) -> np.ndarray:
        """Realce conservador de bordes (post-CLAHE)."""
        self._validate_input(image)
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.5)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    # ─── Data Augmentation (entrenamiento) ────────────────────────────────────

    @staticmethod
    def augment(
        image: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        n_variations: int = 1,
    ) -> Iterator[np.ndarray]:
        """
        Generador de variaciones aleatorias clínicamente plausibles:
          - rotación ± 12°
          - flip horizontal
          - jitter de brillo / contraste
          - ruido gaussiano leve
          - gamma ± 0.25
        Útil para entrenamiento con datasets pequeños — preserva la patología
        sin generar artefactos imposibles en radiografía real.
        """
        rng = rng or np.random.default_rng()
        h, w = image.shape[:2]
        for _ in range(n_variations):
            img = image.copy()

            # Rotación pequeña
            angle = float(rng.uniform(-12, 12))
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Flip horizontal con 50% prob (la anatomía dental es ~simétrica)
            if rng.random() < 0.5:
                img = cv2.flip(img, 1)

            # Brillo/contraste
            alpha = float(rng.uniform(0.85, 1.15))  # contraste
            beta  = float(rng.uniform(-12, 12))     # brillo
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # Gamma
            gamma = float(rng.uniform(0.8, 1.25))
            inv_g = 1.0 / gamma
            table = ((np.arange(256) / 255.0) ** inv_g * 255).astype(np.uint8)
            img = cv2.LUT(img, table)

            # Ruido gaussiano leve
            noise = rng.normal(0, 4.0, img.shape)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            yield img

    @staticmethod
    def _validate_input(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise ProcessingError("La imagen debe ser un array NumPy válido.")
        if image.ndim != 2:
            raise ProcessingError(
                f"Se esperaba imagen en escala de grises (2D), "
                f"recibido shape={image.shape}"
            )


# Instancia global con parámetros por defecto
image_preprocessor = ImagePreprocessor()
