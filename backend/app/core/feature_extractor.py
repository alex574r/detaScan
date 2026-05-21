"""
DentaScan — Módulo de Extracción de Características.

12 features radiométricas base (compatibilidad con modelos existentes)
  + features de textura avanzadas:
    - Haralick GLCM (contraste, homogeneidad, energía, correlación)
    - LBP (Local Binary Patterns)
    - Estadísticos de orden superior (skewness, kurtosis, entropía)
    - Densidad de Gabor multi-orientación

Estas se devuelven separadas en `extract_extended()` para no romper el
contrato de 12-features del scaler entrenado.
"""

from __future__ import annotations

import numpy as np
import cv2

from app.exceptions.custom import FeatureExtractionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """
    Extrae el vector de 12 features radiométricas básicas + features
    avanzadas opcionales (textura GLCM, LBP, Gabor) cuando se requieran.
    """

    # ─── 12 features base (firma estable para los modelos sklearn) ────────────

    def extract(self, image: np.ndarray) -> dict[str, float]:
        if image is None or image.ndim != 2:
            raise FeatureExtractionError("Se requiere imagen 2D en escala de grises.")

        try:
            features: dict[str, float] = {}
            features["media"]   = float(np.mean(image))
            features["std"]     = float(np.std(image))
            features["min_px"]  = float(np.min(image))
            features["max_px"]  = float(np.max(image))

            edges = cv2.Canny(image, 50, 150)
            features["bordes_mean"] = float(np.mean(edges))

            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_mag = np.sqrt(gx**2 + gy**2)
            features["sobel_mean"] = float(np.mean(sobel_mag))

            h, w = image.shape
            hh, hw = h // 2, w // 2
            features["zona_tl"] = float(np.mean(image[:hh, :hw]))
            features["zona_tr"] = float(np.mean(image[:hh, hw:]))
            features["zona_bl"] = float(np.mean(image[hh:, :hw]))
            features["zona_br"] = float(np.mean(image[hh:, hw:]))

            features["prop_oscuros"] = float(np.mean(image < 80))

            features["asimetria"] = float(
                abs(np.mean(image[:, :hw]) - np.mean(image[:, hw:]))
            )

            return features
        except Exception as exc:
            raise FeatureExtractionError(f"Error en extracción de features: {exc}") from exc

    def extract_to_array(self, image: np.ndarray) -> np.ndarray:
        feats = self.extract(image)
        keys = [
            "media", "std", "min_px", "max_px",
            "bordes_mean", "sobel_mean",
            "zona_tl", "zona_tr", "zona_bl", "zona_br",
            "prop_oscuros", "asimetria",
        ]
        return np.array([feats[k] for k in keys], dtype=np.float32)

    # ─── Texture extendido (no rompe el scaler) ───────────────────────────────

    def extract_extended(self, image: np.ndarray) -> dict[str, float]:
        """
        Features adicionales útiles para la calibración de confianza
        (textura local + orientaciones).

        Devuelve un dict separado — no se pasa al modelo ML clásico.
        """
        if image is None or image.ndim != 2:
            raise FeatureExtractionError("Se requiere imagen 2D.")

        out: dict[str, float] = {}

        # Estadísticos de orden superior
        flat = image.flatten().astype(np.float64)
        mean = float(np.mean(flat))
        std  = float(np.std(flat))
        if std > 0:
            out["skewness"] = float(np.mean(((flat - mean) / std) ** 3))
            out["kurtosis"] = float(np.mean(((flat - mean) / std) ** 4) - 3)
        else:
            out["skewness"] = 0.0
            out["kurtosis"] = 0.0
        out["entropy"] = self._entropy(image)

        # GLCM-light (sin scikit-image: aproximación 8-bins)
        glcm_feats = self._glcm_features(image)
        out.update(glcm_feats)

        # LBP-light (sin scikit-image: implementación local 3×3)
        out["lbp_uniformity"] = self._lbp_uniformity(image)

        # Densidad de respuesta Gabor (4 orientaciones)
        out["gabor_energy"] = self._gabor_energy(image)

        # Densidad de "blobs oscuros" (potenciales caries oclusales)
        out["dark_blob_density"] = self._dark_blob_density(image)

        return out

    # ─── Texture helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _entropy(image: np.ndarray) -> float:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1e-9)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist)))

    @staticmethod
    def _glcm_features(image: np.ndarray, levels: int = 8) -> dict[str, float]:
        """
        Aproximación rápida a Haralick GLCM con 8 niveles de gris y offset (1,0).

        Devuelve: contraste, homogeneidad, energía, correlación.
        """
        # Cuantizar a 8 niveles
        q = (image.astype(np.uint16) * levels // 256).astype(np.uint8)
        h, w = q.shape

        # Co-ocurrencia con vecino derecho
        i = q[:, :-1].flatten()
        j = q[:,  1:].flatten()
        glcm = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(glcm, (i, j), 1)
        glcm = glcm + glcm.T  # simetría
        s = glcm.sum()
        if s > 0:
            glcm /= s

        # Coordenadas
        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        diff   = (ii - jj).astype(np.float64)

        contrast    = float(np.sum(glcm * diff ** 2))
        homogeneity = float(np.sum(glcm / (1.0 + diff ** 2)))
        energy      = float(np.sum(glcm ** 2))

        # Correlación
        mu_i  = float(np.sum(ii * glcm))
        mu_j  = float(np.sum(jj * glcm))
        var_i = float(np.sum(glcm * (ii - mu_i) ** 2))
        var_j = float(np.sum(glcm * (jj - mu_j) ** 2))
        if var_i > 0 and var_j > 0:
            correlation = float(np.sum(glcm * (ii - mu_i) * (jj - mu_j) /
                                       np.sqrt(var_i * var_j)))
        else:
            correlation = 0.0

        return {
            "glcm_contrast":    contrast,
            "glcm_homogeneity": homogeneity,
            "glcm_energy":      energy,
            "glcm_correlation": correlation,
        }

    @staticmethod
    def _lbp_uniformity(image: np.ndarray) -> float:
        """
        Uniformity del histograma LBP 3×3. Valores altos = textura repetitiva
        (esmalte sano); valores bajos = textura heterogénea (lesión).
        """
        # Padding 1px
        padded = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        center = image
        lbp = np.zeros_like(center, dtype=np.uint8)
        # 8 vecinos
        offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        for k, (dy, dx) in enumerate(offsets):
            neighbor = padded[1+dy:1+dy+center.shape[0], 1+dx:1+dx+center.shape[1]]
            lbp |= ((neighbor >= center).astype(np.uint8) << k)

        hist = np.bincount(lbp.flatten(), minlength=256).astype(np.float64)
        hist /= max(hist.sum(), 1e-9)
        # Uniformity = sum(p_i^2)
        return float(np.sum(hist ** 2))

    @staticmethod
    def _gabor_energy(image: np.ndarray) -> float:
        """Energía promedio de banco Gabor a 4 orientaciones (0, 45, 90, 135°)."""
        img_f = image.astype(np.float32) / 255.0
        energy = 0.0
        for theta in (0, np.pi/4, np.pi/2, 3*np.pi/4):
            k = cv2.getGaborKernel((11, 11), 3.0, theta, 6.0, 0.5, 0, ktype=cv2.CV_32F)
            r = cv2.filter2D(img_f, cv2.CV_32F, k)
            energy += float(np.mean(np.abs(r)))
        return energy / 4.0

    @staticmethod
    def _dark_blob_density(image: np.ndarray) -> float:
        """
        Detecta blobs oscuros usando DoH (Determinant of Hessian) aproximado.
        Devuelve densidad normalizada (fracción de px que son centros candidatos).
        """
        # Difference-of-Gaussians como detector de blobs simple
        g1 = cv2.GaussianBlur(image, (5, 5), 1.0)
        g2 = cv2.GaussianBlur(image, (15, 15), 4.0)
        dog = g2.astype(np.int16) - g1.astype(np.int16)  # blobs oscuros -> positivos
        # Umbral relativo
        thresh = max(8, int(np.std(dog) * 1.5))
        blobs = (dog > thresh).astype(np.uint8)
        return float(blobs.mean())


feature_extractor = FeatureExtractor()
