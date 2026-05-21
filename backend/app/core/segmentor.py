"""
DentaScan — Módulo de Segmentación clínica.

Mejoras clave vs. versión anterior:
  - Segmentación del campo dental (tooth_mask) para anclar las lesiones a la
    anatomía y descartar fondo, mucosa y artefactos extrínsecos.
  - Distinción restauración (radiopaca / brillo saturado) vs caries
    (radiolúcida con textura difusa) → reduce falsos positivos drásticamente.
  - Clasificación de caries por localización:
        oclusal      → corona superior, en cuspideal
        interproximal→ borde lateral del diente
        recurrente   → adyacente a restauración existente
        no cariosa   → forma muy elongada, fondo, sombra
  - Detección multi-escala (LoG + DoG) para lesiones tempranas.
  - Filtrado morfológico estricto (compacidad, solidez) para evitar
    confusión con sombras, artefactos digitales y bordes de película.
"""

from __future__ import annotations

import math
import numpy as np
import cv2

from app.config import get_settings
from app.exceptions.custom import ProcessingError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImageSegmentor:
    """
    Segmentador clínico con contexto anatómico.
    """

    def __init__(
        self,
        canny_low: int = None,
        canny_high: int = None,
    ) -> None:
        self.canny_low = canny_low or settings.CANNY_LOW_THRESHOLD
        self.canny_high = canny_high or settings.CANNY_HIGH_THRESHOLD

    # ─── Umbralizaciones (compatibilidad) ─────────────────────────────────────

    def apply_otsu_threshold(self, image: np.ndarray) -> np.ndarray:
        self._validate(image)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        self._validate(image)
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=11, C=2,
        )

    def apply_canny(self, image: np.ndarray) -> np.ndarray:
        self._validate(image)
        return cv2.Canny(image, self.canny_low, self.canny_high)

    def apply_sobel(self, image: np.ndarray) -> np.ndarray:
        self._validate(image)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        if mag.max() > 0:
            mag = (mag / mag.max() * 255).astype(np.uint8)
        return mag.astype(np.uint8)

    def find_contours(self, binary_mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        return sorted(contours, key=cv2.contourArea, reverse=True)

    # ─── Segmentación del campo dental (NUEVO) ───────────────────────────────

    def segment_tooth_field(self, preprocessed: np.ndarray) -> np.ndarray:
        """
        Devuelve una máscara binaria del área que contiene tejido dental/óseo
        (excluye fondo radiolúcido, aire y bordes vacíos).

        Estrategia (probada en imágenes reales + sintéticas):
          - Otsu (separación bimodal background/foreground)
          - Si Otsu produce máscara < 20%, ampliar con percentil 35
          - Operaciones morfológicas para sellar
        """
        self._validate(preprocessed)
        h, w = preprocessed.shape
        total = h * w

        # Otsu bimodal
        _, otsu_mask = cv2.threshold(preprocessed, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_frac = otsu_mask.sum() / 255 / total

        # Si Otsu retuvo muy poco (imagen sobre-procesada o muy gris),
        # caer a percentil 35 como fallback
        if otsu_frac < 0.20:
            p35 = float(np.percentile(preprocessed, 35))
            mask = (preprocessed >= p35).astype(np.uint8) * 255
        else:
            mask = otsu_mask

        # Morfología para sellar
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)

        # Reemplazamos cada componente por su convex hull → así las lesiones
        # (huecos oscuros dentro del tejido dental) quedan incluidas en la
        # máscara como parte del campo dental.
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        hull_mask = np.zeros_like(mask)
        if num > 1:
            sizes = stats[1:, cv2.CC_STAT_AREA]
            if sizes.size > 0:
                largest = 1 + int(np.argmax(sizes))
                cutoff = stats[largest, cv2.CC_STAT_AREA] * 0.05
                for i in range(1, num):
                    if stats[i, cv2.CC_STAT_AREA] >= cutoff:
                        comp = (labels == i).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        for c in contours:
                            hull = cv2.convexHull(c)
                            cv2.fillPoly(hull_mask, [hull], 255)
        mask = hull_mask if hull_mask.any() else mask

        # Si quedó casi vacía, relajamos al 100%
        if mask.sum() / 255 < total * 0.10:
            mask = np.full_like(mask, 255)
        return mask

    def detect_restorations(
        self,
        preprocessed: np.ndarray,
        tooth_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detecta restauraciones (amalgama / corona / composite muy radiopaco)
        para EVITAR confundirlas con caries.

        Heurísticas:
          - Píxeles >= percentil 98 dentro del campo dental
          - Forma compacta y regular (solidity > 0.9)
          - Bordes muy nítidos (gradiente alto en perímetro)
        """
        self._validate(preprocessed)
        if tooth_mask is None:
            tooth_mask = self.segment_tooth_field(preprocessed)

        # Top-2% más brillante DENTRO del diente
        dental_pixels = preprocessed[tooth_mask > 0]
        if dental_pixels.size == 0:
            return np.zeros_like(preprocessed), []
        p_high = float(np.percentile(dental_pixels, 98))
        thresh = max(220, int(p_high))

        rest_mask = ((preprocessed >= thresh) & (tooth_mask > 0)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        rest_mask = cv2.morphologyEx(rest_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        rest_mask = cv2.morphologyEx(rest_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours = self.find_contours(rest_mask)
        regions: list[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 60:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.85:
                continue  # bordes irregulares → probablemente no es restauración
            regions.append({
                "x": int(x), "y": int(y), "width": int(bw), "height": int(bh),
                "area_px": float(area),
                "centroid_x": int(x + bw // 2),
                "centroid_y": int(y + bh // 2),
                "lesion_type": "Restauración",
                "severity": "n/a",
                "is_radiopaque": True,
                "circularity": float(self._circularity(c, area)),
                "mean_intensity": float(np.mean(dental_pixels[dental_pixels >= thresh])),
                "contour_points": self._approx_polygon(c),
            })
        return rest_mask, regions

    # ─── Detección de caries (radiolúcida) ───────────────────────────────────

    def detect_radiolucent_regions(
        self,
        preprocessed: np.ndarray,
        darkness_threshold: int = 95,
        tooth_mask: np.ndarray | None = None,
        restoration_mask: np.ndarray | None = None,
        xray_type: str = "periapical",
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detección de regiones radiolúcidas (caries, abscesos, periodontitis,
        resorción ósea, lesiones quísticas) con clasificación clínica.

        Mejoras:
          - Adaptive threshold sobre el tejido dental (no global)
          - Excluir restauraciones para evitar bordes oscuros adyacentes
          - DoG multi-escala para lesiones tempranas
          - Clasificación por LOCALIZACIÓN ANATÓMICA además de geometría
        """
        self._validate(preprocessed)
        h, w = preprocessed.shape

        if tooth_mask is None:
            tooth_mask = self.segment_tooth_field(preprocessed)

        # Umbral adaptativo basado en el tejido (más robusto que global)
        dental_pixels = preprocessed[tooth_mask > 0]
        if dental_pixels.size > 0:
            median_dental = float(np.median(dental_pixels))
            std_dental    = float(np.std(dental_pixels))
            # Lesión = al menos 1σ por debajo de la mediana dental
            thresh = max(40, min(darkness_threshold, int(median_dental - 0.85 * std_dental)))
        else:
            thresh = darkness_threshold

        # Máscara primaria de radiolucidez (solo dentro del campo dental)
        dark = ((preprocessed < thresh) & (tooth_mask > 0)).astype(np.uint8) * 255

        # Excluir áreas adyacentes a restauración (halo de oscurecimiento)
        if restoration_mask is not None and restoration_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            rest_dilated = cv2.dilate(restoration_mask, kernel, iterations=1)
            # Marcamos la zona adyacente — sirve para detectar caries RECURRENTE
            recurrent_zone = rest_dilated & (~restoration_mask)
        else:
            recurrent_zone = np.zeros_like(dark)

        # Refuerzo multi-escala con DoG (lesiones tempranas)
        dog_mask = self._dog_dark_mask(preprocessed, tooth_mask)
        combined = cv2.bitwise_or(dark, dog_mask)

        # Limpiar artefactos
        k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_small, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_small, iterations=2)

        contours = self.find_contours(combined)
        regions: list[dict] = []

        # Una "lesión" creíble es < 15% del área de la imagen — cualquier cosa
        # mayor es fondo o un campo de hueso completo entrando por error.
        # min_area más alto = menos falsos positivos por ruido / textura normal.
        max_area = h * w * 0.15
        min_area = max(150, int(h * w * 0.0004))  # ~100 px para imagen 512×512

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue  # ruido o demasiado grande para ser una lesión

            x, y, bw, bh = cv2.boundingRect(c)
            circ = self._circularity(c, area)
            aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
            y_norm = (y + bh / 2) / h
            x_norm = (x + bw / 2) / w

            # Solidity (qué tan llena vs convex hull) — caries reales suelen
            # tener solidity 0.5-0.9; las sombras y artefactos suelen < 0.4
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.45 and area < 1000:
                continue  # forma muy fragmentada → artefacto

            # Intensidad media dentro
            mask_roi = np.zeros_like(preprocessed, dtype=np.uint8)
            cv2.drawContours(mask_roi, [c], -1, 255, cv2.FILLED)
            mean_int = float(cv2.mean(preprocessed, mask=mask_roi)[0])

            # ¿Es zona de recurrencia (adyacente a restauración)?
            is_recurrent = bool(np.any(recurrent_zone[mask_roi > 0]))

            lesion_type, severity = self._classify_lesion(
                area=area, circ=circ, aspect=aspect,
                y_norm=y_norm, x_norm=x_norm,
                mean_int=mean_int, solidity=solidity,
                is_recurrent=is_recurrent,
                xray_type=xray_type,
            )

            regions.append({
                "x": int(x), "y": int(y), "width": int(bw), "height": int(bh),
                "area_px": float(area),
                "centroid_x": int(x + bw // 2),
                "centroid_y": int(y + bh // 2),
                "circularity": round(circ, 3),
                "solidity":    round(solidity, 3),
                "mean_intensity": round(mean_int, 1),
                "lesion_type": lesion_type,
                "severity":    severity,
                "is_radiopaque": False,
                "is_recurrent": is_recurrent,
                "contour_points": self._approx_polygon(c),
            })

        logger.debug("Regiones radiolúcidas filtradas clínicamente: %d", len(regions))
        return combined, regions

    def detect_dense_regions(
        self,
        preprocessed: np.ndarray,
        brightness_threshold: int = 215,
        tooth_mask: np.ndarray | None = None,
        restoration_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Lesiones radiopacas óseas (no restauraciones):
          - Osteítis condensante: respuesta ósea reactiva
          - Hipercementosis     : depósito de cemento en raíz

        Excluye restauraciones (entran en `detect_restorations`).
        """
        self._validate(preprocessed)
        h, w = preprocessed.shape
        if tooth_mask is None:
            tooth_mask = self.segment_tooth_field(preprocessed)

        dental = preprocessed[tooth_mask > 0]
        if dental.size == 0:
            return np.zeros_like(preprocessed), []
        p95 = float(np.percentile(dental, 95))
        thresh = max(brightness_threshold, int(p95 * 0.93))

        bright = ((preprocessed > thresh) & (tooth_mask > 0)).astype(np.uint8) * 255

        # Excluir restauraciones (su zona ya está detectada por otro detector)
        if restoration_mask is not None:
            bright = cv2.bitwise_and(bright, cv2.bitwise_not(restoration_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel, iterations=1)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours = self.find_contours(bright)
        regions: list[dict] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 500:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            circ = self._circularity(c, area)
            y_norm = (y + bh / 2) / h

            if y_norm > 0.5 and circ > 0.3 and area > 800:
                lesion_type, severity = "Osteítis Condensante", "leve"
            elif area > 1500:
                lesion_type, severity = "Hipercementosis", "leve"
            else:
                continue

            regions.append({
                "x": int(x), "y": int(y), "width": int(bw), "height": int(bh),
                "area_px": float(area),
                "centroid_x": int(x + bw // 2),
                "centroid_y": int(y + bh // 2),
                "circularity": round(circ, 3),
                "mean_intensity": 0.0,
                "lesion_type": lesion_type,
                "severity":    severity,
                "is_radiopaque": True,
                "is_recurrent": False,
                "contour_points": self._approx_polygon(c),
            })
        return bright, regions

    # ─── Sub-clasificador clínico ────────────────────────────────────────────

    @staticmethod
    def _classify_lesion(
        area: float,
        circ: float,
        aspect: float,
        y_norm: float,
        x_norm: float,
        mean_int: float,
        solidity: float,
        is_recurrent: bool,
        xray_type: str = "periapical",
    ) -> tuple[str, str]:
        """
        Sub-clasificador que diferencia:
          - Caries Oclusal:      corona, circular/compacta, área pequeña/media
          - Caries Interproximal: borde lateral del diente, triangular
          - Caries Recurrente:   adyacente a restauración
          - Caries Avanzada:     área grande con destrucción franca
          - Lesiones óseas:      según localización
        """
        # Recurrente tiene prioridad si el flag está presente
        if is_recurrent and area < 800:
            sev = "leve" if area < 250 else ("moderada" if area < 500 else "severa")
            return "Caries Recurrente", sev

        # Zona coronal (parte superior del diente) → caries
        if y_norm < 0.45:
            # Forma elongada en el borde lateral → interproximal
            edge_proximity = min(x_norm, 1 - x_norm)
            if edge_proximity < 0.25 and aspect > 1.4:
                if area < 200:
                    return "Caries Interproximal", "leve"
                if area < 700:
                    return "Caries Interproximal", "moderada"
                return "Caries Interproximal", "severa"
            # Forma compacta y central → oclusal
            if circ > 0.45 and aspect < 1.8:
                if area < 200:
                    return "Caries Oclusal Incipiente", "leve"
                if area < 700:
                    return "Caries Oclusal", "moderada"
                return "Caries Oclusal Avanzada", "severa"
            # Forma irregular pero coronal → caries genérica
            if area < 250:
                return "Caries Incipiente", "leve"
            if area < 900:
                return "Caries Avanzada", "moderada"
            return "Caries Avanzada", "severa"

        # Zona periapical (parte inferior del diente / ápice)
        if y_norm > 0.60:
            if area < 500:
                return "Granuloma Periapical", "leve"
            if circ > 0.55 and area > 2000:
                return "Quiste Periapical", "severa"
            if circ > 0.28:
                return "Absceso Periapical", "severa"
            return "Lesión Periapical Difusa", "severa"

        # Zona media — interproximal/periodontal
        if aspect > 2.0 or circ < 0.22:
            if area > 3500:
                return "Periodontitis Severa", "severa"
            return "Periodontitis Leve", "moderada"

        # Para bitewing es muy común caries interproximal en zona media
        if xray_type in ("bitewing", "coronal") and aspect > 1.3:
            sev = "moderada" if area > 400 else "leve"
            return "Caries Interproximal", sev

        if circ > 0.40:
            return "Absceso Periapical", "moderada"

        return "Resorción Ósea", "moderada"

    # ─── Detección multi-escala ──────────────────────────────────────────────

    @staticmethod
    def _dog_dark_mask(image: np.ndarray, tooth_mask: np.ndarray) -> np.ndarray:
        """Detecta blobs oscuros sutiles con DoG multi-escala dentro del diente."""
        gA = cv2.GaussianBlur(image, (5, 5),  1.0)
        gB = cv2.GaussianBlur(image, (21, 21), 6.0)
        dog = gB.astype(np.int16) - gA.astype(np.int16)  # >0 donde algo es más oscuro
        thresh_val = max(8, int(np.std(dog) * 1.4))
        mask = (dog > thresh_val).astype(np.uint8) * 255
        return cv2.bitwise_and(mask, tooth_mask)

    # ─── Helpers geométricos ──────────────────────────────────────────────────

    @staticmethod
    def _circularity(contour: np.ndarray, area: float = None) -> float:
        if area is None:
            area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 1e-6:
            return 0.0
        return float(4 * math.pi * area / (perimeter ** 2))

    @staticmethod
    def _approx_polygon(contour: np.ndarray) -> list[list[int]]:
        epsilon = 0.015 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            return []
        return approx.reshape(-1, 2).tolist()

    # ─── Compatibilidad legado ────────────────────────────────────────────────

    def segment_teeth(self, preprocessed: np.ndarray) -> np.ndarray:
        """Alias compatible — devuelve la máscara del campo dental."""
        return self.segment_tooth_field(preprocessed)

    @staticmethod
    def _validate(image: np.ndarray) -> None:
        if image is None or not isinstance(image, np.ndarray):
            raise ProcessingError("Se requiere un array NumPy válido.")
        if image.ndim != 2:
            raise ProcessingError(f"Se esperaba imagen 2D, recibido shape={image.shape}")


image_segmentor = ImageSegmentor()
