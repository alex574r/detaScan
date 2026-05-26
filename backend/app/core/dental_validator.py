"""
DentaScan — Validador inteligente de imágenes dentales.

Decide si una imagen cargada es realmente una radiografía dental / foto
intraoral válida ANTES de gastar recursos en el pipeline de análisis.

Estrategia multi-señal (todas baratas en CPU):

  1. Sanity checks básicos
     - Resolución mínima (>= 150 × 150)
     - No corruption (cargable)

  2. Análisis de color
     - Saturación HSV alta -> foto a color -> penalización (no rechazo inmediato)
     - Imágenes con ligero tinte azul/verde (film digitalizado) -> se tolera
     - Imagen casi monocromática -> rx OK

  3. Métricas de calidad
     - Sharpness, Contraste, Ruido

  4. Score de "dentalidad" — 7 señales
     a) Bimodalidad del histograma
     b) Dispersión de intensidades (X-ray: rango amplio de oscuro a brillante)
     c) Densidad de bordes — scoring trapezoidal (acepta 0.03–0.40, rx típica 0.05–0.25)
     d) Regiones brillantes compactas (esmalte/coronas/restauraciones)
     e) Ausencia de texto/screenshot
     f) Firma oscura de rx (bandas negras de aire — panorámicas)
     g) Contraste local vs global

  5. Bonificaciones
     - Relación de aspecto panorámica (ancho:alto = 1.4–3.5)
     - Si la calidad técnica es alta, umbral de dentalidad se relaja

  6. Veredict: dental_likelihood >= 0.42 Y quality_score >= 0.21

Punto de corte: permisivo para rx reales, estricto para fotos naturales/docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


# --- Umbrales ----------------------------------------------------------------

MIN_RESOLUTION        = 150    # px (lado menor) — periapicales web/escaneados pueden ser 150-220px
MIN_DENTAL_LIKELIHOOD = 0.42   # umbral principal — alta calidad baja hasta 0.28 vía tier adaptativo
MIN_QUALITY_SCORE     = 0.21   # calidad mínima — sólo rechaza imágenes verdaderamente corruptas
# Saturación HSV: rx puro -> ~0.0; film digitalizado azul -> 0.10-0.25; foto real -> 0.40+
MAX_COLOR_SATURATION  = 0.38   # era 0.18 — tolera film con tinte de color


@dataclass
class ValidationResult:
    """Resultado de la validación pre-análisis."""
    accepted: bool
    dental_likelihood: float
    quality_score: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    rejection_message: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted":          self.accepted,
            "dental_likelihood": round(self.dental_likelihood, 3),
            "quality_score":     round(self.quality_score, 3),
            "reasons":           self.reasons,
            "metrics":           self.metrics,
            "rejection_message": self.rejection_message,
        }


class DentalImageValidator:
    """
    Validador de imágenes dentales. Stateless — singleton (`dental_validator`).
    """

    REJECTION_GENERIC = (
        "La imagen cargada no corresponde a una radiografía o imagen dental "
        "válida para análisis clínico."
    )

    # --- API pública ----------------------------------------------------------

    def is_intraoral_photo(self, bgr: np.ndarray) -> tuple[bool, float]:
        """Detecta foto intraoral (mucosa + dientes) por dominancia rosa/rojo."""
        if bgr is None or bgr.ndim != 3:
            return False, 0.0
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        pink_mask  = (((H <= 18) | (H >= 160)) & (S > 35) & (V > 60)).astype(np.uint8)
        white_mask = ((V > 180) & (S < 60)).astype(np.uint8)

        pink_ratio  = float(pink_mask.mean())
        white_ratio = float(white_mask.mean())

        is_intraoral = (
            pink_ratio  > 0.18 and
            white_ratio > 0.03 and
            white_ratio < 0.55
        )
        score = float(np.clip(
            (pink_ratio - 0.12) / 0.45 * 0.6 +
            (min(white_ratio, 0.30) / 0.30) * 0.4,
            0.0, 1.0,
        ))
        return is_intraoral, score

    def validate(
        self,
        image_bgr_or_gray: np.ndarray,
        min_resolution: int = MIN_RESOLUTION,
        min_dental_likelihood: float = MIN_DENTAL_LIKELIHOOD,
        min_quality_score: float = MIN_QUALITY_SCORE,
    ) -> ValidationResult:
        """
        Valida una imagen ya cargada.
        Devuelve ValidationResult con .accepted indicando el veredito.

        Flujo de decisión:
          1. Sanity + resolución (rechazos inmediatos indiscutibles)
          2. Calidad técnica (sharpness / contraste / ruido)
          3. Score de dentalidad (7 señales: histograma, bordes, brillo...)
          4. Color saturation como PENALIZACIÓN suave al dental_score
             — solo rechazo duro si la imagen es claramente una foto a color
          5. Bonificaciones (aspecto panorámico, intraoral)
          6. Umbral final adaptativo según calidad técnica
        """
        
        # DEBUG TIP: Descomenta las siguientes dos líneas para guardar la imagen 
        # en disco y comprobar qué es exactamente lo que está recibiendo el validador.
        # import time
        # cv2.imwrite(f"debug_val_{time.time()}.jpg", image_bgr_or_gray)

        reasons: list[str] = []
        metrics: dict = {}

        # -- 0. Sanity de input -----------------------------------------------
        if image_bgr_or_gray is None or not isinstance(image_bgr_or_gray, np.ndarray):
            return ValidationResult(
                accepted=False, dental_likelihood=0.0, quality_score=0.0,
                reasons=["imagen vacía o no válida"],
                rejection_message=self.REJECTION_GENERIC,
            )

        if image_bgr_or_gray.ndim == 2:
            gray  = image_bgr_or_gray
            color = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_GRAY2BGR)
        elif image_bgr_or_gray.ndim == 3 and image_bgr_or_gray.shape[2] in (3, 4):
            color = image_bgr_or_gray[..., :3]
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        else:
            return ValidationResult(
                accepted=False, dental_likelihood=0.0, quality_score=0.0,
                reasons=[f"formato inesperado: shape={image_bgr_or_gray.shape}"],
                rejection_message=self.REJECTION_GENERIC,
            )

        h, w = gray.shape
        metrics["resolution"] = [w, h]

        # -- 1. Resolución mínima ---------------------------------------------
        if min(h, w) < min_resolution:
            reasons.append(
                f"resolución insuficiente ({w}x{h}, mínimo {min_resolution}px)"
            )

        # -- 2. Métricas de calidad -------------------------------------------
        sharpness = self._sharpness(gray)
        contrast  = self._contrast(gray)
        noise     = self._noise(gray)
        metrics["sharpness"]   = round(sharpness, 4)
        metrics["contrast"]    = round(contrast, 4)
        metrics["noise_sigma"] = round(noise, 4)

        if sharpness < 0.004:   # solo imágenes sin ningún detalle (completamente borrosas)
            reasons.append("imagen extremadamente borrosa")
        if contrast < 0.028:    # contraste casi nulo (imagen casi uniforme / en blanco)
            reasons.append("contraste insuficiente para análisis clínico")
        if noise > 0.95:
            reasons.append("nivel de ruido excesivo")

        # -- 3. Score de "dentalidad" -----------------------------------------
        dental_score, dental_metrics = self._dental_score(gray)
        metrics.update(dental_metrics)

        # Penalización fuerte si parece texto/screenshot.
        # Umbral 0.50 en vez de 0.40: evita falsos positivos en rx con fondo muy oscuro
        # (p.ej. periapicales donde casi todos los píxeles caen en hist[:20]).
        # Documentos/screenshots reales suelen tener text_like > 0.65.
        text_like = dental_metrics.get("text_like", 0.0)
        if text_like > 0.50:
            dental_score = min(dental_score, 0.36)
            reasons.append(
                f"patrón compatible con texto/documento/screenshot "
                f"(text_like={text_like:.2f})"
            )

        # -- 4. Color / saturación — penalización SUAVE, no rechazo inmediato -
        sat_score = self._color_saturation(color)
        metrics["color_saturation"] = round(sat_score, 3)

        is_intraoral, intraoral_score = self.is_intraoral_photo(color)
        metrics["intraoral_score"] = round(intraoral_score, 3)
        metrics["is_intraoral"]    = is_intraoral

        # Aplicar penalización de color salvo que la imagen sea CLARAMENTE intraoral
        # (intraoral_score >= 0.35).  Imágenes que apenas superan el umbral de
        # is_intraoral (score típico < 0.25) NO se eximen: podrían ser fotos de color
        # que accidentalmente tienen píxeles rosados (random noise, filtros, etc.)
        if not is_intraoral or intraoral_score < 0.35:
            if sat_score > 0.60:
                # Claramente una foto a color — rechazo duro
                reasons.append(
                    f"imagen con color muy elevado (saturación HSV={sat_score:.2f}) — "
                    "las radiografías dentales son en escala de grises. "
                    "Sube una radiografía (panorámica, periapical o bitewing), "
                    "no una fotografía a color."
                )
            elif sat_score > MAX_COLOR_SATURATION:
                # Tinte de color entre 0.38 y 0.60:
                #   - X-ray de film digitalizado con tinte azul/verde: dental_score >> 0.70
                #     -> penalización moderada, sigue aceptándose
                #   - Foto coloreada genérica: dental_score < 0.60
                #     -> penalización fuerte, cae por debajo del umbral
                #
                # Fórmula: penalización crece con el cuadrado de la saturación excedente
                # (curva convexa: suave al inicio, agresiva cerca de 0.60)
                sat_range = 0.60 - MAX_COLOR_SATURATION          # = 0.22
                sat_excess = sat_score - MAX_COLOR_SATURATION
                penalty_factor = (sat_excess / sat_range) ** 1.6  # curva convexa
                color_penalty = float(np.clip(penalty_factor * 0.48, 0.0, 0.48))
                dental_score = max(0.0, dental_score - color_penalty)
                metrics["color_penalty"] = round(color_penalty, 3)
                logger.debug(
                    "Color saturation moderado (%.2f) — penalizacion: -%.3f",
                    sat_score, color_penalty,
                )

        # -- 5. Bonificaciones -------------------------------------------------
        # Intraoral detectado: alzar dental_score
        if is_intraoral and intraoral_score > 0.50:
            dental_score = max(dental_score, intraoral_score * 0.95)

        # Relación de aspecto panorámica (OPG: ancho = 2-3 x alto)
        aspect = max(w, h) / max(min(w, h), 1)
        metrics["aspect_ratio"] = round(aspect, 2)
        if 1.35 <= aspect <= 3.8:
            panoramic_bonus = float(np.clip((aspect - 1.35) / 2.0, 0.0, 0.12))
            dental_score = min(1.0, dental_score + panoramic_bonus)
            metrics["panoramic_bonus"] = round(panoramic_bonus, 3)

        # Periapical / bitewing bonus: imagen casi cuadrada con firma oscura de rx.
        # Los periapicales muestran 1-4 dientes en un recuadro pequeño, con mucho fondo
        # negro alrededor, por lo que xray_dark_sig suele ser 0.6-1.0.
        # No aplica si ya capturó el bono panorámico (aspect >= 1.35).
        elif aspect < 1.35:
            peri_dark = dental_metrics.get("xray_dark_sig", 0.0)
            # Escala: dark=0.60 -> +0.054; dark=0.80 -> +0.072; dark=1.0 -> +0.09 (max)
            periapical_bonus = float(np.clip(peri_dark * 0.09, 0.0, 0.09))
            dental_score = min(1.0, dental_score + periapical_bonus)
            if periapical_bonus > 0.01:
                metrics["periapical_bonus"] = round(periapical_bonus, 3)

        # -- 6. Calidad combinada ---------------------------------------------
        quality_score = self._combine_quality(sharpness, contrast, noise, h, w)
        metrics["quality_score"]     = round(quality_score, 3)
        metrics["dental_likelihood"] = round(dental_score, 3)

        # -- 7. Decisión final -------------------------------------------------
        # Umbral adaptativo de dentalidad: sólo se activa para imágenes de ALTA calidad
        # (quality_score >= 0.55).  Evitamos tiers intermedios porque la zona gris
        # 0.38–0.42 es donde caen algunas fotos de color con is_intraoral marginal;
        # si aplicáramos la reducción a calidad media, esas fotos podrían coarse.
        #
        # Calidad alta  (>= 0.55): umbral baja de 0.42 a 0.28
        #                 -> rx de buena calidad con señal dental borderline pasan
        # Calidad normal (< 0.55): umbral queda en 0.42
        #                 -> protege frente a fotos de color con dental = 0.41
        effective_min_dental = min_dental_likelihood
        if quality_score >= 0.55:
            effective_min_dental = max(0.28, min_dental_likelihood - 0.15)
        metrics["effective_min_dental"] = round(effective_min_dental, 3)

        if dental_score < effective_min_dental:
            reasons.append(
                f"baja probabilidad de imagen dental ({dental_score*100:.0f}%) — "
                "asegúrate de subir una radiografía dental (panorámica, periapical o bitewing)"
            )
        if quality_score < min_quality_score:
            reasons.append(
                f"calidad técnica insuficiente ({quality_score*100:.0f}%)"
            )

        accepted = len(reasons) == 0
        rejection_msg = self.REJECTION_GENERIC if not accepted else ""

        result = ValidationResult(
            accepted=accepted,
            dental_likelihood=dental_score,
            quality_score=quality_score,
            reasons=reasons,
            metrics=metrics,
            rejection_message=rejection_msg,
        )

        if accepted:
            logger.info(
                "Validacion dental OK | dental=%.2f | calidad=%.2f | sat=%.2f | aspecto=%.2f | %dx%d",
                dental_score, quality_score, sat_score, aspect, w, h,
            )
        else:
            logger.warning(
                "Validacion dental RECHAZADA | dental=%.2f | calidad=%.2f | razones=%s",
                dental_score, quality_score, reasons,
            )
        return result

    # --- Sub-detectores -------------------------------------------------------

    @staticmethod
    def _color_saturation(bgr: np.ndarray) -> float:
        """Saturación promedio HSV [0,1]. rx genuina: ~0.0; foto a color: ~0.50+."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[..., 1].astype(np.float32) / 255.0
        return float(np.mean(s)) if s.size else 0.0

    @staticmethod
    def _sharpness(gray: np.ndarray) -> float:
        """Varianza del Laplaciano normalizada. rx nítida: 0.15-0.50."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.clip(lap.var() / 1500.0, 0.0, 1.0))

    @staticmethod
    def _contrast(gray: np.ndarray) -> float:
        """std normalizada. rx útiles: > 0.18."""
        return float(np.clip(gray.std() / 128.0, 0.0, 1.0))

    @staticmethod
    def _noise(gray: np.ndarray) -> float:
        """Residual mediana: alto = ruidoso."""
        smoothed = cv2.medianBlur(gray, 5)
        residual = gray.astype(np.float32) - smoothed.astype(np.float32)
        return float(np.clip(residual.std() / 40.0, 0.0, 1.0))

    @staticmethod
    def _combine_quality(
        sharpness: float, contrast: float, noise: float,
        h: int, w: int,
    ) -> float:
        """Score técnico [0,1]."""
        res_score = float(np.clip((min(h, w) - 180) / 700.0, 0.0, 1.0))
        return float(np.clip(
            0.30 * sharpness +
            0.30 * contrast  +
            0.25 * (1.0 - noise) +
            0.15 * res_score,
            0.0, 1.0,
        ))

    # --- Detector de "dentalidad" ---------------------------------------------

    def _dental_score(self, gray: np.ndarray) -> tuple[float, dict]:
        """
        Score [0,1] de probabilidad de imagen rx/foto dental — 7 señales.
        """
        metrics: dict = {}

        # a) Bimodalidad del histograma
        bimod_score = self._histogram_bimodality(gray)
        metrics["bimodality"] = round(bimod_score, 3)

        # b) Dispersión de intensidades (X-ray: va de muy oscuro a muy brillante)
        spread_score = self._histogram_spread(gray)
        metrics["hist_spread"] = round(spread_score, 3)

        # c) Densidad de bordes — función TRAPEZOIDAL en lugar de Gaussian estrecha
        edges = cv2.Canny(gray, 50, 130)
        edge_density = float(edges.mean() / 255.0)
        edge_score   = self._edge_density_score(edge_density)
        metrics["edge_density"] = round(edge_density, 4)
        metrics["edge_score"]   = round(edge_score, 3)

        # d) Regiones brillantes compactas (esmalte / coronas / restauraciones)
        bright_score = self._bright_compact_score(gray)
        metrics["bright_compact"] = round(bright_score, 3)

        # e) Ausencia de texto/UI
        text_score = self._text_like_score(gray)
        metrics["text_like"] = round(text_score, 3)
        not_text = 1.0 - text_score

        # f) Firma "rx oscura": bandas negras de aire + regiones blancas de esmalte
        xray_dark = self._xray_dark_signature(gray)
        metrics["xray_dark_sig"] = round(xray_dark, 3)

        # g) Contraste local vs global
        local_global = self._local_global_contrast(gray)
        metrics["local_global"] = round(local_global, 3)

        # Combinación ponderada
        score = (
            0.18 * bimod_score  +
            0.15 * spread_score +
            0.20 * edge_score   +
            0.18 * bright_score +
            0.14 * not_text     +
            0.10 * xray_dark    +
            0.05 * local_global
        )
        return float(np.clip(score, 0.0, 1.0)), metrics

    # --- Señales individuales --------------------------------------------------

    @staticmethod
    def _histogram_bimodality(gray: np.ndarray) -> float:
        """
        Coeficiente de bimodalidad de SAS (BC), corregido.

        Notas sobre el BC original:
          - Distribución normal pura:    BC = 0.33 -> score~0
          - Distribución uniforme:       BC = 0.55 -> FALSO POSITIVO si umbral < 0.55
          - rx panorámica típica:        BC = 0.65-0.90 (fondo oscuro + picos de esmalte)

        Umbral de corte aquí = 0.50 -> la distribución uniforme queda cerca de 0,
        y las rx genuinas (BC > 0.65) obtienen puntaje alto.
        """
        flat = gray.flatten().astype(np.float64)
        n    = flat.size
        if n == 0:
            return 0.0
        mean = flat.mean()
        std  = flat.std()
        if std < 1.0:
            return 0.0
        skew = float(np.mean(((flat - mean) / std) ** 3))
        kurt = float(np.mean(((flat - mean) / std) ** 4) - 3)
        bc   = (skew**2 + 1) / (
            kurt + 3 * ((n - 1)**2) / (max((n - 2) * (n - 3), 1)) + 1e-9
        )
        # Umbral 0.50: uniforme -> score=0.12; rx bimodal -> score=0.5-1.0
        return float(np.clip((bc - 0.50) / 0.45, 0.0, 1.0))

    @staticmethod
    def _histogram_spread(gray: np.ndarray) -> float:
        """
        Dispersión del histograma (p5 – p95).
        Las radiografías van de muy oscuro (aire) a muy brillante (esmalte): spread alto.
        Fotos uniformes o demasiado planas -> spread bajo.
        """
        p5  = float(np.percentile(gray, 5))
        p95 = float(np.percentile(gray, 95))
        spread = (p95 - p5) / 255.0
        # rx real: spread > 0.55 normalmente; foto uniforme: spread < 0.40
        return float(np.clip((spread - 0.25) / 0.45, 0.0, 1.0))

    @staticmethod
    def _edge_density_score(edge_density: float) -> float:
        """
        Función trapezoidal para densidad de bordes.

        Rangos típicos:
          - Imagen en blanco / sólida:    < 0.01  -> score~0
          - rx periapical comprimida:     0.03-0.10
          - rx panorámica:                0.06-0.25
          - rx con muchas restauraciones: hasta 0.30
          - Foto natural muy detallada:   > 0.35

        Acepta un rango amplio (0.03 – 0.35) en lugar del Gaussiano estrecho
        anterior centrado en 0.085, que penalizaba fuertemente rx reales.
        """
        d = edge_density
        if d < 0.01:
            # casi sin bordes -> no es una imagen de diagnóstico
            return d / 0.01 * 0.20
        if d < 0.05:
            # zona de arranque (rx muy blandas o comprimidas)
            return 0.20 + (d - 0.01) / 0.04 * 0.50
        if d < 0.30:
            # rango principal — rx reales caen aquí
            return 1.0
        if d < 0.45:
            # declive suave — alta densidad de bordes (posible foto natural)
            return 1.0 - (d - 0.30) / 0.15 * 0.55
        # muy alta densidad -> probablemente foto natural
        return max(0.0, 0.45 - (d - 0.45) / 0.30 * 0.45)

    @staticmethod
    def _xray_dark_signature(gray: np.ndarray) -> float:
        """
        Firma radiográfica: presencia simultánea de píxeles muy oscuros (aire/fondo)
        y píxeles brillantes (esmalte/restauraciones).

        Panorámicas: bandas negras en la parte superior e inferior (cavidades aéreas).
        Periapicales: fondo oscuro alrededor del diente.
        """
        dark_ratio   = float((gray < 60).mean())    # aire / fondo oscuro
        bright_ratio = float((gray > 185).mean())   # esmalte / restauraciones

        # Señal positiva: muchos píxeles oscuros (rx tiene fondo negro)
        dark_sig   = float(np.clip((dark_ratio  - 0.08) / 0.30, 0.0, 1.0))
        # Señal positiva: algunos píxeles brillantes (coronas, restauraciones)
        bright_sig = float(np.clip(bright_ratio / 0.08, 0.0, 1.0))

        # Combinación: necesita AMBAS señales para máxima puntuación
        return float(np.clip(0.55 * dark_sig + 0.45 * bright_sig, 0.0, 1.0))

    @staticmethod
    def _bright_compact_score(gray: np.ndarray) -> float:
        """Detecta zonas brillantes compactas (coronas/esmalte en rx)."""
        p90 = float(np.percentile(gray, 90))
        if p90 < 50:
            return 0.0
        threshold = max(120, int(p90 * 0.88))
        bright = (gray >= threshold).astype(np.uint8)
        if bright.sum() == 0:
            return 0.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

        num, _, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if num <= 1:
            return 0.0

        score = 0.0
        weight_total = 0.0
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            bw   = stats[i, cv2.CC_STAT_WIDTH]
            bh   = stats[i, cv2.CC_STAT_HEIGHT]
            if area < 15:
                continue
            compactness   = area / max(1, bw * bh)
            score        += compactness * area
            weight_total += area

        if weight_total == 0:
            return 0.0
        avg_compact = score / weight_total

        # Bono si hay 1-20 blobs compactos (dentadura completa, parcial o periapical)
        # mu=5 / sigma=10 -> periapicales con 1-4 blobs siguen puntuando alto (>= 0.88)
        #                     panorámicas con 8-14 blobs también superan 0.85
        n_sig   = sum(1 for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= 40)
        ds_bonus = DentalImageValidator._gauss_like(n_sig, mu=5, sigma=10)
        return float(np.clip(0.55 * avg_compact + 0.45 * ds_bonus, 0.0, 1.0))

    @staticmethod
    def _text_like_score(gray: np.ndarray) -> float:
        """
        Heurística para detectar texto/screenshots/documentos.
        Devuelve [0,1]: alto = parece texto/screenshot.
        """
        h, w = gray.shape
        hist      = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist      = hist / max(hist.sum(), 1)
        # Masa en extremos (blanco puro / negro puro) — documentos: > 0.65
        extreme_mass = float(hist[:20].sum() + hist[-20:].sum())

        extreme_score = 0.0
        if extreme_mass > 0.65:
            extreme_score = min(1.0, (extreme_mass - 0.65) / 0.25)

        # Bandas horizontales repetitivas (texto)
        bin_img    = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        row_sums   = bin_img.sum(axis=1) / 255.0
        active_rows = (row_sums > w * 0.25).astype(np.int32)
        transitions = int(np.sum(np.abs(np.diff(active_rows))))
        line_score  = min(1.0, transitions / 35.0) if transitions >= 14 else 0.0

        return float(np.clip(0.60 * extreme_score + 0.40 * line_score, 0.0, 1.0))

    @staticmethod
    def _local_global_contrast(gray: np.ndarray) -> float:
        """
        Medida de coherencia espacial (suavidad intra-región).

        Las radiografías tienen GRANDES regiones de intensidad homogénea
        (fondo negro = aire, gris medio = hueso/tejido) -> la variación local
        (dentro de una ventana 15x15) es PEQUEÑA comparada con la variación
        global (entre regiones). Ratio local/global BAJO -> buena señal de rx.

        Imágenes ruidosas / fotos naturales: variación local = global -> ratio = 1.

        Devuelve: 1 - ratio (alto = imagen suave y estructurada como una rx).
        """
        gf    = gray.astype(np.float32)
        local = cv2.boxFilter(gf, -1, (15, 15))
        diff2 = (gf - local) ** 2
        local_std  = float(cv2.boxFilter(diff2, -1, (15, 15)).mean() ** 0.5)
        global_std = float(gray.std())
        if global_std < 1e-3:
            return 0.0
        ratio = local_std / global_std
        # ratio = 0.30-0.60 para rx (suaves); = 0.80-1.0 para ruido/fotos
        # Invertimos: (1 - ratio) normalizado para que rx -> score alto
        return float(np.clip((0.80 - ratio) / 0.55, 0.0, 1.0))

    @staticmethod
    def _gauss_like(x: float, mu: float, sigma: float) -> float:
        """Gaussiana sin normalizar, pico en mu."""
        return float(np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2)))


# Instancia global
dental_validator = DentalImageValidator()