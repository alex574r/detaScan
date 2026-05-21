"""
DentaScan — Calibración estadística de confianza clínica.

Convierte la probabilidad cruda del modelo y la evidencia visual en una
"confianza clínica" calibrada que refleja precisión estadística, no solo
output del clasificador.

Componentes:
  - Temperature scaling   → suaviza picos artificiales (modelos sobre-confiados)
  - Evidence weighting    → multiplica por la evidencia geométrica/textural
  - Ensemble agreement    → bonus por acuerdo entre clasificadores
  - Clinical prior        → corrige sobre-detección (prior de prevalencia)

El umbral por defecto es 0.90 — solo se reportan hallazgos con
calibrated_confidence ≥ 0.90.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


CLINICAL_CONFIDENCE_THRESHOLD = 0.90  # umbral mínimo para mostrar hallazgo


@dataclass
class EvidenceContext:
    """
    Evidencia visual asociada a un hallazgo individual.

    Todos los campos son [0,1] excepto donde se indica.
    """
    area_norm:        float   # área normalizada por tamaño imagen
    solidity:         float   # qué tan llena vs convex hull
    circularity:      float   # 4πA/P²
    contrast_local:   float   # contraste local vs vecindario
    edge_strength:    float   # magnitud de borde en el perímetro
    is_radiopaque:    bool
    is_recurrent:     bool
    inside_tooth:     bool    # ¿está dentro de la máscara dental?
    sensitivity_match: float = 1.0  # 1 si encaja con la sensibilidad pedida


def temperature_scale(prob: np.ndarray, T: float = 1.4) -> np.ndarray:
    """
    Re-suaviza una distribución de probabilidad (típicamente sobre-confiada
    en modelos sintéticos). T>1 reduce picos; T<1 los acentúa.
    """
    p = np.clip(prob, 1e-9, 1.0)
    logits = np.log(p)
    scaled = logits / T
    e = np.exp(scaled - scaled.max())
    return e / e.sum()


def evidence_score(ctx: EvidenceContext) -> float:
    """
    Score visual ∈ [0,1] basado en evidencia geométrica/textural.

    Reglas:
      - Fuera del campo dental                       → 0.10 (casi ningún peso)
      - Bordes muy nítidos + textura compacta         → bonus
      - Forma desestructurada (solidity bajo)         → penalización
      - Caries recurrente (adyacente a restauración)  → bonus
    """
    if not ctx.inside_tooth:
        return 0.10

    # Componentes ponderados — calibrados para que una lesión "típica clara"
    # (área notable, contornos definidos, alto contraste) dé ~0.85-0.95.
    geom = (
        0.25 * np.clip(ctx.area_norm * 3.0, 0, 1) +
        0.25 * ctx.solidity +
        0.15 * ctx.circularity +
        0.25 * ctx.contrast_local +
        0.10 * ctx.edge_strength
    )

    # Bonus / penalización
    if ctx.is_recurrent:
        geom *= 1.10
    if ctx.solidity < 0.40:
        geom *= 0.70  # forma irregular → puede ser sombra

    geom *= ctx.sensitivity_match
    return float(np.clip(geom, 0.0, 1.0))


def calibrate_finding(
    model_proba_top: float,
    ensemble_agreement: float,
    evidence: EvidenceContext,
    clinical_prior: float = 0.65,
    temperature: float = 1.3,
) -> float:
    """
    Devuelve una confianza calibrada ∈ [0,1] para un hallazgo individual.

    Política clínica:
      La confianza calibrada refleja primariamente la EVIDENCIA VISUAL del
      hallazgo (lo que objetivamente vemos en píxeles), con el modelo ML y
      el acuerdo entre clasificadores como soporte estadístico.

    Dos regímenes:

      A. Evidencia contundente → confianza ALTA (≥ 0.85 garantizado)
         Cuando un hallazgo cumple TODAS las condiciones siguientes:
           - está dentro del campo dental
           - solidity ≥ 0.70 (forma compacta, no fragmentada)
           - contrast_local ≥ 0.55 (claramente más oscuro que el vecindario)
           - circularity ≥ 0.40 o is_recurrent (forma de lesión real)
         Entonces:
           c = 0.85 + 0.08·agreement + 0.07·p_model + bonus recurrencia
         Esto refleja que cuando la evidencia visual es inequívoca, el
         clínico debe ver el hallazgo aunque el modelo ML global esté inseguro.

      B. Evidencia parcial → confianza MODERADA
         Combinación lineal:
           c = 0.40·ev + 0.30·p_model + 0.20·agreement + 0.10·prior
    """
    p_model_t = float(temperature_scale(
        np.array([model_proba_top, 1.0 - model_proba_top]),
        T=temperature,
    )[0])

    ev = evidence_score(evidence)

    # ── Caso A: evidencia contundente ────────────────────────────────────────
    strong_evidence = (
        evidence.inside_tooth and
        evidence.solidity      >= 0.70 and
        evidence.contrast_local >= 0.55 and
        (evidence.circularity   >= 0.40 or evidence.is_recurrent)
    )

    if strong_evidence:
        c = 0.85 + 0.08 * float(ensemble_agreement) + 0.07 * p_model_t
        if evidence.is_recurrent:
            c *= 1.05
        return float(np.clip(c, 0.0, 1.0))

    # ── Caso B: evidencia parcial ────────────────────────────────────────────
    base = (
        0.40 * ev +
        0.30 * p_model_t +
        0.20 * float(ensemble_agreement) +
        0.10 * float(clinical_prior)
    )

    if not evidence.inside_tooth:
        base *= 0.30
    elif ev < 0.30:
        base *= 0.60

    return float(np.clip(base, 0.0, 1.0))


def ensemble_agreement_score(
    probas: list[np.ndarray],
) -> float:
    """
    Mide cuánto coinciden varios clasificadores. Usa 1 - varianza promedio
    entre las distribuciones de probabilidad.

    Si solo se pasa una distribución, devuelve la altura del pico (1 - entropía
    normalizada).
    """
    if not probas:
        return 0.0

    if len(probas) == 1:
        p = np.asarray(probas[0])
        p = np.clip(p, 1e-9, 1.0)
        H = -np.sum(p * np.log2(p))
        H_max = np.log2(len(p))
        return float(np.clip(1.0 - H / H_max, 0.0, 1.0))

    stacked = np.stack([np.asarray(p) for p in probas])
    avg_var = float(np.mean(np.var(stacked, axis=0)))
    # Cada clase tiene var máx 0.25 (cuando es 0 y 1) → normalizamos
    return float(np.clip(1.0 - avg_var / 0.25, 0.0, 1.0))


def filter_by_confidence(
    findings: list[dict],
    threshold: float = CLINICAL_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Filtra una lista de hallazgos por `calibrated_confidence` >= threshold.
    """
    return [f for f in findings if float(f.get("calibrated_confidence", 0.0)) >= threshold]


def build_evidence_from_region(
    region: dict,
    image_shape: tuple[int, int],
    tooth_mask: Optional[np.ndarray] = None,
    sensitivity_match: float = 1.0,
) -> EvidenceContext:
    """
    Construye un EvidenceContext desde una región segmentada.
    """
    h, w = image_shape
    area = float(region.get("area_px", 0))
    area_norm = float(np.clip((area ** 0.5) / 70.0, 0, 1))

    inside = True
    if tooth_mask is not None:
        cx = int(region.get("centroid_x", 0))
        cy = int(region.get("centroid_y", 0))
        if 0 <= cy < h and 0 <= cx < w:
            inside = bool(tooth_mask[cy, cx] > 0)

    contrast_local = 1.0 - min(1.0, float(region.get("mean_intensity", 128)) / 255.0)

    return EvidenceContext(
        area_norm=area_norm,
        solidity=float(region.get("solidity", 0.6)),
        circularity=float(region.get("circularity", 0.4)),
        contrast_local=contrast_local,
        edge_strength=0.6,  # estimación constante razonable
        is_radiopaque=bool(region.get("is_radiopaque", False)),
        is_recurrent=bool(region.get("is_recurrent", False)),
        inside_tooth=inside,
        sensitivity_match=sensitivity_match,
    )
