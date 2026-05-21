"""
DentaScan — Servicio de Procesamiento de Imágenes.

Pipeline clínico mejorado:
  1. Carga
  2. Resize estándar
  3. Preprocesamiento clínico avanzado (normalización + bilateral + CLAHE multi)
  4. Bordes Canny / Sobel (visualización)
  5. Segmentación dental (tooth_mask)
  6. Detección de restauraciones (excluir de caries)
  7. Detección radiolúcida (caries, granulomas, periodontitis...) con contexto
  8. Detección radiopaca (osteítis, hipercementosis)
  9. (Opcional) YOLOv8
 10. Extracción de features (12 base) + features extendidas (textura)
 11. Clasificación ML (RF / SVM / CNN / Ensemble / ResNet50) con métricas CV
 12. Calibración estadística de confianza por hallazgo (>= 90% por defecto)
 13. Visualización: anotada + heatmap gradient-based + histograma
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.core.loader import image_loader
from app.core.preprocessor import image_preprocessor
from app.core.segmentor import image_segmentor
from app.core.feature_extractor import feature_extractor
from app.core.classifier import dental_classifier
from app.core.visualizer import image_visualizer
from app.core.dental_yolo import detect_regions as yolo_detect, yolo_available
from app.core.calibration import (
    CLINICAL_CONFIDENCE_THRESHOLD,
    EvidenceContext,
    build_evidence_from_region,
    calibrate_finding,
    ensemble_agreement_score,
    filter_by_confidence,
)
from app.exceptions.custom import ProcessingError
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class ImageProcessingService:
    """Coordina el pipeline completo de análisis de una radiografía dental."""

    # Sensibilidad → factor que modifica el umbral clínico mínimo
    # (todos respetan el piso del 90% para mostrar al usuario)
    SENSITIVITY_FACTORS = {
        "screening": 0.90,   # 90% mín
        "normal":    0.90,
        "high":      0.88,   # baja levemente para detectar más casos sutiles
        "research":  0.80,   # solo para investigación, baja el umbral
    }

    def process(
        self,
        file_path: str | Path,
        analysis_id: int,
        xray_type: str = "unknown",
        model: str = "ensemble",
        config: dict | None = None,
    ) -> dict:
        start_time = time.perf_counter()
        path = Path(file_path)
        output_prefix = settings.OUTPUT_DIR / f"analysis_{analysis_id}"

        cfg = config or {}
        analysis_mode  = cfg.get("analysis_mode", "all")
        sensitivity    = cfg.get("sensitivity", "normal")
        user_min_conf  = float(cfg.get("min_confidence", 0.0))
        lesion_filter  = set(cfg.get("lesion_types", []) or [])

        # Umbral clínico final: max(piso clínico, lo que pida el usuario)
        clinical_floor = self.SENSITIVITY_FACTORS.get(sensitivity, CLINICAL_CONFIDENCE_THRESHOLD)
        clinical_threshold = max(clinical_floor, user_min_conf if user_min_conf > 0 else 0)

        logger.info(
            "Pipeline | id=%d | %s | modo=%s | sens=%s | umbral_clinico=%.2f | filtro=%s",
            analysis_id, path.name, analysis_mode, sensitivity,
            clinical_threshold, lesion_filter or "—",
        )

        # ── 1. Carga ─────────────────────────────────────────────────────────
        original, metadata = image_loader.load(path)

        # ── 2. Resize ────────────────────────────────────────────────────────
        resized = image_preprocessor.resize_standard(original, xray_type)

        # ── 3. Preprocesamiento clínico avanzado ─────────────────────────────
        preprocessed = image_preprocessor.preprocess(resized)

        # ── 4. Bordes (visualización) ────────────────────────────────────────
        edges_canny = image_segmentor.apply_canny(preprocessed)
        edges_sobel = image_segmentor.apply_sobel(preprocessed)

        # ── 5. Segmentación dental (tooth_mask) ──────────────────────────────
        tooth_mask = image_segmentor.segment_tooth_field(preprocessed)

        # ── 6. Restauraciones (referencia, no patología) ─────────────────────
        rest_mask, rest_regions = image_segmentor.detect_restorations(
            preprocessed, tooth_mask=tooth_mask,
        )

        # ── 7. Radiolucidez (caries y lesiones óseas) ────────────────────────
        dark_mask, radiolucent_regions = image_segmentor.detect_radiolucent_regions(
            preprocessed,
            tooth_mask=tooth_mask,
            restoration_mask=rest_mask,
            xray_type=xray_type,
        )

        # ── 8. Radiopacas óseas ──────────────────────────────────────────────
        dense_mask, dense_regions = image_segmentor.detect_dense_regions(
            preprocessed,
            tooth_mask=tooth_mask,
            restoration_mask=rest_mask,
        )

        # ── 9. YOLO opcional ─────────────────────────────────────────────────
        yolo_regions = []
        if cfg.get("use_yolo", False) and yolo_available():
            yolo_regions = yolo_detect(preprocessed, conf_threshold=0.25)
            logger.info("YOLOv8 detectó %d regiones", len(yolo_regions))

        # Hallazgos candidatos: radiolúcidas + radiopacas + YOLO
        # (NO incluimos restauraciones — son referencia)
        candidates = radiolucent_regions + dense_regions + yolo_regions

        # ── 10. Features base + extendidas ───────────────────────────────────
        features = feature_extractor.extract(preprocessed)
        try:
            extended = feature_extractor.extract_extended(preprocessed)
        except Exception as exc:
            logger.warning("Features extendidas fallaron: %s", exc)
            extended = {}

        # ── 11. Clasificación ML ─────────────────────────────────────────────
        classification = dental_classifier.classify(features, model=model, image=preprocessed)
        logger.info(
            "Clasificación: %s (%.1f%%) | modelo=%s",
            classification["predicted_label"],
            classification["confidence_score"] * 100,
            classification["model_used"],
        )

        # Probabilidades base para el agreement
        class_probs_dict = classification.get("class_probabilities", {})
        base_proba = np.array(list(class_probs_dict.values()), dtype=np.float32)

        if "ensemble_probas" in classification:
            ensemble_probs = [np.asarray(p) for p in classification["ensemble_probas"]]
            agreement = ensemble_agreement_score(ensemble_probs)
        else:
            agreement = ensemble_agreement_score([base_proba])

        # ── 12. Calibración estadística por hallazgo ─────────────────────────
        calibrated_findings = self._calibrate_regions(
            candidates,
            image_shape=preprocessed.shape,
            tooth_mask=tooth_mask,
            model_proba_top=classification["confidence_score"],
            ensemble_agreement=agreement,
            analysis_mode=analysis_mode,
            lesion_filter=lesion_filter,
        )

        # Filtrar al umbral clínico
        accepted = filter_by_confidence(calibrated_findings, threshold=clinical_threshold)
        logger.info(
            "Hallazgos: %d candidatos -> %d con calibrated_confidence >= %.2f",
            len(calibrated_findings), len(accepted), clinical_threshold,
        )

        # ── 13. Visualización ────────────────────────────────────────────────
        lesion_findings = self._build_lesion_findings(accepted)
        output_paths = self._save_outputs(
            analysis_id=analysis_id,
            original=resized,
            preprocessed=preprocessed,
            edges_canny=edges_canny,
            edges_sobel=edges_sobel,
            dark_mask=dark_mask,
            dense_mask=dense_mask,
            accepted_regions=accepted,
            restoration_regions=rest_regions,
            features=features,
            classification=classification,
            output_prefix=output_prefix,
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Pipeline ok | id=%d | %.1f ms", analysis_id, elapsed_ms)

        # Métricas clínicas del modelo + estadística general
        cv_metrics = dental_classifier.get_cv_metrics()

        return {
            "features": features,
            "extended_features": extended,
            "classification": classification,
            "output_paths": output_paths,
            "radiolucent_regions": radiolucent_regions,
            "restoration_regions": rest_regions,
            "lesion_findings": lesion_findings,
            "clinical_threshold": float(clinical_threshold),
            "ensemble_agreement": float(agreement),
            "model_cv_metrics": cv_metrics,
            "metadata": metadata,
            "processing_time_ms": elapsed_ms,
        }

    # ─── Calibración ──────────────────────────────────────────────────────────

    def _calibrate_regions(
        self,
        regions: list[dict],
        image_shape: tuple[int, int],
        tooth_mask: np.ndarray | None,
        model_proba_top: float,
        ensemble_agreement: float,
        analysis_mode: str = "all",
        lesion_filter: set[str] | None = None,
    ) -> list[dict]:
        """
        Para cada región candidata calcula `calibrated_confidence` y agrega
        un campo `clinical_reasoning` legible.
        """
        out: list[dict] = []
        for r in regions:
            # Sensitivity match: si modo targeted, regiones fuera del filtro
            # tienen sensitivity_match=0 (no aparecerán)
            if analysis_mode == "targeted" and lesion_filter:
                if r.get("lesion_type", "") not in lesion_filter:
                    continue

            evidence = build_evidence_from_region(
                r, image_shape, tooth_mask=tooth_mask, sensitivity_match=1.0,
            )

            calibrated = calibrate_finding(
                model_proba_top=model_proba_top,
                ensemble_agreement=ensemble_agreement,
                evidence=evidence,
                clinical_prior=0.65,
                temperature=1.3,
            )

            reasoning = self._build_reasoning(r, evidence, calibrated)

            r_out = dict(r)
            r_out["calibrated_confidence"] = round(calibrated, 4)
            r_out["confidence"]           = round(calibrated, 4)  # compat
            r_out["clinical_reasoning"]   = reasoning
            out.append(r_out)
        return out

    @staticmethod
    def _build_reasoning(region: dict, evidence: EvidenceContext, calibrated: float) -> str:
        """Genera texto en español explicando la justificación clínica."""
        lt = region.get("lesion_type", "lesión")
        bits = []
        bits.append(f"Tipo: {lt}")
        if not evidence.inside_tooth:
            bits.append("fuera del tejido dental (baja prioridad)")
        if evidence.is_recurrent:
            bits.append("adyacente a restauración (sospecha de caries recurrente)")
        if evidence.solidity > 0.7:
            bits.append("contorno bien definido")
        elif evidence.solidity < 0.45:
            bits.append("contorno irregular (posible sombra/artefacto)")
        if evidence.circularity > 0.55:
            bits.append("forma compacta")
        if evidence.area_norm > 0.5:
            bits.append("extensión significativa")
        bits.append(f"confianza calibrada {calibrated*100:.1f}%")
        return " · ".join(bits)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_lesion_findings(regions: list[dict]) -> list[dict]:
        findings = []
        for r in regions:
            findings.append({
                "lesion_type":            r.get("lesion_type", ""),
                "severity":               r.get("severity", ""),
                "x":                      r.get("x", 0),
                "y":                      r.get("y", 0),
                "width":                  r.get("width", 0),
                "height":                 r.get("height", 0),
                "area_px":                r.get("area_px", 0.0),
                "centroid_x":             r.get("centroid_x", 0),
                "centroid_y":             r.get("centroid_y", 0),
                "circularity":            r.get("circularity", 0.0),
                "solidity":               r.get("solidity", 0.0),
                "mean_intensity":         r.get("mean_intensity", 0.0),
                "is_radiopaque":          r.get("is_radiopaque", False),
                "is_recurrent":           r.get("is_recurrent", False),
                "confidence":             r.get("confidence", 0.0),
                "calibrated_confidence":  r.get("calibrated_confidence", 0.0),
                "clinical_reasoning":     r.get("clinical_reasoning", ""),
            })
        return findings

    def _save_outputs(
        self,
        analysis_id: int,
        original: np.ndarray,
        preprocessed: np.ndarray,
        edges_canny: np.ndarray,
        edges_sobel: np.ndarray,
        dark_mask: np.ndarray,
        dense_mask: np.ndarray,
        accepted_regions: list[dict],
        restoration_regions: list[dict],
        features: dict,
        classification: dict,
        output_prefix: Path,
    ) -> dict[str, str]:
        paths: dict[str, str] = {}
        prefix = str(output_prefix)

        # Preprocesada
        prep_path = f"{prefix}_preprocessed.png"
        image_visualizer.save_image(preprocessed, prep_path)
        paths["preprocessed"] = prep_path

        # Bordes
        edge_overlay = image_visualizer.generate_edge_overlay(preprocessed, edges_canny, edges_sobel)
        image_visualizer.save_image(edge_overlay, f"{prefix}_edges.png")
        paths["edges"] = f"{prefix}_edges.png"

        # Máscara combinada
        combined_mask = np.clip(
            dark_mask.astype(np.uint16) + dense_mask.astype(np.uint16), 0, 255,
        ).astype(np.uint8)
        image_visualizer.save_image(combined_mask, f"{prefix}_mask.png")
        paths["mask"] = f"{prefix}_mask.png"

        # Anotada — solo regiones que superan el umbral clínico.
        # Las restauraciones se dibujan con halo gris para referencia visual.
        regions_for_annot = accepted_regions + restoration_regions
        annotated = image_visualizer.generate_attention_mask(
            original=original,
            regions=regions_for_annot,
            predicted_class=classification.get("predicted_class"),
            predicted_label=classification.get("predicted_label"),
            confidence=classification.get("confidence_score"),
        )
        image_visualizer.save_image(annotated, f"{prefix}_annotated.png")
        paths["annotated"] = f"{prefix}_annotated.png"

        # Heatmap gradient-based (saliency)
        heatmap = image_visualizer.generate_gradient_heatmap(
            preprocessed=preprocessed,
            regions=accepted_regions,
        )
        image_visualizer.save_image(heatmap, f"{prefix}_heatmap.png")
        paths["heatmap"] = f"{prefix}_heatmap.png"

        # Histograma
        hist_bytes = image_visualizer.generate_histogram(
            original=original,
            preprocessed=preprocessed,
            features=features,
            predicted_label=classification.get("predicted_label"),
        )
        image_visualizer.save_bytes_as_png(hist_bytes, f"{prefix}_histogram.png")
        paths["histogram"] = f"{prefix}_histogram.png"

        return paths


image_processing_service = ImageProcessingService()
