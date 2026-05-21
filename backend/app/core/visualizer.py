"""
DentaScan — Módulo de Visualización.
RF-14: Generación de imagen anotada con contornos coloreados por tipo de lesión.
RF-15: Generación de histogramas con Matplotlib.
Nuevo: Heatmap térmico de zonas de interés.

Superpone hallazgos sobre la imagen original sin emitir diagnóstico automático final.
El sistema es una herramienta de APOYO — el odontólogo toma la decisión clínica.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.exceptions.custom import ProcessingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Paleta de colores por tipo de lesión (BGR para OpenCV) ──────────────────

LESION_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    # Lesiones radiolúcidas (oscuras)
    "Caries Incipiente":         (0,  200, 240),   # ámbar
    "Caries Avanzada":           (0,  120, 255),   # naranja
    "Caries Oclusal":            (0,  140, 255),   # naranja claro
    "Caries Oclusal Incipiente": (0,  200, 240),   # ámbar
    "Caries Oclusal Avanzada":   (0,   90, 255),   # naranja oscuro
    "Caries Interproximal":      (40, 180, 255),   # ámbar-cobre
    "Caries Recurrente":         (40,  40, 255),   # rojo intenso
    "Granuloma Periapical":      (80,  60, 255),   # rojo-naranja
    "Absceso Periapical":        (0,   0,  255),   # rojo
    "Quiste Periapical":         (180,  0, 200),   # magenta
    "Lesión Periapical Difusa":  (0,   0,  180),   # rojo oscuro
    "Periodontitis Leve":        (0,  200, 150),   # verde-cian
    "Periodontitis Severa":      (0,   80, 255),   # naranja intenso
    "Resorción Ósea":            (200,  0, 200),   # púrpura
    # Lesiones radiopacas
    "Osteítis Condensante":      (255, 220,   0),  # cian brillante
    "Hipercementosis":           (0,  240, 220),   # amarillo
    "Restauración":              (180, 180, 180),  # gris (referencia, no patología)
    # Fallback
    "Lesión Ósea":               (200,  0,  200),
    "Diente Sano":               (0,  200,   0),
}

LESION_COLORS_HEX: dict[str, str] = {
    "Caries Incipiente":         "#f0c800",
    "Caries Avanzada":           "#ff7800",
    "Caries Oclusal":            "#ff8c00",
    "Caries Oclusal Incipiente": "#f0c800",
    "Caries Oclusal Avanzada":   "#ff5a00",
    "Caries Interproximal":      "#ffb428",
    "Caries Recurrente":         "#ff2828",
    "Granuloma Periapical":      "#ff503c",
    "Absceso Periapical":        "#ff0000",
    "Quiste Periapical":         "#b400c8",
    "Lesión Periapical Difusa":  "#c80000",
    "Periodontitis Leve":        "#00c896",
    "Periodontitis Severa":      "#ff5000",
    "Resorción Ósea":            "#c800c8",
    "Osteítis Condensante":      "#ffdc00",
    "Hipercementosis":           "#00f0dc",
    "Restauración":              "#b4b4b4",
    "Lesión Ósea":               "#c800c8",
    "Diente Sano":               "#00c800",
}

# Colores de fallback por clase ML (mantenidos por compatibilidad)
CLASS_COLORS_BGR = {
    0: (0,   200,   0),
    1: (0,   200, 255),
    2: (0,   100, 255),
    3: (0,     0, 255),
    4: (200,   0, 200),
}


class ImageVisualizer:
    """
    Genera imágenes de resultados y gráficos estadísticos.
    Todas las salidas se guardan en output/ con analysis_id como prefijo.
    """

    def generate_attention_mask(
        self,
        original: np.ndarray,
        regions: list[dict],
        predicted_class: int | None = None,
        predicted_label: str | None = None,
        confidence: float | None = None,
    ) -> np.ndarray:
        """
        RF-14: Imagen anotada con contornos gruesos coloreados por tipo de lesión.

        Por cada región detectada:
          - Relleno semitransparente ~30% opacidad
          - Contorno grueso del polígono real (3px) con halo negro para contraste
          - Cruz de centroide con halo
          - Etiqueta con tipo y severidad sobre fondo sólido
        Si no hay hallazgos, se indica explícitamente.
        """
        if original.ndim == 2:
            annotated = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            annotated = original.copy()

        if not regions:
            self._draw_header(annotated, predicted_label, confidence, predicted_class)
            # Indicar explícitamente que no se detectaron regiones anómalas
            msg = "Sin regiones anomalas detectadas"
            h_img, w_img = annotated.shape[:2]
            cv2.putText(annotated, msg,
                        (w_img // 2 - 160, h_img // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
            self._draw_footer(annotated)
            return annotated

        # ── Capa 1: rellenos semitransparentes ──────────────────────────────
        overlay = annotated.copy()
        for region in regions:
            color = self._region_color(region, predicted_class)
            pts = region.get("contour_points")
            if pts and len(pts) >= 3:
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
            else:
                x, y, w, h = region["x"], region["y"], region["width"], region["height"]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, cv2.FILLED)
        annotated = cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0)

        # ── Capa 2: contornos con halo, centroides y etiquetas ───────────────
        for idx, region in enumerate(regions):
            color = self._region_color(region, predicted_class)
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            pts = region.get("contour_points")

            # Halo negro (grosor + 2) para contraste sobre cualquier fondo
            if pts and len(pts) >= 3:
                arr = np.array(pts, dtype=np.int32)
                cv2.polylines(annotated, [arr], True, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.polylines(annotated, [arr], True, color,    3, cv2.LINE_AA)
            else:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 0, 0), 4)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color,    2)

            # Bounding box punteado (referencia)
            self._draw_dashed_rect(annotated, x, y, x+w, y+h, color, thickness=1)

            # Centroide: círculo con halo
            cx = region.get("centroid_x", x + w // 2)
            cy = region.get("centroid_y", y + h // 2)
            cv2.circle(annotated, (cx, cy), 7,  (0, 0, 0), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(annotated, (cx, cy), 5,  color,     cv2.FILLED, cv2.LINE_AA)
            # Cruz en el centroide
            cv2.line(annotated, (cx-10, cy), (cx+10, cy), (0,0,0), 3, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy-10), (cx, cy+10), (0,0,0), 3, cv2.LINE_AA)
            cv2.line(annotated, (cx-10, cy), (cx+10, cy), color, 1, cv2.LINE_AA)
            cv2.line(annotated, (cx, cy-10), (cx, cy+10), color, 1, cv2.LINE_AA)

            # Número de hallazgo junto al centroide
            num = str(idx + 1)
            cv2.putText(annotated, num, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(annotated, num, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color,   1, cv2.LINE_AA)

            # Etiqueta — incluye tipo, severidad y confianza calibrada
            lesion_type = region.get("lesion_type", "")
            severity    = region.get("severity", "")
            calibrated  = region.get("calibrated_confidence")
            conf_str    = f" {calibrated*100:.0f}%" if calibrated is not None else ""

            label_text = lesion_type if lesion_type else (predicted_label or "Region")
            if severity and severity != "n/a":
                label_text = f"{idx+1}. {label_text} [{severity}]{conf_str}"
            else:
                label_text = f"{idx+1}. {label_text}{conf_str}"
            self._draw_label(annotated, x, y, label_text, color)

        # ── Cabecera y pie ───────────────────────────────────────────────────
        self._draw_header(annotated, predicted_label, confidence, predicted_class)
        self._draw_footer(annotated)
        return annotated

    def generate_gradient_heatmap(
        self,
        preprocessed: np.ndarray,
        regions: list[dict] | None = None,
    ) -> np.ndarray:
        """
        Heatmap "saliency" basado en gradiente — aproxima Grad-CAM para
        modelos clásicos. La intensidad refleja:

          - magnitud del gradiente local (Sobel) — bordes de lesión
          - oscuridad del píxel — radiolucidez
          - cercanía a regiones detectadas (bump gaussiano por hallazgo)

        Devuelve imagen BGR con overlay coloreado (JET).
        """
        if preprocessed.ndim == 2:
            base = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        else:
            base = preprocessed.copy()
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)

        h, w = preprocessed.shape

        # Componente 1: gradiente normalizado
        gx = cv2.Sobel(preprocessed, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(preprocessed, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        grad = grad / (grad.max() + 1e-9)

        # Componente 2: oscuridad
        darkness = 1.0 - preprocessed.astype(np.float32) / 255.0

        # Componente 3: bumps en regiones detectadas
        bumps = np.zeros_like(preprocessed, dtype=np.float32)
        if regions:
            for r in regions:
                cx = int(r.get("centroid_x", 0))
                cy = int(r.get("centroid_y", 0))
                if not (0 <= cy < h and 0 <= cx < w):
                    continue
                sigma = max(8.0, float(r.get("area_px", 200)) ** 0.5 * 0.5)
                yy, xx = np.ogrid[:h, :w]
                bump = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
                weight = float(r.get("calibrated_confidence", r.get("confidence", 0.5)))
                bumps = np.maximum(bumps, bump.astype(np.float32) * weight)

        saliency = 0.30 * grad + 0.30 * darkness + 0.40 * bumps
        saliency = saliency / (saliency.max() + 1e-9)
        saliency_u8 = (saliency * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(saliency_u8, cv2.COLORMAP_JET)
        result = cv2.addWeighted(base, 0.55, heatmap, 0.45, 0)
        self._draw_heatmap_legend(result)
        return result

    def generate_heatmap_overlay(
        self,
        original: np.ndarray,
        dark_mask: np.ndarray,
        dense_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Genera un mapa de calor térmico superponiendo las zonas de interés.

        Las regiones radiolúcidas se muestran en escala caliente (azul→rojo).
        Las regiones radiopacas se superponen en cian si se proporcionan.
        Útil para visualizar la distribución espacial de las anomalías.
        """
        if original.ndim == 2:
            base = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            base = original.copy()

        # Construir mapa de anomalías (float32, 0–1)
        anomaly = dark_mask.astype(np.float32) / 255.0

        if dense_mask is not None:
            dense_norm = dense_mask.astype(np.float32) / 255.0
            anomaly = np.clip(anomaly + dense_norm * 0.65, 0.0, 1.0)

        # Difuminar para transiciones suaves
        blurred = cv2.GaussianBlur(anomaly, (45, 45), 0)

        if blurred.max() > 1e-6:
            blurred = (blurred / blurred.max() * 255).astype(np.uint8)
        else:
            # No hay anomalías: devolver imagen base con velo
            velo = np.zeros_like(base)
            return cv2.addWeighted(base, 0.9, velo, 0.1, 0)

        # Aplicar mapa de color JET (azul=frío, rojo=caliente)
        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

        # Mezclar con imagen original
        result = cv2.addWeighted(base, 0.50, heatmap, 0.50, 0)

        # Leyenda
        self._draw_heatmap_legend(result)
        return result

    def generate_edge_overlay(
        self,
        original: np.ndarray,
        edges_canny: np.ndarray,
        edges_sobel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Superpone bordes Canny (verde) y Sobel (azul) sobre la imagen original."""
        if original.ndim == 2:
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            overlay = original.copy()

        overlay[edges_canny > 0] = [0, 200, 0]

        if edges_sobel is not None:
            sobel_norm = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            overlay[sobel_norm > 128] = [200, 0, 0]

        return overlay

    def generate_histogram(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        features: dict[str, float] | None = None,
        predicted_label: str | None = None,
    ) -> bytes:
        """
        RF-15: Histograma de intensidades antes y después del preprocesamiento.
        Incluye línea de umbral de radiolucidez y panel de features.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#1a1a2e")

        title = "DentaScan — Análisis Radiométrico"
        if predicted_label:
            title += f"  |  Clase predicha: {predicted_label}"
        fig.suptitle(title, fontsize=13, color="white", fontweight="bold")

        hist_orig = cv2.calcHist([original], [0], None, [256], [0, 256]).flatten()
        hist_prep = cv2.calcHist([preprocessed], [0], None, [256], [0, 256]).flatten()

        colors = {
            "bg": "#16213e", "orig": "#00b4d8", "prep": "#90e0ef",
            "threshold": "#ff6b6b", "text": "#e0e0e0", "grid": "#333355",
        }

        for ax, hist, title_ax, color in [
            (axes[0], hist_orig, "Imagen Original", colors["orig"]),
            (axes[1], hist_prep, "Tras CLAHE + Filtros", colors["prep"]),
        ]:
            ax.set_facecolor(colors["bg"])
            ax.plot(hist, color=color, linewidth=1.5, alpha=0.9)
            ax.fill_between(range(256), hist, alpha=0.3, color=color)
            ax.axvline(x=80, color=colors["threshold"], linestyle="--", linewidth=1.2,
                       label="Umbral radiolúcido (80)")
            ax.set_title(title_ax, color=colors["text"], fontsize=10)
            ax.set_xlabel("Nivel de gris (0=negro, 255=blanco)", color=colors["text"], fontsize=8)
            ax.set_ylabel("Frecuencia de píxeles", color=colors["text"], fontsize=8)
            ax.tick_params(colors=colors["text"], labelsize=7)
            ax.grid(True, alpha=0.2, color=colors["grid"])
            ax.legend(fontsize=7, facecolor=colors["bg"], labelcolor=colors["text"])
            for spine in ax.spines.values():
                spine.set_edgecolor(colors["grid"])

        if features:
            feature_text = (
                f"media={features.get('media', 0):.1f}  "
                f"std={features.get('std', 0):.1f}  "
                f"prop_oscuros={features.get('prop_oscuros', 0):.3f}  "
                f"asimetria={features.get('asimetria', 0):.1f}"
            )
            fig.text(0.5, 0.01, feature_text, ha="center", fontsize=7.5,
                     color="#aaaacc", fontstyle="italic")

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def save_image(self, image: np.ndarray, output_path: str | Path) -> str:
        """Guarda imagen NumPy como PNG. Retorna la ruta guardada."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise ProcessingError(f"No se pudo guardar la imagen en {path}")
        return str(path)

    def save_bytes_as_png(self, data: bytes, output_path: str | Path) -> str:
        """Guarda bytes de imagen como archivo PNG."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return str(path)

    # ─── Helpers privados ─────────────────────────────────────────────────────

    @staticmethod
    def _region_color(
        region: dict,
        predicted_class: int | None,
    ) -> tuple[int, int, int]:
        """Devuelve el color BGR para una región según su lesion_type."""
        lesion_type = region.get("lesion_type", "")
        if lesion_type in LESION_COLORS_BGR:
            return LESION_COLORS_BGR[lesion_type]
        return CLASS_COLORS_BGR.get(predicted_class, (0, 200, 255))

    @staticmethod
    def _draw_dashed_rect(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple,
        thickness: int = 1,
        dash: int = 8,
    ) -> None:
        """Dibuja un rectángulo con línea discontinua."""
        pts = [(x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)]
        for ax, ay, bx, by in pts:
            length = int(((bx - ax)**2 + (by - ay)**2) ** 0.5)
            if length == 0:
                continue
            dx, dy = (bx - ax) / length, (by - ay) / length
            i = 0
            draw = True
            while i < length:
                if draw:
                    end = min(i + dash, length)
                    p1 = (int(ax + dx * i), int(ay + dy * i))
                    p2 = (int(ax + dx * end), int(ay + dy * end))
                    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
                i += dash
                draw = not draw

    @staticmethod
    def _draw_label(
        img: np.ndarray,
        x: int, y: int,
        text: str,
        color: tuple,
    ) -> None:
        """Dibuja etiqueta con borde negro + fondo de color sobre la región."""
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.46, 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        # Posicionar sobre la región si hay espacio, o dentro si está muy arriba
        label_y = y - 6 if y > th + 14 else y + th + 14
        label_y = max(th + pad + 2, min(label_y, img.shape[0] - baseline - pad))

        # Borde negro exterior
        cv2.rectangle(img,
                      (x - 1, label_y - th - pad - 1),
                      (x + tw + pad * 2 + 1, label_y + baseline + 1),
                      (0, 0, 0), cv2.FILLED)
        # Fondo de color
        cv2.rectangle(img,
                      (x, label_y - th - pad),
                      (x + tw + pad * 2, label_y + baseline),
                      color, cv2.FILLED)
        # Texto negro legible
        cv2.putText(img, text, (x + pad, label_y - 1),
                    font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_header(
        img: np.ndarray,
        predicted_label: str | None,
        confidence: float | None,
        predicted_class: int | None,
    ) -> None:
        """Cabecera superior con clasificación ML."""
        if not predicted_label:
            return
        color = CLASS_COLORS_BGR.get(predicted_class, (0, 200, 255))
        conf_text = f" ({confidence * 100:.1f}%)" if confidence is not None else ""
        label_text = f"DentaScan: {predicted_label}{conf_text}"

        overlay = img.copy()
        bar_h = 30
        cv2.rectangle(overlay, (0, 0), (img.shape[1], bar_h), (0, 0, 0), cv2.FILLED)
        img[:] = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)
        cv2.putText(img, label_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_footer(img: np.ndarray) -> None:
        """Pie con aviso ético."""
        text = "APOYO DIAGNOSTICO  —  No sustituye criterio clinico"
        cv2.putText(
            img, text,
            (8, img.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_heatmap_legend(img: np.ndarray) -> None:
        """Leyenda de escala del heatmap."""
        h, w = img.shape[:2]
        bar_w, bar_h = 120, 10
        x0, y0 = w - bar_w - 12, h - 28

        # Gradiente de color
        for i in range(bar_w):
            val = int(i / bar_w * 255)
            color_map = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)
            b, g, r = int(color_map[0, 0, 0]), int(color_map[0, 0, 1]), int(color_map[0, 0, 2])
            cv2.line(img, (x0 + i, y0), (x0 + i, y0 + bar_h), (b, g, r), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "Bajo", (x0, y0 + bar_h + 12), font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, "Alto", (x0 + bar_w - 22, y0 + bar_h + 12), font, 0.35, (200, 200, 200), 1)
        cv2.putText(img, "Riesgo", (x0 + bar_w // 2 - 15, y0 - 3), font, 0.35, (200, 200, 200), 1)


image_visualizer = ImageVisualizer()
