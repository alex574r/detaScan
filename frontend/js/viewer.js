/**
 * DentaScan — Visualizador de imágenes (modal-card dedicado).
 *
 * Modos de invocación:
 *   openImageViewer('annotated')           ← desde panel de resultados
 *   openImageViewer('annotated', srcUrl)   ← override directo (historial, etc.)
 *
 * Si __currentAnalysis no está disponible, cae al DOM:
 *   - Lee el <img> correspondiente en el panel
 *   - Usa su src
 *   - El visor sigue funcionando aunque no haya metadata clínica
 *
 * Funciones:
 *   - Zoom (rueda, botones, doble click)
 *   - Pan (drag)
 *   - Pantalla completa
 *   - Capas conmutables
 *   - Cierre rápido (Esc, click fuera, X)
 *   - Skeleton mientras carga
 */

// ── Configuración de capas ──────────────────────────────────────────────────

const VIEWER_LAYERS = {
  annotated: {
    title:     "Imagen anotada",
    desc:      "Hallazgos delimitados con contornos coloreados por tipo de lesión y nivel de confianza calibrada.",
    field:     "output_annotated",
    domImg:    "imgAnnotated",
    technical: "Composición final: original + máscara semitransparente por hallazgo + etiqueta clínica + cabecera con clasificación ML.",
  },
  heatmap: {
    title:     "Mapa de calor",
    desc:      "Distribución espacial de zonas anómalas. Azul = bajo riesgo · Amarillo = medio · Rojo = alta probabilidad de caries.",
    field:     "output_heatmap",
    domImg:    "imgHeatmap",
    technical: "Saliency basada en gradiente Sobel + oscuridad de píxel + bumps gaussianos en hallazgos detectados, fusionada con COLORMAP_JET.",
  },
  preprocessed: {
    title:     "CLAHE + Filtros",
    desc:      "Imagen tras preprocesamiento clínico avanzado.",
    field:     "output_preprocessed",
    domImg:    "imgPreprocessed",
    technical: "Normalización percentil [p1, p99] · corrección de iluminación · filtro bilateral · mediana · CLAHE multi-escala fusionada.",
  },
  edges: {
    title:     "Detección de bordes",
    desc:      "Detectores combinados para localizar contornos esmalte-dentina y límites de lesiones.",
    field:     "output_edges",
    domImg:    "imgEdges",
    technical: "Canny (verde, double-threshold con hysteresis) + Sobel normalizado (azul, gradiente direccional) superpuestos sobre el preprocesado.",
  },
  mask: {
    title:     "Máscara radiolúcida",
    desc:      "Zonas oscuras compatibles con lesiones cariosas potenciales.",
    field:     "output_mask",
    domImg:    "imgMask",
    technical: "Threshold adaptativo dentro del campo dental + DoG multi-escala + morfología (apertura/cierre) para eliminar ruido.",
  },
};

const VIEWER_STATE = {
  currentKey:  null,
  zoom:        1.0,
  panX:        0,
  panY:        0,
  isPanning:   false,
  panStartX:   0,
  panStartY:   0,
  panStartTX:  0,
  panStartTY:  0,
  compareMode: false,
  fullscreen:  false,
  eventsAttached: false,
};

// ── API principal ───────────────────────────────────────────────────────────

/**
 * Abre el visualizador para la capa indicada.
 * @param {string} layerKey  Uno de VIEWER_LAYERS
 * @param {string} [urlOverride]  URL directa para usar (opcional, p.ej. desde historial)
 */
function openImageViewer(layerKey, urlOverride) {
  const cfg = VIEWER_LAYERS[layerKey];
  if (!cfg) {
    console.warn("[viewer] capa desconocida:", layerKey);
    return;
  }

  const overlay = document.getElementById("imageViewer");
  if (!overlay) {
    console.error("[viewer] #imageViewer no existe en el DOM");
    return;
  }

  // 1) Determinar la URL a mostrar — prioridad:
  //    1. urlOverride explícito
  //    2. __currentAnalysis[cfg.field]
  //    3. src del <img> en el DOM
  let url = urlOverride || null;
  const a = (typeof __currentAnalysis !== "undefined") ? __currentAnalysis : null;
  if (!url && a && a[cfg.field]) {
    url = api.outputUrl(a[cfg.field]);
  }
  if (!url) {
    const domImg = document.getElementById(cfg.domImg);
    if (domImg && domImg.src && !domImg.src.endsWith("/")) {
      url = domImg.src;
    }
  }

  if (!url) {
    showViewerError(cfg, "Aún no hay imagen disponible para esta capa. Sube una radiografía y espera a que el análisis termine.");
    return;
  }

  // 2) Reset state
  VIEWER_STATE.currentKey  = layerKey;
  VIEWER_STATE.zoom        = 1.0;
  VIEWER_STATE.panX        = 0;
  VIEWER_STATE.panY        = 0;
  VIEWER_STATE.compareMode = false;

  // 3) Header
  setText("viewerTitle", cfg.title);
  setText("viewerDesc",  cfg.desc);

  // 4) Meta clínica (si tenemos análisis)
  if (a && (a.predicted_label !== undefined || a.lesion_findings !== undefined)) {
    renderViewerMeta(a, cfg);
  } else {
    const meta = document.getElementById("viewerMeta");
    if (meta) {
      meta.innerHTML = `
        <div class="viewer-meta-row viewer-meta-row-simple">
          <div class="viewer-meta-block viewer-meta-block-wide">
            <span class="viewer-meta-label">Procesamiento aplicado</span>
            <span class="viewer-meta-tech">${escapeViewerHtml(cfg.technical)}</span>
          </div>
        </div>
      `;
    }
  }

  // 5) Imagen principal con skeleton de carga
  const img    = document.getElementById("viewerImg");
  const stage  = document.getElementById("viewerStage");
  const canvas = document.getElementById("viewerCanvas");
  const compare = document.getElementById("viewerCompare");

  if (canvas)  canvas.style.display  = "flex";
  if (compare) compare.style.display = "none";

  if (img) {
    img.classList.add("viewer-loading");
    img.style.opacity = "0";
    img.alt = cfg.title;

    img.onload = () => {
      img.classList.remove("viewer-loading");
      img.style.opacity = "1";
      hideStageError();
    };
    img.onerror = () => {
      img.classList.remove("viewer-loading");
      img.style.opacity = "0";
      showStageError("No se pudo cargar la imagen. Verifica que el análisis haya terminado.");
    };
    img.src = url;
    if (img.decode) img.decode().catch(() => {});
  }

  viewerApplyTransform();

  // 6) Capas conmutables
  renderViewerLayers(layerKey, a);

  // 7) Mostrar modal — CSS visibility/opacity handles animation, no display hack needed
  overlay.classList.add("open");
  document.body.style.overflow = "hidden";

  ensureViewerEvents();
}

function closeImageViewer() {
  const overlay = document.getElementById("imageViewer");
  if (!overlay) return;
  // Removing .open triggers the CSS opacity+visibility transition — no setTimeout needed
  overlay.classList.remove("open");
  document.body.style.overflow = "";
  if (document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
  VIEWER_STATE.fullscreen = false;
  hideStageError();
}

// ── Zoom / Pan ──────────────────────────────────────────────────────────────

function viewerZoom(factor) {
  VIEWER_STATE.zoom = Math.min(8, Math.max(0.25, VIEWER_STATE.zoom * factor));
  viewerApplyTransform();
}

function viewerReset() {
  VIEWER_STATE.zoom = 1.0;
  VIEWER_STATE.panX = 0;
  VIEWER_STATE.panY = 0;
  viewerApplyTransform();
}

function viewerApplyTransform() {
  const img = document.getElementById("viewerImg");
  if (!img) return;
  img.style.transform =
    `translate(${VIEWER_STATE.panX}px, ${VIEWER_STATE.panY}px) ` +
    `scale(${VIEWER_STATE.zoom})`;
}

function viewerToggleFullscreen() {
  const card = document.getElementById("viewerCard");
  if (!card) return;
  if (!document.fullscreenElement) {
    card.requestFullscreen?.()
      .then(() => { VIEWER_STATE.fullscreen = true; })
      .catch(err => console.warn("[viewer] fullscreen falló:", err.message));
  } else {
    document.exitFullscreen?.()
      .then(() => { VIEWER_STATE.fullscreen = false; });
  }
}

function viewerToggleCompare() {
  const a = (typeof __currentAnalysis !== "undefined") ? __currentAnalysis : null;
  if (!a) {
    showViewerToast("La comparación requiere un análisis cargado.");
    return;
  }

  VIEWER_STATE.compareMode = !VIEWER_STATE.compareMode;
  const canvas  = document.getElementById("viewerCanvas");
  const compare = document.getElementById("viewerCompare");
  const btn     = document.getElementById("viewerCompareBtn");

  if (VIEWER_STATE.compareMode) {
    const cfg     = VIEWER_LAYERS[VIEWER_STATE.currentKey];
    const overlay = api.outputUrl(a[cfg.field]);
    const base    = api.outputUrl(a.output_preprocessed) || overlay;
    const baseImg = document.getElementById("compareBase");
    const ovImg   = document.getElementById("compareOverlay");
    if (baseImg) baseImg.src = base;
    if (ovImg)   ovImg.src   = overlay;
    if (canvas)  canvas.style.display  = "none";
    if (compare) compare.style.display = "flex";
    if (btn)     btn.classList.add("active");
    initCompareSlider();
  } else {
    if (canvas)  canvas.style.display  = "flex";
    if (compare) compare.style.display = "none";
    if (btn)     btn.classList.remove("active");
  }
}

// ── Renderizado de meta clínica ─────────────────────────────────────────────

function renderViewerMeta(a, cfg) {
  const meta = document.getElementById("viewerMeta");
  if (!meta) return;

  const conf = a.confidence_score != null
    ? (a.confidence_score * 100).toFixed(1) + "%"
    : "—";
  const above90 = (a.lesion_findings || [])
    .filter(f => (f.calibrated_confidence ?? f.confidence ?? 0) >= 0.90).length;

  const lesionsList = (a.lesion_findings || [])
    .slice(0, 4)
    .map(f => {
      const pct = ((f.calibrated_confidence ?? f.confidence ?? 0) * 100).toFixed(0);
      return `<span class="viewer-chip">${escapeViewerHtml(f.lesion_type)} · ${pct}%</span>`;
    })
    .join("");

  meta.innerHTML = `
    <div class="viewer-meta-row">
      <div class="viewer-meta-block">
        <span class="viewer-meta-label">Clasificación ML</span>
        <strong>${escapeViewerHtml(a.predicted_label || "—")}</strong>
        <span class="viewer-meta-sub">${conf} · ${escapeViewerHtml(a.model_used || "—")}</span>
      </div>
      <div class="viewer-meta-block">
        <span class="viewer-meta-label">Hallazgos ≥ 90%</span>
        <strong>${above90}</strong>
        <span class="viewer-meta-sub">${lesionsList || "Sin hallazgos sobre el umbral clínico"}</span>
      </div>
      <div class="viewer-meta-block viewer-meta-block-wide">
        <span class="viewer-meta-label">Procesamiento aplicado</span>
        <span class="viewer-meta-tech">${escapeViewerHtml(cfg.technical)}</span>
      </div>
    </div>
  `;
}

// ── Capas conmutables ───────────────────────────────────────────────────────

function renderViewerLayers(activeKey, analysisOverride) {
  const wrap = document.getElementById("viewerLayers");
  if (!wrap) return;

  const a = analysisOverride
    || ((typeof __currentAnalysis !== "undefined") ? __currentAnalysis : null);

  const html = Object.entries(VIEWER_LAYERS).map(([key, cfg]) => {
    // Disponible si tenemos URL (vía a[field] o via DOM)
    let available = false;
    if (a && a[cfg.field]) {
      available = !!api.outputUrl(a[cfg.field]);
    }
    if (!available) {
      const domImg = document.getElementById(cfg.domImg);
      available = !!(domImg && domImg.src && !domImg.src.endsWith("/"));
    }
    if (!available) return "";

    const active = key === activeKey ? "active" : "";
    return `
      <button type="button" class="viewer-layer ${active}"
              onclick="openImageViewer('${key}')">
        ${escapeViewerHtml(cfg.title)}
      </button>
    `;
  }).join("");

  wrap.innerHTML = '<span class="viewer-layers-label">Capas:</span>' + html;
}

// ── Compare slider ──────────────────────────────────────────────────────────

function initCompareSlider() {
  const slider = document.getElementById("compareSlider");
  const top    = document.getElementById("compareTop");
  if (!slider || !top) return;
  const wrap = slider.parentElement;
  let dragging = false;

  function setPosition(clientX) {
    const r = wrap.getBoundingClientRect();
    const x = Math.max(0, Math.min(r.width, clientX - r.left));
    const pct = (x / r.width) * 100;
    top.style.width    = pct + "%";
    slider.style.left  = pct + "%";
  }

  slider.onmousedown = (e) => { dragging = true; e.preventDefault(); };
  wrap.onmousemove   = (e) => { if (dragging) setPosition(e.clientX); };
  window.addEventListener("mouseup", () => { dragging = false; }, { once: true });

  slider.ontouchstart = () => { dragging = true; };
  wrap.ontouchmove    = (e) => { if (dragging && e.touches[0]) setPosition(e.touches[0].clientX); };
  window.addEventListener("touchend", () => { dragging = false; }, { once: true });

  setPosition(wrap.getBoundingClientRect().left + wrap.offsetWidth / 2);
}

// ── Errores y toast dentro del modal ────────────────────────────────────────

function showViewerError(cfg, message) {
  const overlay = document.getElementById("imageViewer");
  if (!overlay) return;

  setText("viewerTitle", cfg.title);
  setText("viewerDesc",  cfg.desc || "");

  const meta = document.getElementById("viewerMeta");
  if (meta) meta.innerHTML = "";

  const layers = document.getElementById("viewerLayers");
  if (layers) layers.innerHTML = "";

  const img = document.getElementById("viewerImg");
  if (img) { img.src = ""; img.style.opacity = "0"; }

  showStageError(message);

  overlay.classList.add("open");
  document.body.style.overflow = "hidden";

  ensureViewerEvents();
}

function showStageError(msg) {
  let box = document.getElementById("viewerStageError");
  if (!box) {
    const stage = document.getElementById("viewerStage");
    if (!stage) return;
    box = document.createElement("div");
    box.id = "viewerStageError";
    box.className = "viewer-stage-error";
    stage.appendChild(box);
  }
  box.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" width="40" height="40">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="currentColor"/>
    </svg>
    <p>${escapeViewerHtml(msg)}</p>
  `;
  box.style.display = "flex";
}

function hideStageError() {
  const box = document.getElementById("viewerStageError");
  if (box) box.style.display = "none";
}

function showViewerToast(msg) {
  let toast = document.getElementById("viewerToast");
  if (!toast) {
    const card = document.getElementById("viewerCard");
    if (!card) return;
    toast = document.createElement("div");
    toast.id = "viewerToast";
    toast.className = "viewer-toast";
    card.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), 2200);
}

// ── Eventos: pan, wheel, doble-click, Esc ───────────────────────────────────

function ensureViewerEvents() {
  if (VIEWER_STATE.eventsAttached) return;
  const stage = document.getElementById("viewerStage");
  if (!stage) return;
  VIEWER_STATE.eventsAttached = true;

  // Pan
  stage.addEventListener("mousedown", (e) => {
    if (VIEWER_STATE.compareMode) return;
    VIEWER_STATE.isPanning  = true;
    VIEWER_STATE.panStartX  = e.clientX;
    VIEWER_STATE.panStartY  = e.clientY;
    VIEWER_STATE.panStartTX = VIEWER_STATE.panX;
    VIEWER_STATE.panStartTY = VIEWER_STATE.panY;
    stage.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!VIEWER_STATE.isPanning) return;
    VIEWER_STATE.panX = VIEWER_STATE.panStartTX + (e.clientX - VIEWER_STATE.panStartX);
    VIEWER_STATE.panY = VIEWER_STATE.panStartTY + (e.clientY - VIEWER_STATE.panStartY);
    viewerApplyTransform();
  });

  window.addEventListener("mouseup", () => {
    if (VIEWER_STATE.isPanning) {
      VIEWER_STATE.isPanning = false;
      stage.style.cursor = "";
    }
  });

  // Zoom rueda
  stage.addEventListener("wheel", (e) => {
    if (VIEWER_STATE.compareMode) return;
    const overlay = document.getElementById("imageViewer");
    if (!overlay || !overlay.classList.contains("open")) return;
    e.preventDefault();
    viewerZoom(e.deltaY < 0 ? 1.10 : 0.91);
  }, { passive: false });

  // Doble-click resetea
  stage.addEventListener("dblclick", () => {
    if (VIEWER_STATE.compareMode) return;
    viewerReset();
  });

  // Touch pan
  let touchPanX = 0, touchPanY = 0;
  stage.addEventListener("touchstart", (e) => {
    if (VIEWER_STATE.compareMode || e.touches.length !== 1) return;
    VIEWER_STATE.isPanning  = true;
    touchPanX = e.touches[0].clientX;
    touchPanY = e.touches[0].clientY;
    VIEWER_STATE.panStartTX = VIEWER_STATE.panX;
    VIEWER_STATE.panStartTY = VIEWER_STATE.panY;
  }, { passive: true });

  stage.addEventListener("touchmove", (e) => {
    if (!VIEWER_STATE.isPanning || e.touches.length !== 1) return;
    VIEWER_STATE.panX = VIEWER_STATE.panStartTX + (e.touches[0].clientX - touchPanX);
    VIEWER_STATE.panY = VIEWER_STATE.panStartTY + (e.touches[0].clientY - touchPanY);
    viewerApplyTransform();
  }, { passive: true });

  stage.addEventListener("touchend", () => { VIEWER_STATE.isPanning = false; });

  // Esc cierra; F fullscreen; +/- zoom
  window.addEventListener("keydown", (e) => {
    const overlay = document.getElementById("imageViewer");
    if (!overlay || !overlay.classList.contains("open")) return;
    if (e.key === "Escape")        closeImageViewer();
    else if (e.key === "f" || e.key === "F") viewerToggleFullscreen();
    else if (e.key === "+" || e.key === "=") viewerZoom(1.2);
    else if (e.key === "-" || e.key === "_") viewerZoom(0.83);
    else if (e.key === "0")        viewerReset();
  });

  // Click en fondo cierra
  const overlay = document.getElementById("imageViewer");
  if (overlay) {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) closeImageViewer();
    });
  }
}

// Auto-attach (en caso de que DOMContentLoaded ya disparó)
if (document.readyState !== "loading") {
  ensureViewerEvents();
} else {
  document.addEventListener("DOMContentLoaded", ensureViewerEvents, { once: true });
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = txt;
}

function escapeViewerHtml(str) {
  return (str ?? "").toString()
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
