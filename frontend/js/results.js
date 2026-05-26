/**
 * DentaScan — Renderizado de resultados del análisis
 */

// ── Iconos SVG inline ────────────────────────────────────────────────────────

const ICONS = {
  healthy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-5"/></svg>`,
  warning: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="1" fill="currentColor"/></svg>`,
  danger:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="currentColor"/></svg>`,
  bone:    `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M17.5 2A3.5 3.5 0 0014.13 5.6H9.87A3.5 3.5 0 106.5 9a3.48 3.48 0 00.87-.11v6.22A3.5 3.5 0 106.5 18.5a3.48 3.48 0 00.87-.11v.11h6.26a3.5 3.5 0 103.37-4.5 3.48 3.48 0 00-.87.11V9.11A3.5 3.5 0 0017.5 2z"/></svg>`,
  caries:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2C7 2 5.5 3.5 5.5 6c0 1.5.5 2.5.5 4 0 2-1 3-1 5 0 3 2 5 4 5s2.5-1.5 3-1.5S13.5 21 15 21c2 0 4-2 4-5 0-2-1-3-1-5 0-1.5.5-2.5.5-4C18.5 3.5 17 2 15 2H9z"/><circle cx="12" cy="11" r="2" fill="currentColor" opacity="0.4"/></svg>`,
  chart:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>`,
  search:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
  heatmap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M16.9 16.9l2.1 2.1M4.9 19.1l2.1-2.1M16.9 7.1l2.1-2.1"/></svg>`,
  radiolucent: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8" stroke-dasharray="2 2"/></svg>`,
  dense:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z" fill="currentColor" opacity="0.3"/></svg>`,
};

// ── Icono por clase ML ────────────────────────────────────────────────────────

const CLASS_ICONS = {
  "Diente Sano":         ICONS.healthy,
  "Caries Incipiente":   ICONS.caries,
  "Caries Avanzada":     ICONS.caries,
  "Absceso Periapical":  ICONS.warning,
  "Lesión Ósea":         ICONS.bone,
  // Sub-tipos óseos (clasificación por región)
  "Granuloma Periapical":    ICONS.warning,
  "Quiste Periapical":       ICONS.warning,
  "Lesión Periapical Difusa": ICONS.danger,
  "Periodontitis Leve":      ICONS.bone,
  "Periodontitis Severa":    ICONS.bone,
  "Resorción Ósea":          ICONS.bone,
  "Osteítis Condensante":    ICONS.dense,
  "Hipercementosis":         ICONS.dense,
};

// ── Color por clase / tipo ────────────────────────────────────────────────────

const CLASS_COLORS = {
  "Diente Sano":               "#22c55e",
  "Caries Incipiente":         "#f0c800",
  "Caries Avanzada":           "#ff7800",
  "Caries Oclusal":            "#ff8c00",
  "Caries Oclusal Incipiente": "#f0c800",
  "Caries Oclusal Avanzada":   "#ff5a00",
  "Caries Interproximal":      "#ffb428",
  "Caries Recurrente":         "#ff2828",
  "Absceso Periapical":        "#ef4444",
  "Lesión Ósea":               "#a855f7",
  "Granuloma Periapical":      "#ff503c",
  "Quiste Periapical":         "#b400c8",
  "Lesión Periapical Difusa":  "#c80000",
  "Periodontitis Leve":        "#00c896",
  "Periodontitis Severa":      "#ff5000",
  "Resorción Ósea":            "#c800c8",
  "Osteítis Condensante":      "#ffdc00",
  "Hipercementosis":           "#00f0dc",
  "Restauración":              "#9aa0a6",
};

const SEVERITY_COLORS = {
  leve:     "#22c55e",
  moderada: "#f59e0b",
  severa:   "#ef4444",
};

const FEATURE_DESCRIPTIONS = {
  media:        "Media de intensidad de píxeles",
  std:          "Desviación estándar (varianza tonal)",
  min_px:       "Valor mínimo de píxel",
  max_px:       "Valor máximo de píxel",
  bordes_mean:  "Densidad de bordes (Canny)",
  sobel_mean:   "Gradiente Sobel promedio",
  zona_tl:      "Cuadrante superior-izquierdo",
  zona_tr:      "Cuadrante superior-derecho",
  zona_bl:      "Cuadrante inferior-izquierdo",
  zona_br:      "Cuadrante inferior-derecho",
  prop_oscuros: "Proporción de píxeles oscuros (radiolucidez)",
  asimetria:    "Asimetría bilateral",
};

// ── Punto de entrada principal ────────────────────────────────────────────────

let __currentAnalysis = null;

function showResults(analysis) {
  __currentAnalysis = analysis;
  const btn = document.getElementById("btnAnalyze");
  if (btn) setLoading(btn, false);

  document.getElementById("analysisStatus").style.display = "none";
  const panels = document.getElementById("resultPanels");
  panels.style.display = "flex";
  panels.classList.add("fade-in");

  renderClassification(analysis);
  renderLesionFindings(analysis);
  renderImages(analysis);
  renderFeatures(analysis);
  renderMeta(analysis);
}

// ── Clasificación ML ──────────────────────────────────────────────────────────

function renderClassification(a) {
  const label = a.predicted_label || "No disponible";
  const confidence = a.confidence_score ?? 0;
  const color = CLASS_COLORS[label] || "#00c8ff";
  const icon = CLASS_ICONS[label] || ICONS.search;

  const badge = document.getElementById("classBadge");
  badge.innerHTML = icon;
  badge.style.background = color + "22";
  badge.style.border = `2px solid ${color}`;
  badge.style.color = color;

  document.getElementById("classLabel").textContent = label;
  document.getElementById("classLabel").style.color = color;

  const pct = Math.round(confidence * 100);
  document.getElementById("confidenceFill").style.width = pct + "%";
  document.getElementById("confidenceFill").style.background = color;
  document.getElementById("confidenceText").textContent = `Confianza: ${pct}%`;

  const probsEl = document.getElementById("classProbabilities");
  probsEl.innerHTML = "";
  if (a.class_probabilities) {
    Object.entries(a.class_probabilities)
      .sort((x, y) => y[1] - x[1])
      .forEach(([cls, prob]) => {
        const chip = document.createElement("span");
        chip.className = "prob-chip" + (cls === label ? " top" : "");
        chip.textContent = `${cls}: ${(prob * 100).toFixed(1)}%`;
        probsEl.appendChild(chip);
      });
  }

  const findings = a.lesion_findings || [];
  const above90 = findings.filter(f => (f.calibrated_confidence ?? f.confidence ?? 0) >= 0.90).length;
  document.getElementById("modelInfo").innerHTML =
    `Modelo: <strong>${a.model_used || "—"}</strong> · ` +
    `Tiempo: ${a.processing_time_ms ? a.processing_time_ms.toFixed(0) + " ms" : "—"} · ` +
    `Hallazgos clínicos (≥ 90%): <strong>${above90}</strong>`;
}

// ── Panel de hallazgos por región ─────────────────────────────────────────────

function renderLesionFindings(a) {
  const card = document.getElementById("findingsCard");
  const list = document.getElementById("findingsList");

  card.style.display = "block";
  list.innerHTML = "";

  const findings = a.lesion_findings || [];

  // Si no hay hallazgos sobre el umbral clínico, mostrar mensaje explicativo
  if (findings.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-findings";
    empty.innerHTML = `
      <div style="text-align:center;padding:1.5rem;border:1px dashed var(--border);border-radius:10px">
        <p style="color:var(--text-muted);margin:0 0 .35rem 0;font-weight:600">
          Sin hallazgos con confianza clínica ≥ 90%
        </p>
        <p style="color:var(--text-muted);font-size:.85rem;margin:0">
          No se identificaron lesiones cuya evidencia visual y estadística superen el umbral diagnóstico.
          Esto puede significar que la radiografía no muestra anomalías significativas o que las regiones
          dudosas no alcanzaron el nivel de certeza requerido.
        </p>
      </div>`;
    list.appendChild(empty);
    return;
  }

  // Agrupar por tipo
  const byType = {};
  findings.forEach(f => {
    const t = f.lesion_type || "Desconocido";
    if (!byType[t]) byType[t] = [];
    byType[t].push(f);
  });

  // Ordenar por confianza calibrada desc → severidad
  const severityOrder = { severa: 0, moderada: 1, leve: 2 };
  const sorted = findings.slice().sort((x, y) => {
    const cx = y.calibrated_confidence ?? y.confidence ?? 0;
    const cy = x.calibrated_confidence ?? x.confidence ?? 0;
    if (cx !== cy) return cx - cy;
    return (severityOrder[x.severity] ?? 3) - (severityOrder[y.severity] ?? 3);
  });

  sorted.forEach((f) => {
    const color = CLASS_COLORS[f.lesion_type] || "#8b95b0";
    const sevColor = SEVERITY_COLORS[f.severity] || "#8b95b0";
    const icon = CLASS_ICONS[f.lesion_type] || (f.is_radiopaque ? ICONS.dense : ICONS.radiolucent);
    const typeLabel = f.is_radiopaque ? "Radiopaca" : "Radiolúcida";
    const area = f.area_px ? Math.round(f.area_px) : "—";
    const cal = f.calibrated_confidence ?? f.confidence ?? 0;
    const pct = Math.round(cal * 100);
    const reasoning = f.clinical_reasoning ? escapeHtml(f.clinical_reasoning) : "";
    const recurrentTag = f.is_recurrent
      ? `<span class="finding-tag" style="margin-left:.35rem;background:#ff282822;color:#ff5050;border:1px solid #ff505044;padding:.05rem .4rem;border-radius:8px;font-size:.7rem">recurrente</span>`
      : "";

    const item = document.createElement("div");
    item.className = `finding-item severity-${f.severity || "leve"}`;
    item.innerHTML = `
      <div class="finding-icon" style="color:${color}">${icon}</div>
      <div class="finding-detail" style="flex:1">
        <span class="finding-type" style="color:${color}">
          ${escapeHtml(f.lesion_type || "Región anómala")}${recurrentTag}
        </span>
        <span class="finding-meta">${typeLabel} · Área: ${area} px · Pos: (${f.centroid_x}, ${f.centroid_y})</span>
        ${reasoning ? `<span class="finding-meta" style="opacity:.75;font-style:italic">${reasoning}</span>` : ""}
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:.25rem">
        <span class="severity-pill" style="background:${sevColor}22;color:${sevColor};border-color:${sevColor}55">
          ${f.severity || "—"}
        </span>
        <span style="font-size:.75rem;color:${color};font-weight:700">${pct}% conf.</span>
      </div>
    `;
    list.appendChild(item);
  });

  // Resumen de tipos detectados
  const typesFound = Object.keys(byType);
  if (typesFound.length > 0) {
    const summary = document.createElement("div");
    summary.className = "findings-summary";
    summary.innerHTML = typesFound.map(t => {
      const c = CLASS_COLORS[t] || "#8b95b0";
      return `<span class="finding-tag" style="border-color:${c}44;color:${c}">${escapeHtml(t)} (${byType[t].length})</span>`;
    }).join("");
    list.insertBefore(summary, list.firstChild);
  }
}

// ── Imágenes de resultado ─────────────────────────────────────────────────────

function renderImages(a) {
  const imgFields = [
    { id: "imgAnnotated",   dl: "dlAnnotated",   key: "output_annotated" },
    { id: "imgHeatmap",     dl: "dlHeatmap",     key: "output_heatmap" },
    { id: "imgPreprocessed",dl: "dlPreprocessed",key: "output_preprocessed" },
    { id: "imgEdges",       dl: "dlEdges",       key: "output_edges" },
    { id: "imgMask",        dl: "dlMask",        key: "output_mask" },
    { id: "imgHistogram",   dl: "dlHistogram",   key: "output_histogram" },
  ];

  imgFields.forEach(({ id, dl, key }) => {
    const url = api.outputUrl(a[key]);
    const imgEl = document.getElementById(id);
    if (!imgEl) return;
    if (url) {
      imgEl.src = url;
      const dlEl = document.getElementById(dl);
      if (dlEl) { dlEl.href = url; dlEl.download = a[key]?.split("/").pop() || "image.png"; }
    }
  });
}

// ── Tabla de features ─────────────────────────────────────────────────────────

function renderFeatures(a) {
  const tbody = document.getElementById("featuresBody");
  tbody.innerHTML = "";
  if (!a.features) {
    tbody.innerHTML = '<tr><td colspan="3" style="color:var(--text-muted)">No disponible</td></tr>';
    return;
  }
  Object.entries(a.features).forEach(([key, val]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${key}</td>
      <td><span class="feat-val">${typeof val === "number" ? val.toFixed(4) : val}</span></td>
      <td style="color:var(--text-muted)">${FEATURE_DESCRIPTIONS[key] || key}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Metadatos ─────────────────────────────────────────────────────────────────

function renderMeta(a) {
  const el = document.getElementById("metaPanel");
  const items = [
    ["Archivo",    a.original_filename],
    ["Formato",    a.file_format],
    ["Tipo de Rx", a.xray_type],
    ["Estado",     a.status],
    ["Análisis ID","#" + a.id],
    ["Fecha",      new Date(a.created_at).toLocaleString("es-MX")],
  ];
  el.innerHTML = items
    .filter(([, v]) => v)
    .map(([k, v]) => `<span><strong>${k}:</strong> ${v}</span>`)
    .join("");
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

async function loadDashboard() {
  try {
    const summary = await api.getSummary();
    document.getElementById("statTotal").textContent     = summary.total;
    document.getElementById("statCompleted").textContent = summary.completed;
    document.getElementById("statPending").textContent   = summary.pending;
    document.getElementById("statFailed").textContent    = summary.failed;
  } catch { /* silently ignore */ }
}

// ── Historial ─────────────────────────────────────────────────────────────────

async function loadHistory() {
  const list = document.getElementById("historyList");
  try {
    const analyses = await api.listAnalyses(0, 30);
    const btnClear = document.getElementById("btnClearHistory");
    if (!analyses.length) {
      list.innerHTML = '<p class="empty-state">Aún no tienes análisis. Sube tu primera radiografía.</p>';
      if (btnClear) btnClear.disabled = true;
      return;
    }
    list.innerHTML = analyses.map(renderHistoryItem).join("");
    if (btnClear) btnClear.disabled = false;
  } catch (err) {
    list.innerHTML = `<p class="empty-state">Error cargando historial: ${err.message}</p>`;
  }
}

// ── Eliminar todo el historial ────────────────────────────────────────────────
async function clearAllHistory() {
  const btn = document.getElementById("btnClearHistory");
  // Confirmación robusta
  const ok = window.confirm(
    "¿Eliminar TODOS los análisis del historial?\n\n" +
    "Esta acción no se puede deshacer. Se borrarán los registros y sus imágenes asociadas."
  );
  if (!ok) return;

  if (btn) {
    btn.disabled = true;
    btn.classList.add("loading");
  }

  try {
    const result = await api.deleteAllAnalyses();
    const n = result?.deleted ?? 0;
    if (n === 0) {
      showHistoryToast("No había análisis para eliminar.", "info");
    } else {
      showHistoryToast(`${n} análisis eliminados del historial.`, "success");
    }
    // Si estábamos viendo un análisis, limpiar la pantalla de resultados
    __currentAnalysis = null;
    const panels  = document.getElementById("resultPanels");
    const results = document.getElementById("resultsSection");
    if (panels)  panels.style.display  = "none";
    if (results) results.style.display = "none";

    // Refrescar dashboard + historial
    await Promise.all([loadDashboard(), loadHistory()]);
  } catch (err) {
    console.error("[history] eliminar falló:", err);
    showHistoryToast("No se pudo eliminar el historial: " + err.message, "error");
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.classList.remove("loading");
    }
  }
}

function showHistoryToast(msg, kind = "info") {
  let toast = document.getElementById("historyToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "historyToast";
    toast.className = "history-toast";
    document.body.appendChild(toast);
  }
  toast.dataset.kind = kind;
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function renderHistoryItem(a) {
  const label = a.predicted_label || "—";
  const color = CLASS_COLORS[label] || "#8b95b0";
  const thumb = a.output_annotated
    ? `<img class="history-thumb" src="${api.outputUrl(a.output_annotated)}" alt="thumb" onerror="this.style.display='none'" />`
    : `<div class="history-thumb placeholder" style="color:var(--text-muted)">${ICONS.caries}</div>`;

  return `
    <div class="history-item" onclick="openAnalysisDetail(${a.id})">
      ${thumb}
      <div class="history-info">
        <p class="history-filename">${escapeHtml(a.original_filename)}</p>
        <p class="history-meta">${a.xray_type} &middot; ${new Date(a.created_at).toLocaleDateString("es-MX")}</p>
        <p class="history-label" style="color:${color}">${label}</p>
      </div>
      <span class="status-badge ${a.status}">${a.status}</span>
    </div>
  `;
}

async function openAnalysisDetail(id) {
  const modal = document.getElementById("detailModal");
  const body  = document.getElementById("modalBody");
  body.innerHTML = '<p style="color:var(--text-muted)">Cargando...</p>';
  modal.style.display = "flex";

  try {
    const a = await api.getAnalysis(id);
    // Setear el análisis actual para que el visor de imágenes pueda leerlo
    __currentAnalysis = a;
    document.getElementById("modalTitle").textContent = a.original_filename;
    const color = CLASS_COLORS[a.predicted_label] || "#ccc";
    const findingsHtml = (a.lesion_findings && a.lesion_findings.length)
      ? `<div style="margin-top:.75rem">
           <strong>Hallazgos (${a.lesion_findings.length}):</strong>
           <div style="display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.4rem">
             ${a.lesion_findings.map(f => {
               const c = CLASS_COLORS[f.lesion_type] || "#8b95b0";
               return `<span style="font-size:.75rem;padding:.2rem .55rem;border-radius:20px;border:1px solid ${c}44;color:${c}">${escapeHtml(f.lesion_type)} [${f.severity}]</span>`;
             }).join("")}
           </div>
         </div>`
      : "";

    body.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:1rem">
        ${a.output_annotated ? `<img src="${api.outputUrl(a.output_annotated)}" style="width:100%;border-radius:8px;border:1px solid var(--border)" />` : ""}
        <p><strong>Estado:</strong> ${a.status}</p>
        <p><strong>Predicción:</strong> <span style="color:${color}">${a.predicted_label || "—"}</span></p>
        <p><strong>Confianza:</strong> ${a.confidence_score != null ? (a.confidence_score * 100).toFixed(1) + "%" : "—"}</p>
        <p><strong>Modelo:</strong> ${a.model_used || "—"}</p>
        <p><strong>Tiempo:</strong> ${a.processing_time_ms ? a.processing_time_ms.toFixed(0) + " ms" : "—"}</p>
        ${findingsHtml}
        ${a.error_message ? `<p style="color:var(--red)"><strong>Error:</strong> ${escapeHtml(a.error_message)}</p>` : ""}
      </div>
    `;
  } catch (err) {
    body.innerHTML = `<p style="color:var(--red)">Error: ${err.message}</p>`;
  }
}

function closeModal(e) {
  if (e.target.id === "detailModal") {
    document.getElementById("detailModal").style.display = "none";
  }
}

function escapeHtml(str) {
  return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Exportación de resultados ──────────────────────────────────────────────
function exportAnalysisJSON() {
  const a = __currentAnalysis;
  if (!a) { alert("No hay un análisis cargado."); return; }
  const payload = {
    id: a.id,
    fecha: a.created_at,
    archivo: a.original_filename,
    tipo_radiografia: a.xray_type,
    modelo: a.model_used,
    clasificacion: {
      etiqueta: a.predicted_label,
      confianza: a.confidence_score,
      probabilidades: a.class_probabilities,
    },
    hallazgos: a.lesion_findings || [],
    features: a.features || {},
    tiempo_procesamiento_ms: a.processing_time_ms,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  triggerDownload(blob, `analisis_${a.id}.json`);
}

function exportAnalysisCSV() {
  const a = __currentAnalysis;
  if (!a || !a.lesion_findings || !a.lesion_findings.length) {
    alert("No hay hallazgos para exportar.");
    return;
  }
  const headers = ["tipo", "severidad", "confianza_calibrada", "x", "y", "ancho", "alto",
                   "area_px", "circularidad", "solidez", "intensidad_media",
                   "radiopaca", "recurrente", "justificacion"];
  const rows = a.lesion_findings.map(f => [
    f.lesion_type, f.severity, f.calibrated_confidence ?? f.confidence ?? "",
    f.x, f.y, f.width, f.height,
    f.area_px, f.circularity, f.solidity ?? "", f.mean_intensity,
    f.is_radiopaque ? "si" : "no",
    f.is_recurrent ? "si" : "no",
    f.clinical_reasoning ?? "",
  ].map(v => `"${String(v ?? "").replace(/"/g, '""')}"`).join(","));
  const csv = [headers.join(","), ...rows].join("\n");
  triggerDownload(new Blob([csv], { type: "text/csv;charset=utf-8" }), `hallazgos_${a.id}.csv`);
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename;
  document.body.appendChild(link); link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
