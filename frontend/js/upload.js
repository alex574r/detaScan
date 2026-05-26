/**
 * DentaScan — Manejo de carga de archivos y drag & drop
 */

let selectedFile = null;
let analysisMode = "all";          // "all" | "targeted"
const selectedLesionTypes = new Set();

const LESION_TYPES = [
  "Caries Incipiente", "Caries Avanzada",
  "Granuloma Periapical", "Absceso Periapical", "Quiste Periapical",
  "Lesión Periapical Difusa",
  "Periodontitis Leve", "Periodontitis Severa", "Resorción Ósea",
  "Osteítis Condensante", "Hipercementosis",
];

// ── Render chips de filtro de lesiones ──────────────────────────────────────
function renderLesionChips() {
  const wrap = document.getElementById("lesionChips");
  if (!wrap) return;
  wrap.innerHTML = LESION_TYPES.map(t => {
    const active = selectedLesionTypes.has(t) ? "active" : "";
    return `<span class="lesion-chip ${active}" onclick="toggleLesionType('${t.replace(/'/g, "\\'")}')">${t}</span>`;
  }).join("");
}

function toggleLesionType(t) {
  if (selectedLesionTypes.has(t)) selectedLesionTypes.delete(t);
  else selectedLesionTypes.add(t);
  renderLesionChips();
}

function setAnalysisMode(mode) {
  analysisMode = mode;
  document.getElementById("modeAll").classList.toggle("active", mode === "all");
  document.getElementById("modeTargeted").classList.toggle("active", mode === "targeted");
  const filter = document.getElementById("lesionFilter");
  if (filter) filter.style.display = mode === "targeted" ? "block" : "none";
  if (mode === "targeted") renderLesionChips();
}

// ── Drag & Drop ──────────────────────────────────────────────────────────────

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragging");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

// ── File handling ────────────────────────────────────────────────────────────

function setFile(file) {
  const allowed = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    fileInput.value = "";
    alert(`Formato no soportado: ${ext}\nUsa PNG, TIFF, DICOM o JPEG.`);
    return;
  }

  selectedFile = file;
  document.getElementById("previewName").textContent = file.name;
  document.getElementById("previewSize").textContent = formatBytes(file.size);
  document.getElementById("filePreview").style.display = "block";
  document.getElementById("btnAnalyze").disabled = false;
  if (typeof clearValidationError === "function") clearValidationError();
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  document.getElementById("filePreview").style.display = "none";
  document.getElementById("btnAnalyze").disabled = true;
}

// ── Upload & Analyze ─────────────────────────────────────────────────────────

async function uploadAndAnalyze() {
  if (!selectedFile) return;

  const btn = document.getElementById("btnAnalyze");
  setLoading(btn, true);

  const xrayType = document.getElementById("xrayType").value;
  const model = document.getElementById("mlModel").value;
  const sensitivity = document.getElementById("sensitivity").value;
  const minConfidence = parseInt(document.getElementById("minConfidence").value, 10) / 100;
  const useYolo = document.getElementById("useYolo")?.checked || false;

  const options = {
    analysis_mode: analysisMode,
    sensitivity,
    min_confidence: minConfidence,
    lesion_types: analysisMode === "targeted" ? Array.from(selectedLesionTypes) : [],
    use_yolo: useYolo,
  };

  try {
    clearValidationError();
    const analysis = await api.uploadAnalysis(selectedFile, xrayType, model, options);
    showResultsSection(analysis.id);
  } catch (err) {
    setLoading(btn, false);
    clearFile();
    console.error("[upload] error subida:", err, "detail:", err?.detail);
    // Cualquier detail estructurado (objeto con message/reasons) se renderiza
    // como tarjeta de validación.
    const d = err && err.detail;
    if (d && typeof d === "object" && !Array.isArray(d) &&
        (typeof d.message === "string" || Array.isArray(d.reasons))) {
      renderValidationError(d);
      return;
    }
    // Si el detail es array (validación FastAPI de schema), tomar el primer mensaje
    if (Array.isArray(d) && d.length) {
      const first = d[0];
      const msg = (first && typeof first.msg === "string") ? first.msg : JSON.stringify(first);
      alert("Error al subir el archivo: " + msg);
      return;
    }
    // Fallback: garantizamos un string legible — nunca "[object Object]"
    let msg = "Error desconocido";
    if (err && typeof err.message === "string" && err.message !== "[object Object]") {
      msg = err.message;
    } else if (d && typeof d === "string") {
      msg = d;
    } else if (d) {
      try { msg = JSON.stringify(d); } catch { /* keep fallback */ }
    }
    alert("Error al subir el archivo: " + msg);
  }
}

// ── Renderizado del mensaje de validación rechazada ──────────────────────────

function renderValidationError(detail) {
  let box = document.getElementById("validationErrorBox");
  if (!box) {
    box = document.createElement("div");
    box.id = "validationErrorBox";
    box.className = "validation-error";
    const card = document.getElementById("filePreview")?.parentElement
              || document.getElementById("dropZone")?.parentElement
              || document.body;
    card.appendChild(box);
  }
  const reasons = Array.isArray(detail.reasons) ? detail.reasons : [];
  const metrics = detail.metrics || {};
  const dental  = detail.dental_likelihood != null
    ? `Dental: ${(detail.dental_likelihood * 100).toFixed(0)}%` : "";
  const qual    = detail.quality_score != null
    ? `Calidad: ${(detail.quality_score * 100).toFixed(0)}%` : "";
  const res     = metrics.resolution ? `Resolución: ${metrics.resolution[0]}×${metrics.resolution[1]}` : "";
  const sharp   = metrics.sharpness != null ? `Nitidez: ${metrics.sharpness}` : "";
  const contr   = metrics.contrast != null ? `Contraste: ${metrics.contrast}` : "";

  box.innerHTML = `
    <div class="validation-error-title">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18" stroke-linecap="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="17" r="1" fill="currentColor"/>
      </svg>
      Imagen no válida para análisis clínico
    </div>
    <div class="validation-error-msg">
      ${escapeUploadHtml(detail.message ||
        "La imagen cargada no corresponde a una radiografía o imagen dental válida para análisis clínico.")}
    </div>
    ${reasons.length ? `
      <ul class="validation-error-reasons">
        ${reasons.map(r => `<li>${escapeUploadHtml(r)}</li>`).join("")}
      </ul>` : ""}
    <div class="validation-error-metrics">
      ${[dental, qual, res, sharp, contr].filter(Boolean)
         .map(s => `<span>${escapeUploadHtml(s)}</span>`).join("")}
    </div>
  `;
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function clearValidationError() {
  const box = document.getElementById("validationErrorBox");
  if (box) box.remove();
}

function escapeUploadHtml(str) {
  return (str ?? "").toString()
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── Poll until completed ─────────────────────────────────────────────────────

function showResultsSection(analysisId) {
  document.getElementById("resultsSection").style.display = "block";
  document.getElementById("analysisStatus").style.display = "flex";
  document.getElementById("resultPanels").style.display = "none";
  document.getElementById("statusText").textContent = "Procesando radiografía...";
  document.getElementById("resultsSection").scrollIntoView({ behavior: "smooth" });
  pollAnalysis(analysisId);
}

async function pollAnalysis(analysisId, attempts = 0) {
  if (attempts > 60) {
    document.getElementById("statusText").textContent = "El análisis está tardando más de lo esperado. Revisa el historial más tarde.";
    const btnT = document.getElementById("btnAnalyze");
    if (btnT) setLoading(btnT, false);
    return;
  }

  try {
    const analysis = await api.getAnalysis(analysisId);

    if (analysis.status === "completed") {
      showResults(analysis);
      loadDashboard();
      loadHistory();
    } else if (analysis.status === "failed") {
      document.getElementById("statusText").textContent =
        "Error en el análisis: " + (analysis.error_message || "Error desconocido");
      const btnF = document.getElementById("btnAnalyze");
      if (btnF) setLoading(btnF, false);
    } else {
      const msgs = ["Aplicando filtros...", "Detectando bordes...",
                    "Extrayendo características...", "Clasificando..."];
      document.getElementById("statusText").textContent = msgs[attempts % msgs.length];
      setTimeout(() => pollAnalysis(analysisId, attempts + 1), 1500);
    }
  } catch (err) {
    setTimeout(() => pollAnalysis(analysisId, attempts + 1), 2000);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// ── Nuevo análisis — resetea todo y vuelve a la zona de carga ────────────────

function resetForNewAnalysis() {
  clearFile();
  clearValidationError();
  document.getElementById("resultsSection").style.display = "none";
  document.getElementById("upload").scrollIntoView({ behavior: "smooth" });
}
