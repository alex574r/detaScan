/**
 * DentaScan — Cliente API
 * Centraliza llamadas al backend FastAPI con manejo automático de sesión:
 *   - Decode JWT en cliente (para conocer expiración sin pegarle al server)
 *   - Auto-refresh proactivo cuando quedan < REFRESH_THRESHOLD_MS de vida
 *   - Detección 401 → dispara evento auth:expired
 */

const API_BASE    = "http://localhost:8000/api/v1";
const OUTPUT_BASE = "http://localhost:8000/output";

// Renovar token si quedan menos de 30 min para expirar
const REFRESH_THRESHOLD_MS = 30 * 60 * 1000;

const api = {
  _token: null,
  _tokenExp: 0,           // epoch ms
  _refreshPromise: null,  // de-duplica refresh concurrentes

  // ── Token storage ───────────────────────────────────────────────────────────
  setToken(token) {
    this._token = token;
    if (token) {
      this._tokenExp = api._decodeExp(token);
      localStorage.setItem("ds_token", token);
    } else {
      this._tokenExp = 0;
      localStorage.removeItem("ds_token");
    }
  },

  loadToken() {
    const t = localStorage.getItem("ds_token");
    if (!t) { this._token = null; this._tokenExp = 0; return null; }
    this._token    = t;
    this._tokenExp = api._decodeExp(t);
    // Si ya expiró, descartar inmediatamente
    if (this._tokenExp && this._tokenExp < Date.now()) {
      api.setToken(null);
      return null;
    }
    return t;
  },

  hasValidToken() {
    return !!this._token && this._tokenExp > Date.now();
  },

  _decodeExp(token) {
    try {
      const payload = JSON.parse(atob(token.split(".")[1]));
      return (payload.exp || 0) * 1000;
    } catch { return 0; }
  },

  // ── Refresh proactivo ───────────────────────────────────────────────────────
  async _maybeRefresh() {
    if (!this._token) return;
    const remaining = this._tokenExp - Date.now();
    if (remaining > REFRESH_THRESHOLD_MS) return;        // aún hay vida
    if (remaining <= 0) { this.setToken(null); return; } // expirado, no se puede refrescar
    if (this._refreshPromise) return this._refreshPromise;

    this._refreshPromise = (async () => {
      try {
        const resp = await fetch(`${API_BASE}/auth/refresh`, {
          method: "POST",
          headers: { "Authorization": `Bearer ${this._token}` },
        });
        if (!resp.ok) throw new Error("refresh failed");
        const data = await resp.json();
        this.setToken(data.access_token);
      } catch (err) {
        console.warn("[api] refresh falló:", err.message);
      } finally {
        this._refreshPromise = null;
      }
    })();
    return this._refreshPromise;
  },

  // ── Petición base ───────────────────────────────────────────────────────────
  async _request(method, path, body = null, isFormData = false) {
    await this._maybeRefresh();

    const opts = { method, headers: {} };
    if (this._token) opts.headers["Authorization"] = `Bearer ${this._token}`;

    if (body && !isFormData) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    } else if (isFormData) {
      opts.body = body;
    }

    let resp;
    try {
      resp = await fetch(`${API_BASE}${path}`, opts);
    } catch (netErr) {
      throw new Error(`Sin conexión con el servidor: ${netErr.message}`);
    }

    // 401 → sesión inválida; limpiar y notificar
    if (resp.status === 401) {
      const wasAuthed = !!this._token;
      api.setToken(null);
      if (wasAuthed) {
        window.dispatchEvent(new CustomEvent("auth:expired"));
      }
      const err = new Error("Sesión expirada o inválida");
      err.status = 401;
      throw err;
    }

    const data = await resp.json().catch(() => ({ detail: resp.statusText }));

    if (!resp.ok) {
      const detail = data?.detail;
      // Garantizamos que msg SIEMPRE sea string para evitar "[object Object]"
      let msg = `Error ${resp.status}`;
      if (typeof detail === "string") {
        msg = detail;
      } else if (Array.isArray(detail)) {
        // FastAPI 422 con validación de schemas: lista de {loc, msg, type}
        msg = detail
          .map(e => {
            if (typeof e === "string") return e;
            if (e && typeof e.msg === "string") return e.msg;
            try { return JSON.stringify(e); } catch { return String(e); }
          })
          .join("; ");
      } else if (detail && typeof detail === "object") {
        // Nuestros endpoints devuelven {message, reasons, ...}
        if (typeof detail.message === "string") {
          msg = detail.message;
        } else if (Array.isArray(detail.reasons) && detail.reasons.length) {
          msg = detail.reasons.join("; ");
        } else {
          try { msg = JSON.stringify(detail); } catch { msg = `Error ${resp.status}`; }
        }
      }
      const err = new Error(String(msg));
      err.status = resp.status;
      err.detail = detail;
      throw err;
    }

    return data;
  },

  // ── Auth ────────────────────────────────────────────────────────────────────
  register: (email, full_name, password, role) =>
    api._request("POST", "/auth/register", { email, full_name, password, role }),

  login: (email, password) =>
    api._request("POST", "/auth/login", { email, password }),

  getMe: () => api._request("GET", "/auth/me"),

  refresh: () => api._request("POST", "/auth/refresh"),

  // ── Analyses ────────────────────────────────────────────────────────────────
  uploadAnalysis(file, xrayType, model, options = {}) {
    const form = new FormData();
    form.append("file", file);
    form.append("xray_type", xrayType);
    form.append("model", model);
    if (options.analysis_mode)    form.append("analysis_mode", options.analysis_mode);
    if (options.min_confidence != null) form.append("min_confidence", String(options.min_confidence));
    if (options.lesion_types && options.lesion_types.length)
      form.append("lesion_types", options.lesion_types.join(","));
    if (options.sensitivity)      form.append("sensitivity", options.sensitivity);
    if (options.use_yolo)         form.append("use_yolo", "true");
    return api._request("POST", "/analyses/", form, true);
  },

  getAnalysis: (id) => api._request("GET", `/analyses/${id}`),

  listAnalyses: (skip = 0, limit = 20) =>
    api._request("GET", `/analyses/?skip=${skip}&limit=${limit}`),

  getSummary: () => api._request("GET", "/analyses/summary"),

  deleteAnalysis: (id) => api._request("DELETE", `/analyses/${id}`),

  deleteAllAnalyses: () => api._request("DELETE", "/analyses/"),

  health: () => api._request("GET", "/health").catch(() => ({ status: "error" })),

  // ── URL helpers ─────────────────────────────────────────────────────────────
  outputUrl(path) {
    if (!path) return null;
    const filename = path.split(/[/\\]/).pop();
    return `${OUTPUT_BASE}/${filename}`;
  },
};
