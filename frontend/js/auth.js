/**
 * DentaScan — Autenticación + boot loader + manejo de sesión
 *
 * Flujo:
 *  1. DOMContentLoaded → boot loader visible, auth/main ocultos
 *  2. tryAutoLogin():
 *      - si hay token válido en localStorage, llama /auth/me
 *      - éxito → onLoginSuccess()
 *      - fallo → muestra pantalla de login
 *  3. login/register → setToken + onLoginSuccess
 *  4. evento auth:expired (disparado por api.js en 401) → logout silencioso
 */

let currentUser = null;

// ── Tabs login/registro ─────────────────────────────────────────────────────
function switchTab(tab) {
  const isLogin = tab === "login";
  document.getElementById("loginForm").style.display    = isLogin ? "flex" : "none";
  document.getElementById("registerForm").style.display = isLogin ? "none" : "flex";
  document.querySelectorAll(".tab-btn").forEach((btn, i) => {
    btn.classList.toggle("active", isLogin ? i === 0 : i === 1);
  });
  clearErrors();
}

// ── Handlers de formulario ──────────────────────────────────────────────────
async function handleLogin(e) {
  e.preventDefault();
  const email    = document.getElementById("loginEmail").value;
  const password = document.getElementById("loginPassword").value;
  const btn      = document.getElementById("loginBtn");

  setLoading(btn, true);
  clearErrors();
  try {
    const data = await api.login(email, password);
    api.setToken(data.access_token);
    currentUser = data.user;
    onLoginSuccess();
  } catch (err) {
    showError("loginError", err.message);
  } finally {
    setLoading(btn, false);
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const full_name = document.getElementById("regName").value;
  const email     = document.getElementById("regEmail").value;
  const password  = document.getElementById("regPassword").value;
  const role      = document.getElementById("regRole").value;
  const btn       = document.getElementById("registerBtn");

  setLoading(btn, true);
  clearErrors();
  try {
    const data = await api.register(email, full_name, password, role);
    api.setToken(data.access_token);
    currentUser = data.user;
    onLoginSuccess();
  } catch (err) {
    showError("registerError", err.message);
  } finally {
    setLoading(btn, false);
  }
}

// ── Estados de UI ───────────────────────────────────────────────────────────
function showBootLoader(visible) {
  let boot = document.getElementById("bootLoader");
  if (!boot && visible) {
    boot = document.createElement("div");
    boot.id = "bootLoader";
    boot.className = "boot-loader";
    boot.innerHTML = `
      <div class="boot-spinner"></div>
      <p>Validando sesión…</p>
    `;
    document.body.appendChild(boot);
  }
  if (boot) boot.style.display = visible ? "flex" : "none";
  // Solo ocultamos el authSection durante el boot. Al terminar, NO lo tocamos —
  // cada handler (onLoginSuccess / showAuthScreen) decide qué mostrar.
  if (visible) {
    document.getElementById("authSection").style.display = "none";
  }
}

function onLoginSuccess() {
  document.getElementById("authSection").style.display = "none";
  document.getElementById("appMain").style.display     = "block";
  document.getElementById("navUser").textContent       = currentUser.full_name || currentUser.email;
  document.getElementById("btnLogout").style.display   = "inline-flex";
  document.getElementById("nav-upload").style.display  = "inline";
  document.getElementById("nav-history").style.display = "inline";
  showBootLoader(false);
  if (typeof loadDashboard === "function") loadDashboard();
  if (typeof loadHistory   === "function") loadHistory();
}

function showAuthScreen() {
  document.getElementById("authSection").style.display = "flex";
  document.getElementById("appMain").style.display     = "none";
  document.getElementById("btnLogout").style.display   = "none";
  document.getElementById("navUser").textContent       = "";
  document.getElementById("nav-upload").style.display  = "none";
  document.getElementById("nav-history").style.display = "none";
  const rs = document.getElementById("resultsSection");
  if (rs) rs.style.display = "none";
  showBootLoader(false);
}

function logout() {
  api.setToken(null);
  currentUser = null;
  showAuthScreen();
}

// ── Restauración de sesión ──────────────────────────────────────────────────
async function tryAutoLogin() {
  const token = api.loadToken();
  if (!token) {
    showAuthScreen();
    return;
  }

  showBootLoader(true);
  try {
    const user = await api.getMe();
    currentUser = user;
    onLoginSuccess();
  } catch (err) {
    console.warn("[auth] sesión inválida:", err.message);
    api.setToken(null);
    showAuthScreen();
  }
}

// ── Listener: sesión expiró durante uso ─────────────────────────────────────
window.addEventListener("auth:expired", () => {
  if (currentUser) {
    currentUser = null;
    showAuthScreen();
    showError("loginError", "Tu sesión expiró. Inicia sesión nuevamente.");
  }
});

// ── Arranque automático ─────────────────────────────────────────────────────
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", tryAutoLogin);
} else {
  tryAutoLogin();
}

// ── Helpers UI ──────────────────────────────────────────────────────────────
function setLoading(btn, loading) {
  const txt = btn.querySelector(".btn-text");
  const spn = btn.querySelector(".btn-spinner");
  if (txt) txt.style.display = loading ? "none" : "inline";
  if (spn) spn.style.display = loading ? "inline-block" : "none";
  btn.disabled = loading;
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.classList.add("visible"); }
}

function clearErrors() {
  document.querySelectorAll(".form-error").forEach(el => {
    el.textContent = ""; el.classList.remove("visible");
  });
}
