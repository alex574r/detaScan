#!/usr/bin/env bash
# ─── DentaScan — Script de instalación y configuración ───────────────────────
# Uso: bash scripts/setup.sh [--docker | --local | --dev]
# Requiere: Python 3.12+, pip, Node.js (opcional), Docker (si --docker)

set -euo pipefail

# ─── Colores ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}✔${RESET} $*"; }
info() { echo -e "${CYAN}ℹ${RESET} $*"; }
warn() { echo -e "${YELLOW}⚠${RESET} $*"; }
err()  { echo -e "${RED}✘${RESET} $*" >&2; }
die()  { err "$*"; exit 1; }

banner() {
  echo -e "${BLUE}${BOLD}"
  echo "╔═══════════════════════════════════════════════╗"
  echo "║          DentaScan — Setup Automático          ║"
  echo "║   Sistema de Detección de Anomalías Dentales   ║"
  echo "╚═══════════════════════════════════════════════╝"
  echo -e "${RESET}"
}

# ─── Argumentos ───────────────────────────────────────────────────────────────
MODE="${1:---local}"   # --docker | --local | --dev

banner

# ─── Verificar requisitos del sistema ────────────────────────────────────────
info "Verificando requisitos del sistema..."

command -v python3 &>/dev/null || die "Python 3 no encontrado. Instala Python 3.12+"
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)"; then
  ok "Python $PY_VERSION"
else
  die "Se requiere Python 3.10+. Versión actual: $PY_VERSION"
fi

if [[ "$MODE" == "--docker" ]]; then
  command -v docker &>/dev/null      || die "Docker no encontrado."
  command -v docker compose &>/dev/null 2>&1 \
    || docker-compose --version &>/dev/null \
    || die "Docker Compose no encontrado."
  ok "Docker disponible"
fi

# ─── Estructura de directorios ────────────────────────────────────────────────
info "Creando estructura de directorios..."
BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/backend"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p \
  "$PROJECT_DIR/logs" \
  "$BACKEND_DIR/uploads" \
  "$BACKEND_DIR/output" \
  "$BACKEND_DIR/models_ml" \
  "$BACKEND_DIR/logs"
ok "Directorios creados"

# ─── Archivo .env ─────────────────────────────────────────────────────────────
info "Configurando variables de entorno..."
ENV_FILE="$BACKEND_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$BACKEND_DIR/.env.example" "$ENV_FILE"

  # Generar SECRET_KEY aleatoria
  if command -v openssl &>/dev/null; then
    SECRET=$(openssl rand -hex 32)
  else
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  fi

  # Reemplazar en .env
  if [[ "$(uname)" == "Darwin" ]]; then
    sed -i '' "s|cambia_esto_por_una_clave_segura|$SECRET|g" "$ENV_FILE"
  else
    sed -i "s|cambia_esto_por_una_clave_segura|$SECRET|g" "$ENV_FILE"
  fi
  ok ".env creado con SECRET_KEY aleatoria"
else
  warn ".env ya existe, se omite la creación"
fi

# ─── Instalación según modo ───────────────────────────────────────────────────

if [[ "$MODE" == "--docker" ]]; then
  # ── Modo Docker ──────────────────────────────────────────────────────────────
  echo ""
  info "Modo Docker — construyendo y levantando servicios..."

  COMPOSE_FILE="$PROJECT_DIR/config/docker-compose.yml"

  docker compose -f "$COMPOSE_FILE" build --no-cache
  docker compose -f "$COMPOSE_FILE" up -d

  info "Esperando que el backend esté listo..."
  MAX_WAIT=60; WAITED=0
  until curl -sf http://localhost:8000/health &>/dev/null || [[ $WAITED -ge $MAX_WAIT ]]; do
    sleep 2; WAITED=$((WAITED + 2)); printf "."
  done
  echo ""

  if curl -sf http://localhost:8000/health &>/dev/null; then
    ok "Backend disponible en http://localhost:8000"
    ok "Frontend disponible en http://localhost:80"
  else
    warn "El backend tardó más de lo esperado. Revisa: docker compose logs backend"
  fi

else
  # ── Modo Local / Dev ─────────────────────────────────────────────────────────
  info "Modo local — instalando dependencias de Python..."

  VENV_DIR="$BACKEND_DIR/.venv"

  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "Entorno virtual creado en .venv"
  else
    warn "Entorno virtual ya existe, se reutiliza"
  fi

  # Activar venv
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  pip install --upgrade pip --quiet
  pip install -r "$BACKEND_DIR/requirements.txt" --quiet
  ok "Dependencias instaladas"

  # Inicializar base de datos SQLite
  info "Inicializando base de datos..."
  cd "$BACKEND_DIR"
  python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, '.')
from app.database import engine, Base
from app.models import user, analysis  # noqa: importar para registrar modelos
Base.metadata.create_all(bind=engine)
print("  Base de datos inicializada (SQLite)")
PYEOF
  ok "Base de datos lista"

  # Ejecutar seeds
  info "Cargando datos de prueba..."
  python3 database/seeds/seed.py && ok "Seeds cargados" || warn "Los seeds fallaron (puede que ya existan)"

  echo ""
  echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
  echo -e "${GREEN}${BOLD}║       Instalación completada ✔           ║${RESET}"
  echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
  echo ""
  info "Para iniciar el backend:"
  echo -e "  ${CYAN}cd backend && source .venv/bin/activate${RESET}"
  echo -e "  ${CYAN}uvicorn app.main:app --reload --port 8000${RESET}"
  echo ""
  info "Para el frontend:"
  echo -e "  ${CYAN}# Abre frontend/index.html en tu navegador, o usa un servidor local:${RESET}"
  echo -e "  ${CYAN}python3 -m http.server 3000 --directory frontend/${RESET}"
  echo ""
  info "Credenciales de prueba:"
  echo -e "  Admin:       ${YELLOW}admin@dentascan.mx${RESET} / ${YELLOW}Admin1234!${RESET}"
  echo -e "  Odontólogo:  ${YELLOW}odonto@dentascan.mx${RESET} / ${YELLOW}Dentista1234!${RESET}"
  echo -e "  Estudiante:  ${YELLOW}estudiante@dentascan.mx${RESET} / ${YELLOW}Estudiante1234!${RESET}"
  echo ""
  info "Documentación: docs/README_ARQUITECTURA.md"
fi

echo ""
ok "Setup completado."
