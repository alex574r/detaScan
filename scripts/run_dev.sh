#!/usr/bin/env bash
# ─── DentaScan — Iniciar backend en modo desarrollo ───────────────────────────
# Uso: bash scripts/run_dev.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV="$BACKEND_DIR/.venv"

echo "🦷 DentaScan — Servidor de desarrollo"
echo "────────────────────────────────────────"

# Activar entorno virtual si existe
if [[ -d "$VENV" ]]; then
  source "$VENV/bin/activate"
  echo "✔ Entorno virtual activado"
else
  echo "⚠ No se encontró .venv. Ejecuta primero: bash scripts/setup.sh"
  exit 1
fi

# Cargar variables de entorno
ENV_FILE="$BACKEND_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
  echo "✔ Variables de entorno cargadas"
fi

# Cambiar al directorio del backend
cd "$BACKEND_DIR"

# Iniciar servidor con recarga automática
echo ""
echo "Iniciando en http://localhost:8000"
echo "Swagger UI:  http://localhost:8000/docs"
echo "Presiona Ctrl+C para detener"
echo ""

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info
