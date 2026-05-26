# ─── DentaScan — Makefile ─────────────────────────────────────────────────────
# Uso: make <comando>
# Requiere: Python 3.12+, bash. Docker para comandos docker-*

.PHONY: help setup run frontend test lint clean docker-up docker-down docker-logs seed train

BACKEND_DIR := backend
VENV        := $(BACKEND_DIR)/.venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
UVICORN     := $(VENV)/bin/uvicorn
PYTEST      := $(VENV)/bin/pytest

help:
	@echo ""
	@echo "DentaScan — Comandos disponibles"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make setup        Instalar dependencias y configurar el proyecto"
	@echo "  make run          Iniciar backend en modo desarrollo (hot-reload)
  make frontend     Iniciar servidor de frontend con caché deshabilitada (puerto 3000)"
	@echo "  make test         Ejecutar suite de pruebas pytest"
	@echo "  make lint         Verificar estilo de código (ruff)"
	@echo "  make seed         Cargar datos de prueba en la base de datos"
	@echo "  make train        Entrenar modelos ML (requiere dataset)"
	@echo "  make clean        Eliminar artefactos generados"
	@echo "  make docker-up    Levantar todos los servicios con Docker Compose"
	@echo "  make docker-down  Detener y eliminar contenedores"
	@echo "  make docker-logs  Ver logs de los contenedores"
	@echo ""

# ─── Instalación ──────────────────────────────────────────────────────────────
setup:
	@bash scripts/setup.sh --local

# ─── Desarrollo ───────────────────────────────────────────────────────────────
frontend:
	@python3 frontend/server.py

run:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info

# ─── Pruebas ──────────────────────────────────────────────────────────────────
test:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  pytest tests/ -v --tb=short --color=yes

test-cov:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# ─── Calidad de código ────────────────────────────────────────────────────────
lint:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  ruff check app/ && echo "✔ Sin errores de estilo"

format:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  ruff format app/

# ─── Base de datos ────────────────────────────────────────────────────────────
seed:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  python database/seeds/seed.py

db-init:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  python -c "from app.database import engine, Base; from app.models import user, analysis; Base.metadata.create_all(bind=engine); print('Base de datos inicializada')"

# ─── Machine Learning ─────────────────────────────────────────────────────────
train:
	@cd $(BACKEND_DIR) && \
	  source .venv/bin/activate && \
	  python scripts/train_model.py --dataset_dir input/balanceado

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	@docker compose -f config/docker-compose.yml up --build -d
	@echo "✔ Servicios levantados. Frontend: http://localhost | Backend: http://localhost:8000"

docker-down:
	@docker compose -f config/docker-compose.yml down

docker-logs:
	@docker compose -f config/docker-compose.yml logs -f --tail=100

docker-clean:
	@docker compose -f config/docker-compose.yml down -v --remove-orphans

# ─── Limpieza ─────────────────────────────────────────────────────────────────
clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name ".coverage" -delete 2>/dev/null || true
	@echo "✔ Artefactos eliminados"
