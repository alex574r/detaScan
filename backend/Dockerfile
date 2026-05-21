# ─── DentaScan Backend — Dockerfile ──────────────────────────────────────────
# Python 3.12 slim; multi-stage para imagen más pequeña en producción

# ── Etapa 1: dependencias ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Dependencias del sistema necesarias para OpenCV y librerías científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Etapa 2: imagen final ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Librerías de runtime para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar paquetes instalados
COPY --from=builder /install /usr/local

WORKDIR /app

# Crear directorios de trabajo
RUN mkdir -p uploads output models_ml logs

# Copiar código de la aplicación
COPY app/ ./app/
COPY database/ ./database/
COPY scripts/ ./scripts/

# Usuario no-root para seguridad
RUN useradd -r -s /bin/false dentascan && \
    chown -R dentascan:dentascan /app
USER dentascan

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Iniciar con uvicorn (workers según CPU disponibles)
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
