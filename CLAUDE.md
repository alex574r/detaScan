# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DentaScan is a full-stack web application for dental anomaly detection using computer vision and ML. It analyzes dental radiographs (X-rays) in DICOM/PNG/JPEG/TIFF formats and produces annotated diagnostic output.

## Common Commands

All primary workflows run through `make` from the project root:

```bash
make setup        # Create venv, install deps, init DB, seed demo users
make run          # Start backend on http://localhost:8000 (hot-reload)
make test         # Run pytest suite (15 pipeline tests)
make test-cov     # Run tests with coverage report
make lint         # Check style with ruff
make format       # Auto-format with ruff
make train        # Train RF and SVM models on dataset
make seed         # Reload demo users into DB
make docker-up    # Launch all services (backend + frontend + PostgreSQL)
make docker-logs  # Tail container logs
```

Frontend dev server (separate terminal):
```bash
python3 -m http.server 3000 --directory frontend/
```

Run a single test:
```bash
cd backend && python -m pytest tests/test_pipeline.py::TestPreprocessor::test_gaussian -v
```

## Architecture

### Stack

- **Backend:** Python 3.12+, FastAPI, SQLAlchemy ORM, SQLite (dev) / PostgreSQL (prod)
- **ML/CV:** OpenCV 4.10+, scikit-learn (Random Forest + SVM), pydicom, joblib
- **Frontend:** Vanilla HTML/CSS/JS SPA — no framework
- **DevOps:** Docker Compose (Nginx + FastAPI + PostgreSQL)

### Request Flow

```
Browser (frontend/) → HTTP REST → FastAPI (backend/app/) → BackgroundTask → Pipeline → DB + Filesystem
```

1. User authenticates → JWT stored in localStorage
2. Upload via `POST /analyses/` (multipart) → file saved to `backend/input/` with UUID name
3. Analysis record created with `status=PENDING`; pipeline runs in `BackgroundTasks`
4. Pipeline: `loader → preprocessor → segmentor → feature_extractor → classifier → visualizer`
5. Results written to DB; annotated images saved to `backend/output/`
6. Frontend polls `GET /analyses/{id}` every 1.5s until status is `COMPLETED`

### Key Directories

```
backend/
  app/
    main.py              # FastAPI app factory, router registration, static file mounts
    config.py            # Pydantic Settings with @lru_cache; controls paths, CLAHE params, model paths
    routers/             # auth.py, analyses.py — thin HTTP layer
    services/            # auth_service.py, image_processing_service.py — business logic
    core/                # Pipeline modules (loader, preprocessor, segmentor, feature_extractor,
    │                    #   classifier, visualizer) — each independently testable
    models/              # SQLAlchemy ORM models (User, Analysis)
    schemas/             # Pydantic request/response schemas
  database/
    seeds/seed.py        # Demo users (admin/odontólogo/student roles)
  tests/                 # pytest suite covering pipeline stages
  models_ml/             # Trained .pkl files (RF + SVM); absent → demo heuristics used
  input/ output/         # Upload and result image storage
frontend/
  index.html             # Entry point — tabbed auth UI + upload/history sections
  js/
    app.js               # AppState, navigation, module initialization
    api.js               # Centralized HTTP client — injects JWT on all calls
    auth.js upload.js results.js  # Feature modules
config/
  docker-compose.yml     # Three-service orchestration
Makefile                 # All dev/test/deploy targets
```

### Image Processing Pipeline

Each stage is a class in `backend/app/core/`:

| Module | Responsibility |
|---|---|
| `loader.py` | Reads DICOM/PNG/JPEG/TIFF → uint8 grayscale numpy array |
| `preprocessor.py` | Gaussian blur + Median filter + CLAHE adaptive histogram equalization |
| `segmentor.py` | Otsu threshold, Canny edges, Sobel gradients, radiolucency detection |
| `feature_extractor.py` | Extracts fixed 12-feature vector (radiometric stats) |
| `classifier.py` | RF (default) or SVM inference; falls back to demo heuristics if no `.pkl` files |
| `visualizer.py` | Generates annotated output images and result summaries |

### Patterns

- **Dependency Injection:** FastAPI `Depends()` for DB sessions, auth, logging throughout routers
- **Singletons:** `DentalClassifier` and core pipeline objects are module-level instances loaded once at startup
- **Configuration:** All paths/params flow from `config.py` via `get_settings()` — never hardcoded
- **Enums:** `UserRole`, `AnalysisStatus`, `XRayType` for all domain state
- **DB JSON columns:** Feature vectors and class probabilities stored as JSON for schema flexibility
- **Spanish naming:** Class names, DB tables, comments often in Spanish (academic project context)

### Auth

- JWT (HS256, 8-hour expiry), bcrypt passwords (12 rounds)
- Configure via `backend/.env` (copy from `.env.example`): `SECRET_KEY`, `DATABASE_URL`, `ALGORITHM`
- Demo credentials after `make seed`: admin/admin123, odonto/odonto123, student/student123

### Testing

Tests use pytest with synthetic test images generated in `conftest.py` — no real radiographs needed. The 15 tests cover each pipeline stage independently.
