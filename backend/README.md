# DentaScan — Backend

API REST construida con FastAPI. Gestiona autenticación, recibe imágenes radiográficas, ejecuta el pipeline de visión artificial y persiste los resultados.

---

## Arranque rápido

```bash
# Desde la carpeta backend/
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # editar SECRET_KEY al menos
uvicorn app.main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

---

## Estructura

```
backend/
├── app/
│   ├── api/             # Endpoints (auth.py, analysis.py)
│   ├── core/            # Pipeline de imagen
│   │   ├── loader.py            # Carga DICOM/PNG/JPEG/TIFF
│   │   ├── preprocessor.py      # Gaussiano + Mediana + CLAHE
│   │   ├── segmentor.py         # Otsu + Canny + Sobel
│   │   ├── feature_extractor.py # 12 features radiométricas
│   │   ├── classifier.py        # SVM + Random Forest
│   │   └── visualizer.py        # Imagen anotada + histograma
│   ├── models/          # Modelos SQLAlchemy (User, Analysis)
│   ├── schemas/         # Schemas Pydantic (DTOs)
│   ├── services/        # Lógica de negocio (image_service, auth_service)
│   ├── utils/           # security.py (JWT/bcrypt), logger.py
│   ├── exceptions/      # Excepciones tipadas de dominio
│   ├── config.py        # Settings por entorno
│   ├── database.py      # SQLAlchemy session factory
│   └── main.py          # App factory + middleware + rutas
├── database/
│   ├── migrations/init.sql
│   └── seeds/seed.py
├── scripts/train_model.py
├── tests/
│   ├── conftest.py
│   └── test_pipeline.py   # 15 tests del pipeline
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## Pipeline de imagen

```
Imagen → loader → preprocessor → segmentor → feature_extractor → classifier → visualizer → Resultado
```

Cada módulo en `core/` es independiente y recibe/devuelve tipos bien definidos. El orquestador es `services/image_service.py`.

---

## Ejecutar pruebas

```bash
pytest tests/ -v
```

15 tests cubren carga de imagen, cada etapa del pipeline, extracción de features y el clasificador en modo demo.

---

## Entrenar el clasificador

```bash
# Dataset esperado: una subcarpeta por clase, imágenes dentro
python scripts/train_model.py --dataset_dir input/balanceado

# Los modelos .pkl quedan en models_ml/
# Reiniciar el servidor para que los cargue automáticamente
```

Si no hay modelos en `models_ml/`, el clasificador corre en **modo demo** (reglas heurísticas sobre las features). Funcional para demos, no recomendado para uso clínico real.

---

## Base de datos

SQLite por defecto (archivo `dentascan.db`). Para PostgreSQL:

```bash
# .env
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/dentascan
```

El esquema se crea automáticamente al arrancar. Los seeds generan 3 usuarios de prueba:

| Email                    | Contraseña      | Rol          |
|--------------------------|-----------------|--------------|
| admin@dentascan.mx       | Admin1234!      | admin        |
| odonto@dentascan.mx      | Dentista1234!   | odontologist |
| estudiante@dentascan.mx  | Estudiante1234! | student      |

---

## Dependencias principales

| Paquete          | Versión  | Uso                                |
|------------------|----------|------------------------------------|
| fastapi          | ≥0.111   | Framework web                      |
| uvicorn          | ≥0.29    | Servidor ASGI                      |
| sqlalchemy       | ≥2.0     | ORM                                |
| pydantic-settings| ≥2.2     | Configuración por entorno          |
| python-jose      | ≥3.3     | JWT                                |
| passlib[bcrypt]  | ≥1.7     | Hash de contraseñas                |
| opencv-python    | ≥4.9     | Procesamiento de imagen            |
| numpy            | ≥1.26    | Operaciones matriciales            |
| scikit-learn     | ≥1.4     | SVM + Random Forest                |
| pydicom          | ≥2.4     | Lectura de archivos DICOM          |
| matplotlib       | ≥3.8     | Generación de histogramas          |
| pillow           | ≥10.3    | Fallback para carga de imágenes    |
