# 🦷 DentaScan

**Sistema de detección de caries y anomalías dentales por visión artificial**

> DentaScan analiza radiografías dentales (DICOM, PNG, JPEG, TIFF) y clasifica automáticamente el estado del diente usando un pipeline de procesamiento de imagen y modelos de machine learning (Random Forest + SVM).

Desarrollado por estudiantes de Ingeniería de Software — UAEM Tianguistenco  
Materia: Técnicas de Minería de Patrones de Imagen

---

## Índice

- [Características](#características)
- [Instalación rápida](#instalación-rápida)
- [Instalación con Docker](#instalación-con-docker)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Credenciales de prueba](#credenciales-de-prueba)
- [Uso de la API](#uso-de-la-api)
- [Entrenamiento del modelo](#entrenamiento-del-modelo)
- [Documentación técnica](#documentación-técnica)
- [Equipo](#equipo)

---

## Características

- **Análisis de radiografías** en formatos DICOM, PNG, JPEG y TIFF
- **Pipeline de visión artificial**: suavizado Gaussiano + Mediana → CLAHE → Canny/Sobel → segmentación por umbral Otsu → extracción de 12 features radiométricas
- **Clasificador ML**: Random Forest y SVM (con fallback heurístico si no hay modelos entrenados)
- **5 clases diagnósticas**: Diente Sano, Caries Incipiente, Caries Avanzada, Absceso Periapical, Lesión Ósea
- **API REST** con FastAPI: autenticación JWT, análisis en background con polling, historial por usuario
- **Frontend SPA** con autenticación, drag & drop, visualización de resultados e historial
- **3 roles de usuario**: Administrador, Odontólogo, Estudiante
- **Modo demo** activo por defecto (reglas heurísticas); el clasificador real se activa entrenando con tu dataset

---

## Instalación rápida

**Requisitos**: Python 3.12+, pip

```bash
# 1. Clonar o descomprimir el proyecto
cd dentascan

# 2. Configurar y levantar
make setup          # crea .venv, instala dependencias, BD, seeds
make run            # inicia backend en http://localhost:8000

# 3. Servir el frontend (otra terminal)
python3 -m http.server 3000 --directory frontend/
# Abrir http://localhost:3000
```

O con el script directo:

```bash
bash scripts/setup.sh --local
bash scripts/run_dev.sh
```

---

## Instalación con Docker

**Requisitos**: Docker + Docker Compose

```bash
# Levantar backend + frontend + PostgreSQL
make docker-up

# Ver logs
make docker-logs

# Detener
make docker-down
```

Servicios expuestos:
- Frontend: http://localhost:80
- Backend API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

---

## Estructura del proyecto

```
dentascan/
├── backend/               # API FastAPI + lógica de negocio
│   ├── app/
│   │   ├── api/           # Controladores REST (auth, analyses)
│   │   ├── core/          # Pipeline de imagen (loader, preprocessor, segmentor…)
│   │   ├── models/        # Modelos SQLAlchemy (User, Analysis)
│   │   ├── schemas/       # Schemas Pydantic (DTOs)
│   │   ├── services/      # Lógica de negocio (image_service, auth_service)
│   │   ├── utils/         # Security (JWT/bcrypt), Logger
│   │   ├── exceptions/    # Excepciones de dominio tipadas
│   │   ├── config.py      # Settings por entorno (pydantic-settings)
│   │   ├── database.py    # SQLAlchemy session factory
│   │   └── main.py        # App factory FastAPI
│   ├── database/
│   │   ├── migrations/    # SQL de creación de tablas
│   │   └── seeds/         # Usuarios demo
│   ├── scripts/           # train_model.py
│   ├── tests/             # 15 tests pytest del pipeline
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
│
├── frontend/              # SPA HTML/CSS/JS vanilla
│   ├── index.html
│   ├── css/styles.css
│   ├── js/
│   │   ├── app.js         # Entry point, navegación, estado
│   │   ├── api.js         # Cliente REST centralizado
│   │   ├── auth.js        # Login/registro/logout
│   │   ├── upload.js      # Drag & drop + polling
│   │   └── results.js     # Renderizado de resultados e historial
│   └── Dockerfile
│
├── config/
│   ├── docker-compose.yml
│   └── nginx.conf
│
├── docs/                  # Documentación técnica
│   ├── README_ARQUITECTURA.md
│   ├── API_REFERENCE.md
│   └── DEPLOYMENT.md
│
├── scripts/
│   ├── setup.sh           # Instalación automatizada
│   └── run_dev.sh         # Arranque rápido en desarrollo
│
├── logs/                  # Logs de la aplicación (generados en runtime)
├── assets/                # Recursos estáticos
├── Makefile
└── README.md
```

---

## Credenciales de prueba

| Rol          | Email                     | Contraseña       |
|--------------|---------------------------|------------------|
| Admin        | admin@dentascan.mx        | Admin1234!       |
| Odontólogo   | odonto@dentascan.mx       | Dentista1234!    |
| Estudiante   | estudiante@dentascan.mx   | Estudiante1234!  |

---

## Uso de la API

```bash
# Autenticarse
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@dentascan.mx","password":"Admin1234!"}'

# Subir una radiografía para análisis
curl -X POST http://localhost:8000/analyses/ \
  -H "Authorization: Bearer <TOKEN>" \
  -F "file=@radiografia.png"

# Consultar resultado (polling)
curl http://localhost:8000/analyses/<ID> \
  -H "Authorization: Bearer <TOKEN>"
```

Documentación interactiva completa: http://localhost:8000/docs

---

## Entrenamiento del modelo

Por defecto el sistema opera en **modo demo** (clasificación por reglas heurísticas).  
Para usar los modelos reales (RF + SVM entrenados con tu dataset):

```bash
# Dataset esperado en backend/input/balanceado/
# Estructura: una carpeta por clase con imágenes dentro
make train

# O directamente:
cd backend
source .venv/bin/activate
python scripts/train_model.py --dataset_dir input/balanceado
```

Los modelos se guardan en `backend/models_ml/` y se cargan automáticamente al reiniciar.

---

## Documentación técnica

| Documento                        | Contenido                                   |
|----------------------------------|---------------------------------------------|
| `docs/README_ARQUITECTURA.md`    | Arquitectura, módulos, flujo del sistema    |
| `docs/API_REFERENCE.md`          | Endpoints, schemas, ejemplos de respuesta   |
| `docs/DEPLOYMENT.md`             | Despliegue en producción, Docker, variables |

---

## Notas importantes

> ⚠️ **DentaScan es una herramienta de apoyo diagnóstico.** Los resultados generados **no sustituyen el criterio clínico** de un profesional odontológico certificado. Toda interpretación diagnóstica debe ser validada por un especialista.

- El pipeline de imagen se implementa con OpenCV 4.9 sobre imágenes en escala de grises
- El clasificador usa 12 features radiométricas definidas en la documentación del proyecto
- La precisión del clasificador depende directamente de la calidad y tamaño del dataset de entrenamiento

---

## Equipo

- Francisco Javier Martínez Peña
- Alejandro Hernández Maya  
- Nadia Montserrat Ortiz Nuñez

**Asesora**: Rocio Elizabeth Pulido Alba  
Universidad Autónoma del Estado de México — Unidad Académica Tianguistenco  
Ingeniería de Software
