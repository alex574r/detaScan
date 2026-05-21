# DentaScan — Arquitectura del Sistema

## Visión general

DentaScan sigue una arquitectura en capas desacopladas: el frontend (SPA HTML/JS) se comunica con el backend (FastAPI) a través de una API REST. El backend orquesta el pipeline de imagen, persiste resultados en SQLite/PostgreSQL y sirve las imágenes de salida como archivos estáticos.

```
[Navegador / SPA]
        │ HTTP (JSON + multipart)
        ▼
[FastAPI — Backend]
   ├── Auth layer (JWT)
   ├── API controllers
   ├── Services layer
   ├── Image pipeline (core/)
   └── Database (SQLAlchemy)
        │
        ├── SQLite (dev) / PostgreSQL (prod)
        └── Filesystem (uploads/, output/, models_ml/)
```

---

## Módulos del backend

### `app/config.py`
Carga configuración desde variables de entorno usando `pydantic-settings`. Un único objeto `Settings` se instancia al arrancar y se importa donde se necesite. Soporta `.env` para desarrollo y variables del sistema en producción.

### `app/database.py`
Factory de sesiones SQLAlchemy. Exporta `engine`, `SessionLocal` y `Base`. La dependencia `get_db()` maneja el ciclo de vida de la sesión en cada request.

### `app/models/`
- **`user.py`**: tabla `users` con columnas id, email, password_hash, nombre, rol (admin / odontologist / student), activo, timestamps.
- **`analysis.py`**: tabla `analyses` con FK a users, metadata de la imagen, features JSON, resultados del clasificador (clase predicha, probabilidades, confianza), referencias a imágenes de salida, estado del pipeline.

### `app/schemas/`
Schemas Pydantic para validación de entrada/salida. Separación clara entre schemas de creación, respuesta y autenticación.

### `app/api/`
Controladores REST:
- **`auth.py`**: `POST /auth/register`, `POST /auth/login`, `GET /auth/me`
- **`analysis.py`**: `POST /analyses/` (upload + análisis en background), `GET /analyses/` (historial), `GET /analyses/{id}` (polling), `DELETE /analyses/{id}`

### `app/services/`
- **`auth_service.py`**: registro, login, búsqueda de usuario. Desacoplado de los endpoints.
- **`image_service.py`**: orquestador del pipeline completo. Recibe la ruta del archivo y devuelve el resultado del análisis. Llama a cada módulo de `core/` en orden.

### `app/core/` — Pipeline de imagen

El pipeline se ejecuta secuencialmente. Cada módulo es independiente.

```
loader → preprocessor → segmentor → feature_extractor → classifier → visualizer
```

| Módulo               | Función                                                                |
|----------------------|------------------------------------------------------------------------|
| `loader.py`          | Lee DICOM, PNG, JPEG, TIFF. Normaliza a array uint8 en escala de grises. |
| `preprocessor.py`    | Suavizado Gaussiano, filtro Mediana, CLAHE (ecualización adaptativa).  |
| `segmentor.py`       | Umbral Otsu, detección de bordes Canny + gradientes Sobel, cálculo de radiolucidez. |
| `feature_extractor.py` | Extrae 12 features: media, std, min_px, max_px, bordes_mean, sobel_mean, zonas TL/TR/BL/BR, prop_oscuros, asimetría. |
| `classifier.py`      | Carga modelos RF/SVM desde `models_ml/`. Opera en modo demo (heurístico) si no hay modelos. |
| `visualizer.py`      | Genera imagen de salida con bounding boxes, etiqueta de diagnóstico, histograma. |

### `app/utils/`
- **`security.py`**: generación y verificación de JWT, hash/verify de contraseñas con bcrypt. Dependencia `get_current_user` para proteger endpoints.
- **`logger.py`**: RotatingFileHandler configurado por entorno. Los logs van a `logs/dentascan.log`.

### `app/exceptions/`
Excepciones tipadas de dominio: `ImageLoadError`, `PipelineError`, `AuthenticationError`, etc. Se mapean a códigos HTTP en los manejadores de `main.py`.

---

## Pipeline completo — flujo paso a paso

```
1. Usuario sube imagen (POST /analyses/)
   └─► Se guarda en uploads/
   └─► Se crea registro Analysis con estado "pendiente"
   └─► Se lanza BackgroundTask

2. BackgroundTask ejecuta image_service.run_pipeline()
   ├─► loader.load()          → array numpy uint8
   ├─► preprocessor.process() → imagen suavizada + CLAHE
   ├─► segmentor.segment()    → máscaras + detección de bordes
   ├─► feature_extractor.extract() → vector de 12 features
   ├─► classifier.predict()   → clase + probabilidades
   └─► visualizer.generate()  → imagen PNG anotada en output/

3. Se actualiza Analysis en BD: estado "completado" + resultados

4. Frontend hace polling GET /analyses/{id} cada 1.5s
   └─► Cuando estado == "completado", renderiza resultados
```

---

## Frontend — estructura de módulos

```
app.js       ← entry point: init, auth check, navegación global, toasts
  ├── auth.js     ← login, register, logout, checkAuth (JWT en localStorage)
  ├── upload.js   ← drag & drop, validación de archivo, upload + polling
  ├── results.js  ← render de resultados, historial, modal de detalle
  └── api.js      ← cliente fetch centralizado (base URL, headers, errores)
```

La SPA tiene 4 vistas principales:
- `auth-view`: login / registro
- `dashboard-view`: formulario de upload + estadísticas rápidas
- `results-view`: resultado del análisis más reciente
- `history-view`: historial paginado del usuario

La navegación es puramente DOM (sin router externo). El estado de autenticación se mantiene en `AppState` dentro de `app.js` y en `localStorage` (solo el token JWT).

---

## Base de datos

### SQLite (desarrollo)
Archivo `backend/dentascan.db`. Generado automáticamente al arrancar.

### PostgreSQL (producción)
Configurar `DATABASE_URL` en `.env`:
```
DATABASE_URL=postgresql://user:password@host:5432/dentascan
```

### Esquema resumido

```sql
users (
  id UUID PK, email UNIQUE, password_hash, nombre,
  rol ENUM('admin','odontologist','student'),
  activo BOOL, created_at, updated_at
)

analyses (
  id UUID PK, user_id FK→users,
  filename, original_path, output_path,
  estado ENUM('pendiente','procesando','completado','error'),
  features JSONB,                -- vector de 12 features
  predicted_class VARCHAR,       -- etiqueta diagnóstica
  probabilities JSONB,           -- dict clase→probabilidad
  confidence FLOAT,
  processing_time_s FLOAT,
  error_message TEXT,
  created_at, updated_at
)
```

---

## Seguridad

- Contraseñas hasheadas con bcrypt (cost factor 12)
- JWT firmado con HS256; expiración configurable (default 8h)
- Validación de tipo MIME y extensión en uploads (no solo extensión)
- Tamaño máximo de archivo configurable (default 20 MB)
- CORS configurado explícitamente (no wildcard en producción)
- Headers de seguridad HTTP via Nginx en producción
- Usuario de sistema no-root en el contenedor Docker

---

## Variables de entorno relevantes

| Variable                     | Default              | Descripción                                |
|------------------------------|----------------------|--------------------------------------------|
| `DATABASE_URL`               | sqlite:///dentascan.db | Conexión a la BD                         |
| `SECRET_KEY`                 | (requerido)          | Clave para firmar JWT                      |
| `ACCESS_TOKEN_EXPIRE_MINUTES`| 480                  | Expiración del token (8h)                  |
| `MAX_FILE_SIZE_MB`           | 20                   | Tamaño máximo de imagen                    |
| `UPLOAD_DIR`                 | uploads/             | Directorio de archivos subidos             |
| `OUTPUT_DIR`                 | output/              | Directorio de imágenes anotadas            |
| `MODELS_DIR`                 | models_ml/           | Directorio de modelos ML (.pkl)            |
| `ENVIRONMENT`                | development          | `development` | `production`              |
| `LOG_LEVEL`                  | INFO                 | Nivel de logging                           |
| `CORS_ORIGINS`               | http://localhost:3000| Orígenes permitidos para CORS              |
