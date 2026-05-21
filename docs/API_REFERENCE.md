# DentaScan — Referencia de API

Base URL: `http://localhost:8000`  
Documentación interactiva (Swagger): `http://localhost:8000/docs`  
Redoc: `http://localhost:8000/redoc`

---

## Autenticación

Todos los endpoints (excepto `/auth/login` y `/auth/register`) requieren el header:

```
Authorization: Bearer <token>
```

El token se obtiene en el login y expira en 8 horas por defecto.

---

## Endpoints de autenticación

### `POST /auth/register`
Crear una cuenta nueva.

**Body (JSON)**
```json
{
  "email": "usuario@ejemplo.com",
  "password": "MiContraseña123!",
  "nombre": "Dr. García",
  "rol": "odontologist"
}
```
Roles válidos: `admin`, `odontologist`, `student`

**Respuesta 201**
```json
{
  "id": "uuid-aqui",
  "email": "usuario@ejemplo.com",
  "nombre": "Dr. García",
  "rol": "odontologist",
  "activo": true,
  "created_at": "2024-03-15T10:30:00Z"
}
```

**Errores**
- `400` — email ya registrado
- `422` — datos de entrada inválidos

---

### `POST /auth/login`
Iniciar sesión y obtener el JWT.

**Body (JSON)**
```json
{
  "email": "usuario@ejemplo.com",
  "password": "MiContraseña123!"
}
```

**Respuesta 200**
```json
{
  "access_token": "eyJhbGci...",
  "token_type": "bearer",
  "expires_in": 28800,
  "user": {
    "id": "uuid-aqui",
    "email": "usuario@ejemplo.com",
    "nombre": "Dr. García",
    "rol": "odontologist"
  }
}
```

**Errores**
- `401` — credenciales inválidas
- `403` — cuenta desactivada

---

### `GET /auth/me`
Datos del usuario autenticado.

**Respuesta 200**
```json
{
  "id": "uuid-aqui",
  "email": "usuario@ejemplo.com",
  "nombre": "Dr. García",
  "rol": "odontologist",
  "activo": true,
  "created_at": "2024-03-15T10:30:00Z"
}
```

---

## Endpoints de análisis

### `POST /analyses/`
Subir una radiografía y lanzar el análisis.

El análisis corre en background. La respuesta inicial devuelve estado `pendiente`. Usa el endpoint de consulta para hacer polling.

**Content-Type**: `multipart/form-data`

| Campo  | Tipo   | Descripción                                    |
|--------|--------|------------------------------------------------|
| `file` | archivo | Imagen radiográfica (DICOM, PNG, JPEG, TIFF) |

Tamaño máximo: 20 MB.

**Respuesta 202**
```json
{
  "id": "analysis-uuid",
  "filename": "radiografia.png",
  "estado": "pendiente",
  "created_at": "2024-03-15T10:31:00Z",
  "message": "Análisis iniciado. Use el ID para consultar el resultado."
}
```

**Errores**
- `400` — formato de archivo no soportado, o archivo vacío
- `413` — archivo demasiado grande

---

### `GET /analyses/`
Historial de análisis del usuario autenticado.

**Query params**

| Param    | Default | Descripción                              |
|----------|---------|------------------------------------------|
| `skip`   | 0       | Offset para paginación                   |
| `limit`  | 20      | Máximo de resultados (1–100)             |
| `estado` | null    | Filtrar por estado: `completado`, `error`, etc. |

**Respuesta 200**
```json
{
  "analyses": [
    {
      "id": "uuid",
      "filename": "rx_001.png",
      "estado": "completado",
      "predicted_class": "Caries Avanzada",
      "confidence": 0.82,
      "created_at": "2024-03-15T10:31:00Z",
      "processing_time_s": 3.2
    }
  ],
  "total": 15,
  "skip": 0,
  "limit": 20
}
```

---

### `GET /analyses/{analysis_id}`
Resultado completo de un análisis (usar para polling).

**Respuesta 200 — análisis completado**
```json
{
  "id": "uuid",
  "filename": "radiografia.png",
  "estado": "completado",
  "predicted_class": "Caries Incipiente",
  "confidence": 0.74,
  "probabilities": {
    "Diente Sano": 0.08,
    "Caries Incipiente": 0.74,
    "Caries Avanzada": 0.12,
    "Absceso Periapical": 0.04,
    "Lesión Ósea": 0.02
  },
  "features": {
    "media": 127.4,
    "std": 42.1,
    "min_px": 0,
    "max_px": 255,
    "bordes_mean": 18.3,
    "sobel_mean": 22.7,
    "zona_tl": 134.2,
    "zona_tr": 121.8,
    "zona_bl": 118.5,
    "zona_br": 130.1,
    "prop_oscuros": 0.31,
    "asimetria": 0.07
  },
  "output_image_url": "/output/radiografia_annotated_abc123.png",
  "processing_time_s": 3.2,
  "created_at": "2024-03-15T10:31:00Z",
  "completed_at": "2024-03-15T10:31:03Z",
  "disclaimer": "Este resultado no sustituye el criterio clínico de un profesional odontológico."
}
```

**Respuesta 200 — análisis en curso**
```json
{
  "id": "uuid",
  "estado": "procesando",
  "predicted_class": null,
  "confidence": null
}
```

**Respuesta 200 — análisis con error**
```json
{
  "id": "uuid",
  "estado": "error",
  "error_message": "No se pudo leer el archivo: formato DICOM inválido."
}
```

**Errores HTTP**
- `404` — análisis no encontrado o de otro usuario

---

### `DELETE /analyses/{analysis_id}`
Eliminar un análisis y sus archivos asociados.

**Respuesta 200**
```json
{
  "message": "Análisis eliminado correctamente."
}
```

**Errores**
- `404` — no encontrado
- `403` — intentar eliminar análisis de otro usuario (solo admin puede hacerlo)

---

## Endpoints del sistema

### `GET /health`
Estado del servicio. No requiere autenticación.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "environment": "development",
  "database": "connected",
  "ml_mode": "demo"
}
```

El campo `ml_mode` puede ser `demo` (heurístico) o `trained` (modelos RF/SVM cargados).

---

## Imágenes de salida

Las imágenes anotadas se sirven como archivos estáticos:

```
GET /output/{filename}
```

Ejemplo: `GET /output/radiografia_annotated_abc123.png`

No requieren autenticación, pero los nombres son generados con UUID y no son predecibles.

---

## Clases diagnósticas

| Clase                | Descripción                                       |
|----------------------|---------------------------------------------------|
| `Diente Sano`        | Sin anomalías detectadas                          |
| `Caries Incipiente`  | Lesión superficial en esmalte o dentina exterior  |
| `Caries Avanzada`    | Compromiso profundo, posible afectación pulpar    |
| `Absceso Periapical` | Infección en el ápice radicular                   |
| `Lesión Ósea`        | Pérdida de densidad ósea periapical o alveolar    |

---

## Códigos de error comunes

| Código | Significado                                                |
|--------|------------------------------------------------------------|
| 400    | Datos de entrada inválidos o archivo incorrecto            |
| 401    | Token ausente, inválido o expirado                         |
| 403    | Acción no permitida para el rol del usuario                |
| 404    | Recurso no encontrado                                      |
| 413    | Archivo demasiado grande (> 20 MB)                         |
| 422    | Error de validación Pydantic (body incorrecto)             |
| 500    | Error interno del servidor (revisar logs)                  |

---

## Ejemplo completo (curl)

```bash
# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"odonto@dentascan.mx","password":"Dentista1234!"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Subir análisis
ANALYSIS_ID=$(curl -s -X POST http://localhost:8000/analyses/ \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@mi_radiografia.png" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 3. Polling hasta completar
while true; do
  ESTADO=$(curl -s http://localhost:8000/analyses/$ANALYSIS_ID \
    -H "Authorization: Bearer $TOKEN" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['estado'])")
  echo "Estado: $ESTADO"
  [[ "$ESTADO" == "completado" || "$ESTADO" == "error" ]] && break
  sleep 2
done

# 4. Ver resultado completo
curl -s http://localhost:8000/analyses/$ANALYSIS_ID \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```
