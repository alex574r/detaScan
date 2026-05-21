# DentaScan — Guía de Despliegue

## Opciones de despliegue

- **Local / desarrollo**: SQLite + uvicorn con hot-reload
- **Docker (recomendado)**: contenedores orquestados con PostgreSQL
- **VPS / servidor propio**: Docker Compose en cualquier servidor Linux

---

## Despliegue con Docker Compose

### 1. Configurar variables de entorno

Copia `.env.example` y edítalo antes de levantar:

```bash
cp backend/.env.example backend/.env
```

Variables que **debes** cambiar en producción:

```bash
SECRET_KEY=genera_una_clave_con_openssl_rand_hex_32
DB_PASSWORD=contraseña_segura_para_postgres
ENVIRONMENT=production
CORS_ORIGINS=https://tu-dominio.com
```

Genera la clave secreta:
```bash
openssl rand -hex 32
```

### 2. Levantar servicios

```bash
make docker-up
# o directamente:
docker compose -f config/docker-compose.yml up --build -d
```

Esto levanta:
- **PostgreSQL 16** en el puerto 5432 (interno)
- **Backend FastAPI** en el puerto 8000
- **Frontend Nginx** en el puerto 80

### 3. Verificar que todo funciona

```bash
# Estado de los contenedores
docker compose -f config/docker-compose.yml ps

# Logs del backend
docker compose -f config/docker-compose.yml logs backend -f

# Health check
curl http://localhost:8000/health
```

### 4. Cargar usuarios iniciales

Los seeds se cargan automáticamente la primera vez que se inicializa la BD. Si necesitas cargarlos manualmente:

```bash
docker compose -f config/docker-compose.yml exec backend \
  python database/seeds/seed.py
```

---

## Despliegue local (sin Docker)

```bash
# Instalar todo y arrancar
make setup
make run

# En otra terminal, servir el frontend
python3 -m http.server 3000 --directory frontend/
```

---

## Variables de entorno completas

| Variable                       | Ejemplo / Default                    | Descripción                                  |
|-------------------------------|--------------------------------------|----------------------------------------------|
| `DATABASE_URL`                | `sqlite:///./dentascan.db`           | Conexión a la BD. Cambiar a PostgreSQL en prod |
| `SECRET_KEY`                  | *(sin default, requerida)*           | Clave para firmar JWT                        |
| `ALGORITHM`                   | `HS256`                              | Algoritmo JWT                                |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `480`                                | Duración del token (8 horas)                 |
| `ENVIRONMENT`                 | `development`                        | `development` o `production`                 |
| `LOG_LEVEL`                   | `INFO`                               | `DEBUG`, `INFO`, `WARNING`, `ERROR`          |
| `MAX_FILE_SIZE_MB`            | `20`                                 | Tamaño máximo de imagen en MB               |
| `UPLOAD_DIR`                  | `uploads/`                           | Dónde se guardan las imágenes subidas        |
| `OUTPUT_DIR`                  | `output/`                            | Dónde se guardan las imágenes anotadas       |
| `MODELS_DIR`                  | `models_ml/`                         | Directorio de modelos `.pkl`                 |
| `CORS_ORIGINS`                | `http://localhost:3000`              | Orígenes permitidos para CORS (separados por coma) |
| `DB_PASSWORD`                 | `DentaScan2024!`                     | Contraseña de PostgreSQL (solo Docker)       |

---

## Configurar HTTPS con Nginx

Si tienes un dominio propio, puedes añadir TLS con Certbot:

```bash
# En el servidor, instalar certbot
sudo apt install certbot python3-certbot-nginx

# Obtener certificado (reemplaza tu-dominio.com)
sudo certbot --nginx -d tu-dominio.com

# Certbot modifica nginx.conf automáticamente
```

Alternativamente, usa un proxy inverso como Caddy que maneja TLS automáticamente:

```caddyfile
tu-dominio.com {
    reverse_proxy /api/* localhost:8000
    reverse_proxy /output/* localhost:8000
    root * /ruta/al/frontend
    file_server
}
```

---

## Actualizar el proyecto

```bash
# Detener servicios
make docker-down

# Actualizar código (git pull o reemplazar archivos)

# Reconstruir y levantar
make docker-up
```

---

## Persistencia de datos

Los volúmenes de Docker se crean automáticamente y persisten entre reinicios:

| Volumen          | Contenido                          |
|------------------|------------------------------------|
| `postgres_data`  | Base de datos PostgreSQL completa  |
| `uploads_data`   | Radiografías subidas por usuarios  |
| `output_data`    | Imágenes anotadas por el pipeline  |
| `models_data`    | Modelos ML entrenados (`.pkl`)     |
| `logs_data`      | Logs de la aplicación              |

Hacer backup de los volúmenes:
```bash
# Backup de la BD
docker compose -f config/docker-compose.yml exec db \
  pg_dump -U dentascan_user dentascan > backup_$(date +%F).sql

# Restaurar
docker compose -f config/docker-compose.yml exec -T db \
  psql -U dentascan_user dentascan < backup_2024-03-15.sql
```

---

## Solución de problemas comunes

**El backend no arranca (error en BD)**
```bash
# Ver logs detallados
docker compose logs backend

# Verificar que la BD esté healthy
docker compose ps db
```

**Subida de imágenes falla con 413**
- Aumentar `MAX_FILE_SIZE_MB` en `.env`
- Si usas Nginx como proxy inverso externo, ajustar `client_max_body_size`

**El clasificador siempre responde en modo demo**
- Los modelos `.pkl` no están en `models_ml/`
- Entrena los modelos: `make train`
- Reinicia el backend para que los cargue

**Error de CORS en el frontend**
- Verificar que `CORS_ORIGINS` incluye la URL exacta del frontend
- En desarrollo: `http://localhost:3000`
- En producción: `https://tu-dominio.com`
