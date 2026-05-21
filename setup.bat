@echo off
:: ─── DentaScan — Setup para Windows ────────────────────────────────────────
:: Uso: Doble clic o ejecutar desde CMD: setup.bat
:: Requiere: Python 3.10+

setlocal enabledelayedexpansion
title DentaScan - Instalacion

echo.
echo =====================================================
echo   DentaScan -- Setup automatico para Windows
echo =====================================================
echo.

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado.
    echo Instala Python 3.12 desde https://python.org
    echo Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Python %PY_VER% encontrado

:: Ir al directorio del backend
cd /d "%~dp0backend"
echo [INFO] Directorio: %CD%

:: Crear entorno virtual
if not exist ".venv\" (
    echo [INFO] Creando entorno virtual...
    python -m venv .venv
    echo [OK] Entorno virtual creado
) else (
    echo [OK] Entorno virtual ya existe
)

:: Activar entorno virtual
call .venv\Scripts\activate.bat

:: Instalar dependencias
echo [INFO] Instalando dependencias (puede tardar unos minutos)...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    echo Revisa tu conexion a internet e intenta de nuevo.
    pause
    exit /b 1
)
echo [OK] Dependencias instaladas

:: Configurar .env
if not exist ".env" (
    copy ".env.example" ".env" >nul
    :: Generar SECRET_KEY aleatoria
    python -c "import secrets; key=secrets.token_hex(32); content=open('.env').read(); open('.env','w').write(content.replace('cambia_esto_por_una_clave_segura', key))"
    echo [OK] Archivo .env creado con clave secreta aleatoria
) else (
    echo [OK] .env ya existe, se conserva
)

:: Inicializar base de datos
echo [INFO] Inicializando base de datos...
python -c "from app.database import engine, Base; from app.models import user, analysis; Base.metadata.create_all(bind=engine); print('[OK] Base de datos lista')"

:: Cargar datos de prueba
echo [INFO] Cargando usuarios de prueba...
python database\seeds\seed.py
echo [OK] Seeds cargados

:: Crear directorios necesarios
if not exist "uploads\" mkdir uploads
if not exist "output\" mkdir output
if not exist "models_ml\" mkdir models_ml
if not exist "logs\" mkdir logs

echo.
echo =====================================================
echo   Instalacion completada exitosamente
echo =====================================================
echo.
echo Credenciales de prueba:
echo   Admin:      admin@dentascan.mx       / Admin1234!
echo   Odontologo: odonto@dentascan.mx      / Dentista1234!
echo   Estudiante: estudiante@dentascan.mx  / Estudiante1234!
echo.
echo Para iniciar el sistema:
echo   1. Ejecuta start_backend.bat
echo   2. Ejecuta start_frontend.bat
echo   3. Abre http://localhost:3000 en tu navegador
echo.
pause
