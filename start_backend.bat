@echo off
title DentaScan - Backend

cd /d "%~dp0backend"

if not exist ".venv\" (
    echo [ERROR] No se encontro el entorno virtual.
    echo Ejecuta setup.bat primero.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo =====================================================
echo   DentaScan Backend
echo   API:     http://localhost:8000
echo   Swagger: http://localhost:8000/docs
echo   Presiona Ctrl+C para detener
echo =====================================================
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
pause