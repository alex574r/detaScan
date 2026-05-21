@echo off
:: ─── DentaScan — Iniciar Frontend ───────────────────────────────────────────
title DentaScan - Frontend

cd /d "%~dp0frontend"

echo.
echo =====================================================
echo   DentaScan Frontend
echo   URL: http://localhost:3000
echo   Presiona Ctrl+C para detener
echo =====================================================
echo.

:: Abrir navegador automaticamente (espera 2 segundos)
start "" /b cmd /c "timeout /t 2 >nul && start http://localhost:3000"

python -m http.server 3000
pause
