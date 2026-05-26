#!/usr/bin/env python3
"""
DentaScan — Servidor de desarrollo para el frontend.

Sirve el directorio actual con cabeceras Cache-Control: no-cache
para que el navegador siempre descargue los archivos más recientes.

Uso (desde la raíz del proyecto):
    python3 frontend/server.py          # puerto por defecto: 3000
    python3 frontend/server.py 8080     # puerto personalizado

O directamente desde frontend/:
    cd frontend && python3 server.py
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

# Cambiar al directorio del script para servir los archivos correctamente
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Handler que inyecta cabeceras no-cache en cada respuesta."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Mostrar solo método + ruta + código de estado (sin fecha/hora larga)
        print(f"  {self.command} {self.path}  →  {args[1]}")


if __name__ == "__main__":
    with http.server.HTTPServer(("", PORT), NoCacheHandler) as httpd:
        print(f"\n  DentaScan dev server  →  http://localhost:{PORT}\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Servidor detenido.")
