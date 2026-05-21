# ─── DentaScan Frontend — Dockerfile ─────────────────────────────────────────
# Nginx sirve los archivos estáticos del frontend

FROM nginx:1.25-alpine

# Eliminar configuración default de nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copiar archivos del frontend
COPY . /usr/share/nginx/html/

# Nginx corre como usuario nginx por defecto (seguro)
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/index.html || exit 1

CMD ["nginx", "-g", "daemon off;"]
