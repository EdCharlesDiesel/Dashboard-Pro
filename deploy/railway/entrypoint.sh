#!/bin/sh
# Entrypoint for the Railway basic-auth proxy service.
#
# Three jobs, all at container start so nothing secret is ever committed or
# baked into the image:
#   1. generate /etc/nginx/.htpasswd from the BASIC_AUTH_* env vars,
#   2. detect the container's DNS resolver (Railway's internal names resolve to
#      IPv6, so the address needs bracketing for nginx),
#   3. render nginx.conf.template and exec nginx in the foreground.
set -eu

# Fail fast and loudly: an unauthenticated dashboard is worse than a dead one.
: "${BASIC_AUTH_USER:?BASIC_AUTH_USER is required (set it on the proxy service)}"
: "${BASIC_AUTH_PASSWORD:?BASIC_AUTH_PASSWORD is required (set it on the proxy service)}"

# Railway injects PORT for the service that owns the public domain.
PORT="${PORT:-8080}"
# The Streamlit service's private hostname + fixed port (see DEPLOY-RAILWAY.md).
UI_HOST="${UI_HOST:-ui.railway.internal}"
UI_PORT="${UI_PORT:-8501}"

# bcrypt (-B), created (-c) from stdin-free batch mode (-b). Root-only.
htpasswd -cbB /etc/nginx/.htpasswd "$BASIC_AUTH_USER" "$BASIC_AUTH_PASSWORD" >/dev/null
chmod 600 /etc/nginx/.htpasswd

# nginx wants an IPv6 resolver address in brackets. 127.0.0.11 is the Docker
# embedded DNS, so the same image also runs under plain docker compose locally.
NAMESERVER="$(awk '/^nameserver/ { print $2; exit }' /etc/resolv.conf 2>/dev/null || true)"
case "$NAMESERVER" in
    '')  NGINX_RESOLVER='127.0.0.11' ;;
    *:*) NGINX_RESOLVER="[$NAMESERVER]" ;;
    *)   NGINX_RESOLVER="$NAMESERVER" ;;
esac

export PORT UI_HOST UI_PORT NGINX_RESOLVER

# Restrict envsubst to our own variables — otherwise it would eat nginx's
# $host / $remote_addr / $connection_upgrade and produce an empty config.
envsubst '${PORT} ${UI_HOST} ${UI_PORT} ${NGINX_RESOLVER}' \
    < /etc/nginx/app.conf.template \
    > /etc/nginx/conf.d/default.conf

echo "[proxy] listening on ${PORT}, upstream ${UI_HOST}:${UI_PORT}, resolver ${NGINX_RESOLVER}"
nginx -t

exec nginx -g 'daemon off;'
