# Dashboard-Pro — Railway basic-auth proxy (the public entry point).
#
# Streamlit ships with no authentication, so the `ui` service is kept off the
# public internet and only this tiny image gets a Railway domain. Railway's edge
# handles TLS; nginx here handles the password and the WebSocket upgrade.
#
# Build context is the repo root (Railway builds from the repo root and this
# path is set as the service's "Dockerfile Path"), hence the deploy/railway/…
# COPY paths. .dockerignore re-includes deploy/railway/ for exactly this.
#
# Required env vars: BASIC_AUTH_USER, BASIC_AUTH_PASSWORD. See DEPLOY-RAILWAY.md.
FROM nginx:1.27-alpine

# apache2-utils → htpasswd (bcrypt); gettext → envsubst.
RUN apk add --no-cache apache2-utils gettext

COPY deploy/railway/nginx.conf.template /etc/nginx/app.conf.template
COPY deploy/railway/entrypoint.sh       /usr/local/bin/entrypoint.sh

# The repo is checked out on Windows, so the exec bit can't come from git.
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
