#!/usr/bin/env bash
#
# One-shot provisioner for a fresh Ubuntu 24.04 DigitalOcean Droplet.
# Installs Postgres + Python + nginx, deploys the app under /opt/dashboard,
# and wires up the systemd service and nginx reverse proxy.
#
# Usage (run as root on the Droplet):
#   curl -fsSL https://raw.githubusercontent.com/EdCharlesDiesel/Dashboard-Pro/Production/deploy/bootstrap.sh | bash
#   # or, after cloning: sudo bash deploy/bootstrap.sh
#
# Idempotent-ish: safe to re-run to pick up code changes. It will NOT overwrite
# an existing secrets.toml or reset the database password.

set -euo pipefail

# ---- Config (override via env vars) -----------------------------------------
REPO_URL="${REPO_URL:-https://github.com/EdCharlesDiesel/Dashboard-Pro.git}"
BRANCH="${BRANCH:-Production}"
APP_DIR="${APP_DIR:-/opt/dashboard}"
DB_NAME="${DB_NAME:-dashboard}"
DB_USER="${DB_USER:-dashuser}"
DB_PASS="${DB_PASS:-}"          # if empty, a random password is generated
ADD_SWAP="${ADD_SWAP:-auto}"    # auto = add 2G swap when RAM < 1.5G; yes/no to force

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo bash deploy/bootstrap.sh)." >&2
  exit 1
fi

# ---- Swap (heavy deps: pandas/numpy/scipy/statsmodels can OOM on 1GB) --------
mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
if [[ "$ADD_SWAP" == "yes" ]] || { [[ "$ADD_SWAP" == "auto" ]] && (( mem_kb < 1572864 )); }; then
  if [[ ! -f /swapfile ]]; then
    log "Adding 2G swap (low RAM detected)"
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
  fi
fi

# ---- System packages --------------------------------------------------------
log "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip postgresql nginx git curl

# ---- Postgres: database + user ----------------------------------------------
log "Configuring Postgres"
systemctl enable --now postgresql

if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
  if [[ -z "$DB_PASS" ]]; then
    DB_PASS="$(openssl rand -base64 18)"
    echo "Generated DB password for ${DB_USER}: ${DB_PASS}"
    echo "(saved into ${APP_DIR}/.streamlit/secrets.toml below)"
  fi
  sudo -u postgres psql <<SQL
CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';
CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
SQL
  sudo -u postgres psql -d "${DB_NAME}" -c "GRANT ALL ON SCHEMA public TO ${DB_USER};"
else
  echo "DB user ${DB_USER} already exists — leaving it (and its password) untouched."
fi

# ---- App code ---------------------------------------------------------------
if [[ -d "${APP_DIR}/.git" ]]; then
  log "Updating existing checkout at ${APP_DIR}"
  git -C "${APP_DIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
else
  log "Cloning ${REPO_URL} (${BRANCH}) into ${APP_DIR}"
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${APP_DIR}"
fi

log "Installing Python dependencies"
python3 -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# ---- Secrets ----------------------------------------------------------------
mkdir -p "${APP_DIR}/.streamlit"
SECRETS="${APP_DIR}/.streamlit/secrets.toml"
if [[ ! -f "$SECRETS" ]]; then
  log "Writing ${SECRETS} (edit later to add FRED/Gmail keys)"
  cat > "$SECRETS" <<TOML
[database]
host = "localhost"
port = 5432
dbname = "${DB_NAME}"
user = "${DB_USER}"
password = "${DB_PASS}"

[api]
fred_api_key = ""

[gmail]
sender = ""
app_password = ""
recipient = ""
TOML
  chmod 600 "$SECRETS"
else
  echo "secrets.toml already exists — not overwriting."
fi

# ---- systemd service --------------------------------------------------------
log "Installing systemd service"
install -m 644 "${APP_DIR}/deploy/dashboard.service" /etc/systemd/system/dashboard.service
systemctl daemon-reload
systemctl enable --now dashboard
systemctl restart dashboard

# ---- nginx ------------------------------------------------------------------
log "Configuring nginx reverse proxy"
install -m 644 "${APP_DIR}/deploy/nginx.conf" /etc/nginx/sites-available/dashboard
ln -sf /etc/nginx/sites-available/dashboard /etc/nginx/sites-enabled/dashboard
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

# ---- Firewall ---------------------------------------------------------------
if command -v ufw >/dev/null; then
  log "Configuring firewall"
  ufw allow OpenSSH
  ufw allow 'Nginx HTTP'
  ufw --force enable
fi

log "Done. App is live on http://$(curl -fsSL ifconfig.me 2>/dev/null || echo YOUR_DROPLET_IP)/"
echo "Next steps:"
echo "  - Edit ${SECRETS} to add FRED/Gmail keys, then: systemctl restart dashboard"
echo "  - Point a domain at this IP, set server_name in /etc/nginx/sites-available/dashboard,"
echo "    then enable HTTPS:  certbot --nginx -d YOUR_DOMAIN"
