#!/usr/bin/env bash
#
# Pull the latest code and restart the service. Run on the Droplet after a push:
#   sudo bash /opt/dashboard/deploy/update.sh

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/dashboard}"
BRANCH="${BRANCH:-Production}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root (sudo bash deploy/update.sh)." >&2
  exit 1
fi

git -C "${APP_DIR}" fetch --depth 1 origin "${BRANCH}"
git -C "${APP_DIR}" reset --hard "origin/${BRANCH}"
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
systemctl restart dashboard
echo "Updated to $(git -C "${APP_DIR}" rev-parse --short HEAD) and restarted."
