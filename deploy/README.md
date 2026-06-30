# Deploy — DigitalOcean (single Droplet)

Cheapest production setup: **one Droplet running both Streamlit and Postgres**.
DigitalOcean Managed Postgres starts at $15/mo — more than a whole Droplet — so
we self-host Postgres on `localhost` alongside the app.

| Option | Monthly |
|---|---|
| **Droplet (this setup)** | **$6–12** |
| App Platform + Managed Postgres | $27+ |

**Droplet size:** the deps are heavy (pandas, numpy, scipy, statsmodels,
matplotlib). The 512MB–1GB droplets can OOM during `pip install`. Use the
**$12/mo 2GB** plan, or take the 1GB and let `bootstrap.sh` add a 2GB swap file
automatically.

## Files

| File | Purpose |
|---|---|
| `bootstrap.sh` | One-shot provisioner: Postgres + Python + nginx + systemd + firewall. |
| `update.sh` | Pull latest code, reinstall deps, restart the service. |
| `dashboard.service` | systemd unit that runs Streamlit on `127.0.0.1:8501`. |
| `nginx.conf` | Reverse proxy `:80 → :8501` (with the WebSocket headers Streamlit needs). |
| `secrets.toml.example` | Template for `.streamlit/secrets.toml` (gitignored on the server). |

## Quick start

1. Create an Ubuntu 24.04 Droplet (Basic, 2GB) and add your SSH key.
2. SSH in as root and run:

   ```bash
   curl -fsSL https://raw.githubusercontent.com/EdCharlesDiesel/Dashboard-Pro/Production/deploy/bootstrap.sh | bash
   ```

   The script clones the repo to `/opt/dashboard`, creates the `dashboard`
   database + `dashuser` (random password), writes `secrets.toml`, installs the
   venv, and starts the service behind nginx. The app's `auto_connect()`
   migrates the `trade_setups` schema on first run — no manual SQL needed.

3. Open `http://YOUR_DROPLET_IP/`.

### Override defaults

```bash
DB_NAME=trading DB_USER=trader DB_PASS='s3cret' BRANCH=Production \
  bash deploy/bootstrap.sh
```

## Add a domain + HTTPS (free)

```bash
# point an A record at the Droplet IP, then:
sed -i 's/YOUR_DOMAIN_OR_IP/yourdomain.com/' /etc/nginx/sites-available/dashboard
systemctl restart nginx
apt install -y certbot python3-certbot-nginx
certbot --nginx -d yourdomain.com
```

## Updating after a push

```bash
sudo bash /opt/dashboard/deploy/update.sh
```

## Security notes

- Postgres listens on `localhost` only — **never** open port 5432 to the
  internet. The firewall (ufw) allows just SSH and HTTP.
- `secrets.toml` is written `chmod 600` and is gitignored — keys never enter the
  repo.
- Edit `secrets.toml` to add the FRED key / Gmail alert credentials, then
  `systemctl restart dashboard`.

## Troubleshooting

```bash
systemctl status dashboard          # service state
journalctl -u dashboard -f          # live app logs
nginx -t && systemctl status nginx  # proxy
sudo -u postgres psql -d dashboard  # inspect the DB
```
