# HTTPS access with nginx + Let's Encrypt (no SSH tunnel)

This fronts the dashboard with an **nginx reverse proxy** doing TLS and HTTP
basic-auth, so you can reach it from any browser at `https://your-domain`
instead of the SSH tunnel. It layers on top of [DEPLOY-AWS.md](DEPLOY-AWS.md) —
do that first (the `worker` + `ui` containers must already build and run).

Why basic-auth on top of TLS: **Streamlit has no login of its own.** TLS
encrypts the traffic; basic-auth is what actually stops a stranger who finds the
URL from opening your trading dashboard.

Files involved:
- `deploy/nginx/app.conf` — the proxy config (edit the domain in it).
- `docker-compose.aws-tls.yml` — adds the `nginx` + `certbot` services.
- `deploy/nginx/.htpasswd` — the basic-auth users (you create it below).

---

## 1. Point a domain at the instance

You need a domain (or subdomain) you control. In your DNS provider, add an **A
record**:

```
dashboard.example.com   →   <EC2_PUBLIC_IP>
```

Wait for it to resolve (`ping dashboard.example.com` shows the EC2 IP). A bare
IP won't work — Let's Encrypt issues certificates for domain names, not IPs.

---

## 2. Open 80 and 443 in the security group

Add two inbound rules to the instance's security group:

- HTTP (TCP 80) — needed for the ACME challenge and the HTTPS redirect.
- HTTPS (TCP 443) — the actual entry point.

Source: **My IP** if only you use it (recommended); `0.0.0.0/0` only if you
genuinely need access from anywhere — in which case TLS + basic-auth are your
sole protection, so use a strong password in step 4.

Keep SSH (22) restricted to your IP as before.

---

## 3. Put your domain in the nginx config

Edit `deploy/nginx/app.conf` and replace **all four** `dashboard.example.com`
occurrences (two `server_name`, two `ssl_certificate` paths) with your real
domain:

```bash
cd dashboard-pro
sed -i 's/dashboard\.example\.com/YOUR.DOMAIN.COM/g' deploy/nginx/app.conf
```

---

## 4. Create the basic-auth user

Generate a bcrypt htpasswd entry (no local tooling needed — use the httpd
image):

```bash
docker run --rm httpd:alpine htpasswd -nbB you 'a-strong-password' \
  > deploy/nginx/.htpasswd
```

Replace `you` / `a-strong-password`. Add more users by appending more lines
(drop the `>` redirect for the second one, or use `>>`).

---

## 5. Issue the first certificate (the bootstrap step)

nginx can't start referencing a certificate that doesn't exist yet, and certbot
can't use nginx's webroot before nginx is up — so the **first** certificate is
issued with certbot in *standalone* mode (it binds port 80 itself, briefly,
while nginx is not yet running). This writes into the same named volume nginx
will mount.

> The volume name is prefixed with the compose project name, which defaults to
> the directory name. If you cloned into `dashboard-pro`, the volume is
> `dashboard-pro_letsencrypt` (lowercase). Adjust if your directory differs.

```bash
docker run --rm -p 80:80 \
  -v dashboard-pro_letsencrypt:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d YOUR.DOMAIN.COM \
  --email you@example.com --agree-tos --no-eff-email
```

You should see *"Successfully received certificate"*. (If port 80 is busy, stop
anything using it first — e.g. `docker compose ... down`.)

---

## 6. Bring up the full stack with TLS

```bash
docker compose -f docker-compose.aws.yml -f docker-compose.aws-tls.yml up -d --build
```

Check nginx came up cleanly and is serving:

```bash
docker compose -f docker-compose.aws.yml -f docker-compose.aws-tls.yml ps
docker compose -f docker-compose.aws.yml -f docker-compose.aws-tls.yml logs nginx
```

Now open **https://YOUR.DOMAIN.COM** — the browser prompts for the basic-auth
user/password from step 4, then the dashboard loads over HTTPS.

---

## Renewals (automatic)

- The **certbot** service retries `certbot renew` every 12h via nginx's webroot
  (`/var/www/certbot`), which the `:80` server block serves. Let's Encrypt certs
  last 90 days and renew at 30 days remaining.
- The **nginx** service reloads every 6h, so a freshly renewed certificate is
  picked up without any manual step.

Nothing to cron; nothing to touch. You can watch a dry run:

```bash
docker compose -f docker-compose.aws.yml -f docker-compose.aws-tls.yml \
  exec certbot certbot renew --webroot -w /var/www/certbot --dry-run
```

---

## Troubleshooting

- **Browser hangs "Connecting…" / blank app** → the WebSocket isn't proxied.
  Confirm the `Upgrade`/`Connection "upgrade"` lines are present in
  `app.conf` (they are by default) and that you edited the domain, not those.
- **`nginx: cannot load certificate … No such file`** → step 5 didn't land in
  the volume nginx mounts. Re-check the volume name prefix (project/dir name)
  and that the domain in the cert path matches the one you issued.
- **502 Bad Gateway** → the `ui` container isn't healthy yet; give it ~30s after
  `up`, then `docker compose ... logs ui`.
- **ACME "challenge failed"** → DNS A record isn't pointing at this instance
  yet, or port 80 isn't open in the security group.
- **Streamlit XSRF/CORS behind proxy** → already handled: the `ui` service in
  `docker-compose.aws.yml` runs with `--server.enableCORS=false
  --server.enableXsrfProtection=false` (the two must be set together — CORS
  alone is overridden back on). Single-origin behind nginx with TLS +
  basic-auth is what makes that safe. No action needed.
