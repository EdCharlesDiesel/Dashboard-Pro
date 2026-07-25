# Deploying Dashboard-Pro on AWS EC2 (with Neon)

This runs the dashboard **and** the always-on ingest→score worker on a single
small EC2 instance in **`us-east-1`** — the same AWS region as your Neon
database — so every DB round-trip is sub-millisecond instead of the ~200ms it
costs from South Africa. The worker keeps the precomputed board fresh **day and
night**; the UI just reads it.

Two containers from one image (`docker-compose.aws.yml`):

- **worker** — `python -m src.services.background_scanner`, the 24/7 engine.
- **ui** — the Streamlit dashboard, bound to localhost (reached via SSH tunnel).

**Cost:** `t3.micro` is free for your first 12 months (AWS Free Tier), ~$7.50/mo
after. Neon stays on whatever plan you already use.

---

## 0. Before you start

- An AWS account and an EC2 **key pair** (`.pem`) for SSH.
- Your **Neon connection string** (Neon console → *Connection Details* → copy
  the **pooled** URI, the one whose host contains `-pooler`).
- ⚠️ **Rotate first.** Rotate the Neon password and the Anthropic key that were
  committed to `.streamlit/secrets.toml.example`, and scrub them from git
  history, *before* pushing this repo anywhere public. The `.env` you create
  below is gitignored and dockerignored, so real secrets never enter the image
  or a commit.

---

## 1. Launch the instance

In the EC2 console (make sure the region selector reads **N. Virginia /
us-east-1**):

1. **Launch instance** → Name: `dashboard-pro`.
2. **AMI:** Ubuntu Server 24.04 LTS (Free Tier eligible).
3. **Instance type:** `t3.micro`.
4. **Key pair:** select or create one.
5. **Network / Security group** — create a new one with a **single inbound
   rule**:
   - SSH (TCP 22) — Source: **My IP** (not `0.0.0.0/0`).
   - **Do NOT open port 8501.** Streamlit has no authentication; we reach the UI
     through an SSH tunnel instead. (Outbound stays "all traffic" — the worker
     needs to reach Yahoo and Neon.)
6. **Storage:** 20 GB gp3 (the 8 GB default is tight once images are built).
7. Launch.

---

## 2. Connect and prepare the host

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

Add swap (protects the 1 GB `t3.micro` from OOM when pandas + the worker run
together), then install Docker (the convenience script includes the Compose v2
plugin):

```bash
# 2 GB swap, persistent across reboots
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Docker + Compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
newgrp docker      # apply the group without logging out
```

Verify: `docker version` and `docker compose version` both print without sudo.

---

## 3. Get the code and configure secrets

```bash
git clone <your-repo-url> dashboard-pro
cd dashboard-pro
cp .env.example .env
nano .env          # paste your Neon DATABASE_URL (+ optional API/email keys)
```

`DATABASE_URL` is the only required line. Keep `sslmode=require` and use the
`-pooler` host. Save and exit.

---

## 4. Build and start

```bash
docker compose -f docker-compose.aws.yml up -d --build
```

First build takes a few minutes (compiling the scientific stack). Then check
both services are healthy and the worker is cycling:

```bash
docker compose -f docker-compose.aws.yml ps
docker compose -f docker-compose.aws.yml logs -f worker   # look for "[bg-scanner] {...}"
```

Within a couple of minutes the worker logs a cycle line
(`scored`/`pairs`/`grade_a`) — that's the board being written to Neon.

---

## 5. Open the dashboard (SSH tunnel)

From **your local machine** (not the server):

```bash
ssh -i /path/to/key.pem -L 8501:localhost:8501 ubuntu@<EC2_PUBLIC_IP>
```

Leave that session open and browse to **http://localhost:8501**. The tunnel
forwards your local 8501 to the container's 8501 over SSH — encrypted, and
nothing is exposed to the public internet.

---

## 6. Updating after code changes

```bash
cd dashboard-pro
git pull
docker compose -f docker-compose.aws.yml up -d --build
```

`restart: unless-stopped` means both containers also come back automatically
after an instance reboot.

---

## 7. (Optional) Expose it properly with HTTPS

To reach the dashboard from a browser without the SSH tunnel, do **not** just
open port 8501 — Streamlit authenticates nobody. Instead front it with the
nginx + TLS + basic-auth reverse proxy that ships in this repo
(`docker-compose.aws-tls.yml` + `deploy/nginx/app.conf`):

```bash
docker compose -f docker-compose.aws.yml -f docker-compose.aws-tls.yml up -d --build
```

That needs a domain, a certificate, and an htpasswd file first — the full
step-by-step (DNS, security-group rules, Let's Encrypt issuance, renewals) is in
**[DEPLOY-AWS-TLS.md](DEPLOY-AWS-TLS.md)**.

Until that's in place, the SSH tunnel in step 5 is the safe way in.

---

## Troubleshooting

- **Worker logs `board store failed` / DB errors** → check `DATABASE_URL` in
  `.env` (pooled host, `sslmode=require`), and that the Neon project isn't
  paused.
- **UI OOM / killed** → confirm swap is on (`free -h` shows 2 GB), or move to
  `t3.small` (2 GB RAM) in the console.
- **`docker: permission denied`** → you skipped `newgrp docker` (or need to log
  out/in once after `usermod -aG`).
- **Slow first page** → the very first load before the worker's first cycle
  computes live; subsequent loads read the board.
