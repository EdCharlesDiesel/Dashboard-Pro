# Deploying Dashboard-Pro on Railway (with Neon)

This runs the dashboard **and** the always-on ingest→score worker on
[Railway](https://railway.com), deployed to **US East (Virginia)** — the same
metro as your Neon database in AWS `us-east-1` — so every DB round-trip is
single-digit milliseconds instead of the ~200 ms it costs from South Africa. The
worker keeps the precomputed board fresh **day and night**; the UI just reads
it.

Three services in one Railway project, all built from this repo:

| Service | Built from | What it is | Public? |
|---|---|---|---|
| **worker** | `Dockerfile` | `python -m src.services.background_scanner` — the 24/7 engine. | no |
| **ui** | `Dockerfile` | The Streamlit dashboard, private-network only. | no |
| **proxy** | `deploy/railway/Dockerfile` | nginx + HTTP basic-auth in front of `ui`. | **yes** |

The proxy exists because **Streamlit authenticates nobody**. Railway hands every
service an optional public HTTPS domain, so the safe shape is: give the domain
to a password-protected nginx and keep `ui` reachable only over Railway's
private network.

**Cost:** roughly **$11–17/month** on the Hobby plan — see
[Cost](#cost) for the breakdown.

---

## 0. Before you start

- A Railway account (Hobby plan, $5/mo, includes $5 of usage).
- Your **Neon connection string** (Neon console → *Connection Details* → copy
  the **pooled** URI, the one whose host contains `-pooler`).
- This repo pushed to GitHub (Railway deploys from a branch).
- ⚠️ **Rotate first.** The Neon password and Anthropic key committed in
  `.streamlit/secrets.toml.example` are live credentials in git history. Rotate
  both and scrub them from history *before* making the repo public. Everything
  below goes in Railway environment variables, which never enter a commit or the
  image.

---

## 1. Create the project and pin the region

1. Railway dashboard → **New Project** → **Deploy from GitHub repo** → pick this
   repo and the branch you deploy from (`Production`).
2. **Account Settings → Preferred deployment region → `US East (Virginia)`.**
   Do this *before* the first deploy. This is the whole latency argument — a
   worker in Singapore talking to a Neon database in Virginia throws away the
   reason for deploying at all.
3. Railway will auto-create one service from the repo — it defaults to the repo
   name, `dashboard-pro`. **Rename it to `ui`** (service → Settings → Name).
   You'll add `worker` and `proxy` next.

   ⚠️ **The private hostname follows the service name**: a service called
   `dashboard-pro` is `dashboard-pro.railway.internal`, not
   `ui.railway.internal`. Either rename it to `ui` as above, or leave the name
   and set the proxy's `UI_HOST` to match. A mismatch here is the #1 cause of a
   502 from the proxy. The service's Networking panel always shows the exact
   hostname to use.

### Project-level variables (set once, shared by `worker` and `ui`)

Project → **Variables** → *Shared Variables*, then reference them from each
service as `${{shared.NAME}}`:

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | **yes** | The pooled Neon URI. Keep `sslmode=require`. `src/core/secrets.py` reads this directly, so this one line configures the whole app + worker. |
| `ANTHROPIC_API_KEY` | no | Claude polish for the Setup Ranker email narrative. |
| `FRED_API_KEY` | no | FRED macro data (macro pages). |
| `GMAIL_SENDER` | no | Grade-A email alerts — use a Gmail **App Password**. |
| `GMAIL_APP_PASSWORD` | no | |
| `GMAIL_RECIPIENT` | no | |

`.env.example` documents the same set. Every optional key degrades gracefully
when unset.

---

## 2. The `worker` service

Project → **New** → **GitHub Repo** → same repo. Name it **`worker`**, then in
Settings:

| Setting | Value |
|---|---|
| Builder | `Dockerfile` |
| Dockerfile Path | `Dockerfile` |
| Start Command | `python -m src.services.background_scanner` |
| Public Networking | **none** (don't generate a domain) |
| Healthcheck Path | *(leave empty — it serves no HTTP)* |
| Restart Policy | `ALWAYS` |

Variables: `DATABASE_URL=${{shared.DATABASE_URL}}` plus whichever optional keys
you set. The email vars belong here — the worker is what fires unattended
Grade-A alerts.

The image's `HEALTHCHECK` (a Streamlit probe) is inert on Railway, which uses
the Healthcheck Path setting instead. Leaving it empty is correct for a headless
loop.

---

## 3. The `ui` service

On the service Railway created in step 1:

| Setting | Value |
|---|---|
| Builder | `Dockerfile` |
| Dockerfile Path | `Dockerfile` |
| Start Command | see below |
| Public Networking | **none** — remove the generated domain if there is one |
| Healthcheck Path | `/_stcore/health` |
| Restart Policy | `ALWAYS` |

Start Command (one line):

```bash
streamlit run app.py --server.port=${PORT:-8501} --server.address=:: --server.headless=true --browser.gatherUsageStats=false --server.enableCORS=false --server.enableXsrfProtection=false
```

Three details in there matter:

- **`--server.address=::`** — Railway's private network is dual-stack (the
  service's Networking panel shows *IPv4 & IPv6* next to its `.railway.internal`
  name), and the internal name resolves to an IPv6 address. `::` on Linux
  dual-stack accepts both families, so it's the binding that works either way;
  `0.0.0.0` can leave the service unreachable over the private network.
- **`--server.port=${PORT:-8501}`**, not a bare `8501` — Railway injects `PORT`
  for whichever service owns a public domain and routes the edge to *that* port,
  so a hard-pinned 8501 gives "Application failed to respond" the moment you
  generate a domain on this service (e.g. to smoke-test it before the proxy
  exists). The `:-8501` fallback covers the normal private case and keeps it in
  sync with the proxy's `UI_PORT`. Railway runs the start command through a
  shell, so the expansion works.
- **CORS and XSRF off together** — Streamlit re-enables CORS if XSRF protection
  is on, with a warning. Safe here for the same reason it was safe behind the
  self-hosted nginx: single origin, and the only route in is the basic-auth
  proxy.

Variables: `DATABASE_URL=${{shared.DATABASE_URL}}` (+ optional keys). Note the
UI also starts its own in-process scanner thread via the sidebar nav — harmless,
and it shares the worker's dedupe ledger, so nothing double-sends.

---

## 4. The `proxy` service (the public entry point)

Project → **New** → **GitHub Repo** → same repo. Name it **`proxy`**:

| Setting | Value |
|---|---|
| Builder | `Dockerfile` |
| Dockerfile Path | `deploy/railway/Dockerfile` |
| Root Directory | `/` (the build context must be the repo root) |
| Start Command | *(leave empty — the image's entrypoint)* |
| Public Networking | **Generate Domain** |
| Restart Policy | `ALWAYS` |

Variables:

| Variable | Value |
|---|---|
| `BASIC_AUTH_USER` | your username |
| `BASIC_AUTH_PASSWORD` | a long random password |
| `UI_HOST` | **Prefer a reference**, not a literal: `${{ui.RAILWAY_PRIVATE_DOMAIN}}` (substitute the UI service's actual name — `${{Dashboard-Pro.RAILWAY_PRIVATE_DOMAIN}}` if you kept the default). Railway maintains that value, so renaming the UI service can't silently break the proxy. Use the **Reference** button on the Variables tab to build the syntax. A hardcoded `ui.railway.internal` / `dashboard-pro.railway.internal` also works. |
| `UI_PORT` | `8501` |

`PORT` is injected by Railway once the domain exists — don't set it yourself.
The container generates `/etc/nginx/.htpasswd` from those two variables at
start, so no password hash is ever committed;
[entrypoint.sh](deploy/railway/entrypoint.sh) refuses to start without them.

---

## 5. Verify

```bash
railway logs --service worker    # look for "[bg-scanner] {...}"
railway logs --service proxy     # look for "[proxy] listening on ..."
```

Within a couple of minutes the worker logs a cycle line
(`scored`/`pairs`/`grade_a`) — that's the board being written to Neon. Then open
the proxy's Railway domain in a browser: you should get a password prompt, then
the dashboard.

---

## 6. Updating after code changes

```bash
git push
```

That's it — Railway rebuilds every service watching that branch. To avoid
rebuilding all three on every commit, set **Watch Paths** per service (e.g. the
`proxy` service only needs `deploy/railway/**`).

---

## Cost

Railway meters actual consumption per second rather than selling instances:
**$10/GB-month RAM, $20/vCPU-month, $0.05/GB egress**. For this workload:

| Service | Resident RAM | Avg CPU | Monthly |
|---|---|---|---|
| worker (300 s cycles over ~24 pairs, mostly network-wait) | ~0.4–0.5 GB | ~0.05–0.15 vCPU | $5–8 |
| ui (Streamlit + plotly + pandas, resident even when idle) | ~0.4–0.6 GB | ~0.03–0.10 vCPU | $5–8 |
| proxy (nginx alpine) | ~0.02 GB | negligible | ~$0.25 |
| Egress (single user) | | ~2–6 GB | ~$0.30 |
| **Total** | | | **~$11–17** |

The dominant cost is *resident memory 24/7*, not CPU — the scientific stack's
baseline RSS is what you pay for whether or not a cycle is running. Don't enable
app sleeping on `worker`; it would defeat the entire point of the service.

---

## What's different from a VM

- **The filesystem is ephemeral.** `worker_board.json`, the
  `*_notify_cache.json` dedupe ledgers, `setup_ranker_score_history.json` and
  `account_state.json` are wiped on every redeploy. This is *already safe by
  design*: Postgres is the source of truth for all of them, `precomputed.py` and
  `account_state.py` dual-write, and `NotifyCache.load()` unions local ∪ DB. So
  **don't attach volumes** — a volume mounts to a single service, so `worker`
  and `ui` couldn't share the JSON anyway, and the DB already covers it.
- **No swap to configure.** The old `t3.micro` needed a 2 GB swapfile to survive
  pandas + worker together. Railway meters memory, so a burst to 1.2 GB costs a
  couple of dollars instead of triggering an OOM kill.
- **TLS is free and automatic.** No certbot, no renewal loop, no `:443` server
  block, no DNS record required (Railway's generated domain works immediately).
- **Postgres stays on Neon.** Railway sells a Postgres add-on, but moving off
  Neon would mean re-pointing `DATABASE_URL` and re-migrating; there's no reason
  to. `docker-compose.yml` still bundles a local Postgres for a self-contained
  laptop run.

---

## Troubleshooting

- **Proxy returns 502** → the `ui` service isn't reachable on the private
  network. Almost always `--server.address` left at `0.0.0.0` instead of `::`,
  or a `UI_PORT` / `--server.port` mismatch.
- **Dashboard loads then hangs on "connecting…"** → the WebSocket upgrade isn't
  getting through. Check `nginx -t` output in the proxy logs and that
  `deploy/railway/nginx.conf.template` still carries the `Upgrade` /
  `Connection` headers.
- **Proxy exits immediately with `BASIC_AUTH_USER is required`** → the variables
  are set on the wrong service. They belong on `proxy`, not on `ui`.
- **Worker logs `board store failed` / DB errors** → check `DATABASE_URL`
  (pooled host, `sslmode=require`) and that the Neon project isn't paused.
- **Everything feels slow** → confirm the deployment region is US East. A
  service in `europe-west4` or `asia-southeast1` pays the same WAN penalty the
  local machine did.
- **Slow first page load after a deploy** → expected. The first load before the
  worker's first cycle computes the board live; subsequent loads read it.
