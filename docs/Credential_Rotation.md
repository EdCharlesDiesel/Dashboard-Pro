# Credential Rotation

**Status: OPEN.** Audited 2026-09-06. Nothing in this document is rotated yet.

This repository is **public** (`github.com/EdCharlesDiesel/Dashboard-Pro`,
unauthenticated API returns 200). Twelve distinct credential values were
committed to it and every one is still reachable on the remote. Removing a
secret from the working tree does not remove it from history, and a public
repo is cloned, forked, mirrored and scraped continuously — so **rotation at
the provider is the only fix.** Rewriting history is not a substitute and is
not worth doing first; it cannot recall what has already been read.

No value appears in this document. Credentials are identified by name, length
and a 10-character SHA-256 prefix, which is how the audit was run.

---

## What is actually exposed

Measured by hashing every blob of `.env`, `.streamlit/secrets.toml`,
`.streamlit/secrets.production.toml` and `docker-compose.yml` across all
history, then matching those hashes against the values live on this machine.

### Rotate these — public *and* still in use

| Credential | hash | chars | Where it is live | Severity |
|---|---|---|---|---|
| Neon Postgres URL | `a61dbba69d` | 119 | Neon cloud, endpoint still resolving | **Critical** |
| `GMAIL_APP_PASSWORD` | `cdf835319a` | 19 | `.env` → every container | **Critical** |
| `FRED_API_KEY` | `570a8e4444` | 32 | `.env` + `secrets.toml` | High |
| `FMP_API_KEY` | `a61dfe7eb7` | 32 | `.env` + `secrets.toml` | High |
| `FINNHUB_API_KEY` | `371fa206af` | 40 | `.env` + `secrets.toml` | High |
| `DB_PASSWORD` / `POSTGRES_PASSWORD` | `b2a41e3ee4` | 10 | local Docker Postgres only | Low |

The Neon string is the worst of these: an internet-reachable managed Postgres
whose password is public. `nslookup` on
`ep-billowing-sky-aip5p12a.c-4.us-east-1.aws.neon.tech` still returns live AWS
addresses. It is no longer referenced by `.env` or `secrets.toml`, so it looks
retired — but a retired *reference* is not a retired *database*.

### The Gmail password is worse than it looks

`src/core/secrets.py:133` resolves `[gmail] app_password` **before** the
environment. So:

* On the host, `secrets.toml` wins — and its value (`48bcc7e8d0`) is clean.
* In the containers, `.dockerignore` keeps `secrets.toml` out of the image, so
  resolution falls through to `.env` — and that value (`cdf835319a`) **is
  public.**

The two files disagree, and the containers are on the burned side. A Gmail app
password authorises full SMTP send-as for the account.

### Already rotated — no action

`TELEGRAM_BOT_TOKEN` (`346a6733cc`) and `TELEGRAM_CHAT_ID` (`fa47e37350`) are
not in history; the earlier rotation worked. The old truncated token
(`b32b6c2f72`, 35 chars — the one missing its `<bot-id>:` prefix) is public but
dead. Also public and dead: an older Gmail `app_password` (`cf75db112b`), an
`smtp_password` (`223282be15`), a database `password` (`726d349c3c`) and a
10-character `ANTHROPIC_API_KEY` (`dda59792fb`, too short to be a real key).

### Railway production is clean

`DATABASE_URL` (`42a74cd953`), `PGPASSWORD` and `POSTGRES_PASSWORD`
(`8bbadadb42`) for `postgres.railway.internal` are **not** in history. An
earlier note in this repo assumed the Railway URL was exposed; it is not. The
only Railway-shaped values that matched history were `PGPORT` = `5432` and
`PGUSER` = `postgres` — a port number and a default username, not secrets.

---

## Order of work

Worst-reachable first, cheapest-blast-radius last.

### 1. Neon Postgres — critical

Either delete it or rotate it; deleting is better if it is genuinely retired.

1. Sign in to <https://console.neon.tech>.
2. Find the project containing endpoint `ep-billowing-sky-aip5p12a`
   (database `neondb`, region `us-east-1`).
3. If unused: **delete the project.** That invalidates the string outright.
4. If still used: *Roles* → reset the password, then update every consumer.
5. Confirm it is gone: the endpoint should stop accepting the old credentials.

**Check whether it is still production before deleting it.** No code reads a
Neon connection string from config: the seven `grep -ri neon src/ pages/
deploy/` hits are all prose — latency rationale in comments
(`background_scanner.py:13`, `market_cache.py:39`) — and the local
`secrets.toml` points `DATABASE_URL` at `postgres.railway.internal`. But
`deploy/README.md:4` still describes the deploy of record as "Railway + Neon",
and the value the *production* app runs on is the GitHub environment secret
`DATABASE_URL`, which cannot be read from here. Open that secret, or the
Railway variables, and confirm which database production actually uses before
you delete anything. If it turns out production is still on Neon, this is not
a cleanup — it is a live credential rotation with a deploy behind it.

Either way `deploy/README.md` and those comments need a follow-up pass, since
one of the two stories about the production database is stale.

### 2. Gmail app password — critical

1. <https://myaccount.google.com/apppasswords> (requires 2-Step Verification).
2. **Revoke** the existing Dashboard-Pro app password.
3. Create a new one; Google shows 16 characters in four groups.
4. Update: `.env`, `secrets.toml`, **and** the GitHub `production` environment
   secret `GMAIL_APP_PASSWORD`. This is the one place the two local files
   currently disagree, so set both from the same new value.

### 3. Market-data API keys — high

| Key | Where to rotate |
|---|---|
| `FRED_API_KEY` | <https://fredaccount.stlouisfed.org/apikeys> — request a new key, delete the old |
| `FMP_API_KEY` | <https://site.financialmodelingprep.com/developer/docs/dashboard> — regenerate |
| `FINNHUB_API_KEY` | <https://finnhub.io/dashboard> — revoke and reissue |

Each must land in **four** places (see below).

### 4. Local Postgres password — low

`b2a41e3ee4` is 10 characters and only ever reachable on `127.0.0.1:5433`, so
exposure is close to harmless — but it is public and 10 characters is weak.
Rotate it with the stack down:

```bash
cd /c/x/Dashboard-Pro && docker compose down
```

Set a new `DB_PASSWORD` **and** `POSTGRES_PASSWORD` in `.env` to the same
value, then recreate. Note that changing `POSTGRES_PASSWORD` alone does not
change the password inside an existing data volume — Postgres only reads it
when initialising. Either `ALTER USER postgres WITH PASSWORD ...` first, or
recreate the volume and re-run the backfill.

---

## Where each value has to land

Missing one of these is the usual way a rotation half-lands and shows up days
later as a scheduled-job outage.

| Location | What reads it | How to set |
|---|---|---|
| `.env` | the five containers, via compose interpolation | rotation tool, below |
| `.streamlit/secrets.toml` | the app on this host | rotation tool, below |
| GitHub → Settings → Environments → **production** | `build.yml` writes `secrets.toml` on the runner at deploy time | web UI |
| Railway → service → Variables | the production runtime | Railway dashboard |

The GitHub `production` environment holds nine secrets: `DATABASE_URL`,
`FRED_API_KEY`, `FMP_API_KEY`, `FINNHUB_API_KEY`, `GMAIL_SENDER`,
`GMAIL_APP_PASSWORD`, `GMAIL_RECIPIENT`, `TELEGRAM_BOT_TOKEN`,
`TELEGRAM_CHAT_ID`, plus `RAILWAY_TOKEN`. The deploy job's sanity-check step
prints a present/absent table for all nine, so a missed one is visible in the
run log without ever printing a value.

---

## The propagation tool

Updating `.env` and `secrets.toml` by hand invites three specific mistakes this
desk has already made: writing a `$` un-doubled into `.env` (compose silently
resolves it to empty), flipping the files from CRLF to LF (a whole-file diff in
a `core.autocrlf=true` repo), and "rotating" a key to the value it already had.

`rotate_credentials.py` handles all three. It lives in the session scratchpad
rather than `scratchpad/` in the repo, because **`scratchpad/` is not
gitignored** and neither the tool nor its input should ever be near a commit.

```bash
python rotate_credentials.py --new /path/to/new_values.env
```

Dry-run by default; `--apply` writes, after backing both files up. Behaviour:

* re-derives every burned hash from git history at runtime and **refuses** a
  new value matching one — a paste slip cannot rotate a key to itself;
* refuses a value identical to the current one, or of the wrong shape
  (a Telegram token without its `<bot-id>:` prefix, a Gmail password that is
  not 16 characters, a FRED key that is not 32 lowercase alphanumerics);
* doubles `$` when writing `.env` and **only** there — TOML is not
  interpolated, so `secrets.toml` gets the raw value;
* preserves CRLF in both files and leaves untouched keys byte-identical;
* refuses an input file that sits inside the repo unless it is gitignored;
* prints names, lengths and hash prefixes — never a value.

The input file is `KEY=value` lines. Delete it once applied.

---

## Verification

1. Re-run the audit; every live credential must come back `no` for exposed.
2. `docker compose config --quiet` parses, and the stack comes up healthy.
3. From inside a container, the database resolves:
   `docker exec dashboard-pro-app-1 python -c "from src.db.market_cache import _resolve_cfg; print(_resolve_cfg() is not None)"`
4. One live call per rotated provider (FRED, FMP, Finnhub), plus one test email
   and one Telegram send.
5. `python -m pytest -q` — the suite must stay at its known baseline
   (2073 passing, 2 known GARCH failures).
6. A production merge: the deploy job's secret table must show 9/9 present.

## Afterwards

* `tests/test_no_secrets_tracked.py` already guards against re-committing a
  secrets file. It does **not** check whether a *live* value appears in
  history — that check exists only as the ad-hoc audit behind this document,
  and is worth promoting to a guard test.
* Consider whether this repository needs to be public at all. Everything here
  is a private trading desk; public visibility buys nothing and is what turned
  a mistake into an incident.
