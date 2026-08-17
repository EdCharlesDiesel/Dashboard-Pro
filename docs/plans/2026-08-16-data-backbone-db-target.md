# One Way to Find the Database — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/data_backbone` resolve its database the same way every other module does. It has its own env-only resolver that defaults to `localhost:5432/trading` and never reads `secrets.toml`, so on the host it reaches the **native** PostgreSQL, fails authentication, and silently falls back to the FRED API on every macro read.

**Architecture:** Delete the second resolver rather than patch its defaults. `src/core/secrets.db_config()` already does this job for the whole app — `secrets.toml` first, then `DB_*` env vars, then defaults — and it imports nothing from `src`, so `data_backbone` can call it without a cycle. `config.py` keeps exposing `DB_URL` and the `DB_*` names so its four consumers are untouched; only where the values come from changes.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), SQLAlchemy 2.0, psycopg2, Postgres 18, pytest, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — "Single sources of truth (never duplicate these)". A second way to find the database is exactly the duplication that rule forbids.

## Global Constraints

- Never commit. Make changes only; the repository owner reviews and commits.
- **A plan gets its own bump too.** `VERSION` read **1.10.15**, so this plan takes **1.10.16** — the patch, plus one. Each task then bumps to whatever `VERSION` reads when it completes, plus one. Never a minor bump, never a reserved block, never a skipped number.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`; `src/data_backbone/` is in scope, so this needs real unit tests.
- **The container must keep working.** Inside Docker, `DB_HOST=db` / `DB_PORT=5432` is *correct* — the app talks to the `db` service on its internal port. Verify this after the change; see the precedence note below, which is not what you would assume.

---

## Measured starting state (2026-08-16, v1.10.15)

Running `carry_pct()` on the host, which reads through `fred_data` → `data_backbone`:

```
[fred_data] stored read failed for FEDFUNDS: connection to server at "localhost" (::1),
port 5432 failed: FATAL:  password authentication failed for user "postgres"
[fred_data] write-back failed for FEDFUNDS: ...same...
```

— repeated for every series. The call still returned correct numbers (USD/ZAR −3.37%, USD/JPY +2.79%) **because `fred_data` falls back to the FRED API**, so the failure is invisible unless you read the log. The cost is that `fred_data`'s "Postgres first" contract is false on the host: every macro read is an uncached network call, and every write-back is discarded.

### The two resolvers

| | `src/core/secrets.py:db_config()` | `src/data_backbone/config.py` |
|---|---|---|
| reads `secrets.toml` | **yes**, `[database]` first | **no** |
| reads `DATABASE_URL` / `url` | yes | only `DB_URL` |
| env vars | yes, as fallback | yes, as the only source |
| host default | `localhost:5432`, dbname `trading` | `localhost:5432`, dbname `trading` |
| what it resolves to on this host | **127.0.0.1:5433 / DashboardproDBv1** (from `secrets.toml`) | **localhost:5432 / trading** |

Both default to `5432/trading`; the difference is that `db_config()` is overridden by `secrets.toml` and `data_backbone` is not. Note `trading` is a database that does not exist on either server, so even reaching the right port would fail.

### Precedence: secrets.toml wins, and the container relies on its absence

**Verified 2026-08-16, and it is the opposite of the obvious assumption.** In
`db_config()` the `secrets.toml` value is tried *first*:

```python
cfg["host"] = str(db.get("host") or os.environ.get("DB_HOST", "") or cfg["host"])
```

So `secrets.toml` beats `DB_HOST`. If a `secrets.toml` pointing at
`127.0.0.1:5433` were present inside the container, every service would try to
reach the host's loopback from inside Docker and find nothing.

It works today because **`.dockerignore` line 3 excludes `.streamlit/secrets.toml`
from the image**. The container ships only `config.toml` and
`secrets.toml.example`, so the `[database]` section is empty there and the env
vars are what remain. Confirmed by running `db_config()` inside
`dashboard-pro-worker-1`: `{'host': 'db', 'port': 5432, 'dbname': 'DashboardproDBv1', ...}`.

That makes a `.dockerignore` line load-bearing for production's database target,
which nothing currently records. Task 1 adds a guard test for it.

### Why 5432 is not simply "wrong"

`docker-compose.yml` passes `DB_HOST: db` and `DB_PORT: 5432` to `worker`, `scanner`, `sweeper` and `app`. **Inside the container that is correct** — the app reaches the `db` service on its internal port. The container publishes **5433** to the host only because a native Windows PostgreSQL already owns 5432 on both loopbacks.

So the fix is *not* changing the default to 5433. It is making one resolver serve both contexts, which `db_config()` already does.

### Consumers that must not break

`src/data_backbone/`: `db.py`, `data_access.py`, `worker.py`, `seed_history.py` — all import `DB_URL` and/or the `DB_*` names from `config.py`. The names stay; only their source changes.

### Cycle check

`src/core/secrets.py` imports nothing from `src` — verified 2026-08-16, it is a leaf. `data_backbone.config` → `core.secrets` is therefore safe, and the existing chain (`core.data_provider` → `services.fred_data` → `data_backbone.config`) gains no loop.

---

## File structure

- **Modify** `src/data_backbone/config.py` — the DB block only. Watchlists, `STALE_DAYS` and `PRICE_PERIOD` are untouched.
- **Create** `tests/test_data_backbone_config.py` — proves one resolver, both contexts.
- **Unchanged:** `db.py`, `data_access.py`, `worker.py`, `seed_history.py`, `docker-compose.yml`.

---

### Task 1: One resolver

**Files:**
- Modify: `src/data_backbone/config.py:19-28` (the Postgres block)
- Test: `tests/test_data_backbone_config.py`

**Interfaces (unchanged — this is the point):**
- `DB_HOST: str`, `DB_PORT: int`, `DB_NAME: str`, `DB_USER: str`, `DB_PASSWORD: str`, `DB_URL: str` all keep their names and types.

- [ ] **Step 1: Write the failing tests**

`tests/test_data_backbone_config.py`:

```python
"""data_backbone must find the database the same way the rest of the app does.

It had its own env-only resolver defaulting to localhost:5432/trading and never
read secrets.toml, so on the host it reached the *native* PostgreSQL, failed
authentication, and fell back to the FRED API on every macro read -- silently,
because fred_data's fallback works. Measured 2026-08-16: every series logged
"password authentication failed for user postgres" and still returned a number.

The container is the other half of the constraint: inside Docker, DB_HOST=db and
DB_PORT=5432 are CORRECT. Note the precedence is NOT what you would guess --
db_config() prefers secrets.toml over DB_*, and the container is safe only
because .dockerignore keeps secrets.toml out of the image. That line is
load-bearing, so it gets a guard test here.
"""
from __future__ import annotations

import importlib

import pytest


def _reload(monkeypatch, *, env: dict, secrets: dict | None = None):
    """Re-import config under a given environment and secrets.toml."""
    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if secrets is not None:
        from src.core import secrets as secrets_module
        monkeypatch.setattr(secrets_module, "db_config", lambda: secrets)
    import src.data_backbone.config as config
    return importlib.reload(config)


class TestItUsesTheSharedResolver:
    def test_secrets_toml_wins_on_the_host(self, monkeypatch):
        cfg = _reload(monkeypatch, env={}, secrets={
            "host": "127.0.0.1", "port": 5433, "dbname": "DashboardproDBv1",
            "user": "postgres", "password": "secret"})
        assert cfg.DB_PORT == 5433
        assert cfg.DB_NAME == "DashboardproDBv1"
        assert "5433" in cfg.DB_URL and "DashboardproDBv1" in cfg.DB_URL

    def test_it_no_longer_defaults_to_the_native_postgres(self, monkeypatch):
        cfg = _reload(monkeypatch, env={}, secrets={
            "host": "127.0.0.1", "port": 5433, "dbname": "DashboardproDBv1",
            "user": "postgres", "password": "secret"})
        assert cfg.DB_PORT != 5432, "5432 is the native server, not the container"
        assert cfg.DB_NAME != "trading", "no such database exists on either server"


class TestTheContainerStillWorks:
    def test_env_vars_are_used_when_there_is_no_secrets_toml(self, monkeypatch):
        # This is the container's situation: .dockerignore keeps secrets.toml
        # out of the image, so the [database] section is empty and DB_* is all
        # that is left. NOT a test that env beats secrets -- it does not; see
        # the guard below.
        cfg = _reload(monkeypatch, env={
            "DB_HOST": "db", "DB_PORT": "5432", "DB_NAME": "DashboardproDBv1",
            "DB_USER": "postgres", "DB_PASSWORD": "pw"}, secrets=None)
        assert cfg.DB_HOST == "db"
        assert cfg.DB_PORT == 5432

    def test_dockerignore_still_excludes_secrets_toml(self):
        """A .dockerignore line is load-bearing for production's DB target.

        db_config() prefers secrets.toml over DB_*. The container resolves to
        `db:5432` only because secrets.toml is absent from the image. Ship one
        and every service chases the host's 127.0.0.1:5433 from inside Docker.
        """
        from pathlib import Path
        ignore = Path(__file__).resolve().parent.parent / ".dockerignore"
        lines = {ln.strip() for ln in ignore.read_text(encoding="utf-8").splitlines()}
        assert ".streamlit/secrets.toml" in lines, (
            "secrets.toml must stay out of the image: db_config() prefers it "
            "over the DB_* env vars, so shipping it would repoint every "
            "container at the host loopback")

    def test_an_explicit_db_url_still_overrides_everything(self, monkeypatch):
        cfg = _reload(monkeypatch, env={
            "DB_URL": "postgresql+psycopg2://u:p@example:1234/db"}, secrets={})
        assert cfg.DB_URL == "postgresql+psycopg2://u:p@example:1234/db"


class TestItDegradesRatherThanCrashes:
    def test_a_broken_secrets_read_does_not_stop_import(self, monkeypatch):
        from src.core import secrets as secrets_module

        def boom():
            raise RuntimeError("no secrets.toml")

        monkeypatch.setattr(secrets_module, "db_config", boom)
        for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        import src.data_backbone.config as config
        cfg = importlib.reload(config)
        # Importing a config module must never be what takes the app down.
        assert isinstance(cfg.DB_URL, str) and cfg.DB_URL
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_data_backbone_config.py -q --no-cov`
Expected: the `TestItUsesTheSharedResolver` cases FAIL — `DB_PORT == 5432` and `DB_NAME == "trading"`, because `config.py` never consults `secrets.toml`. The container and degradation cases should already pass.

- [ ] **Step 3: Replace the Postgres block**

`src/data_backbone/config.py` — swap lines 19-28 for:

```python
# ── Postgres (SQLAlchemy URL) ────────────────────────────────────────────────
# Resolved through `src.core.secrets.db_config()`, the same path the rest of the
# app uses, so there is exactly one answer to "where is the database".
#
# This module used to read the DB_* env vars itself and default to
# localhost:5432/trading. Inside Docker that was right -- compose passes
# DB_HOST=db / DB_PORT=5432 and the container reaches the `db` service on its
# internal port. On the *host* it was wrong twice over: a native Windows
# PostgreSQL owns 5432, and no database called `trading` exists on either
# server. Every macro read therefore failed authentication and fell back to the
# FRED API, silently, because `fred_data`'s fallback works. Measured 2026-08-16.
#
# NOTE the precedence, which is not the obvious one: db_config() tries the
# secrets.toml value BEFORE the env var. The container is safe only because
# .dockerignore keeps secrets.toml out of the image, leaving the DB_* env vars
# as the only source there. Mount a secrets.toml into a container and every
# service will chase the host's 127.0.0.1:5433 from inside Docker.
def _resolve_db() -> dict:
    """Connection settings, or safe defaults if secrets cannot be read.

    Importing a config module must never be what takes the app down, so a
    failure here degrades to the same defaults the old code used rather than
    raising at import time.
    """
    try:
        from src.core.secrets import db_config
        return dict(db_config())
    except Exception:                       # noqa: BLE001 — import must not fail
        return {"host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "dbname": os.getenv("DB_NAME", "postgres"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "")}


_DB = _resolve_db()

DB_HOST: str = str(_DB.get("host", "localhost"))
DB_PORT: int = int(_DB.get("port", 5432))
DB_NAME: str = str(_DB.get("dbname", "postgres"))
DB_USER: str = str(_DB.get("user", "postgres"))
DB_PASSWORD: str = str(_DB.get("password", ""))

DB_URL: str = os.getenv(
    "DB_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)
```

Two details that matter:
- `DB_URL` keeps its `os.getenv("DB_URL", ...)` override — an explicit URL still wins over everything, unchanged.
- The fallback dbname is `postgres`, not `trading`. `postgres` always exists; `trading` never did, so a failure now surfaces as an empty database rather than a confusing "does not exist".

- [ ] **Step 4: Run the tests** — expect PASS.

- [ ] **Step 5: Prove it on the host**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src.data_backbone.config import DB_HOST, DB_PORT, DB_NAME
print(DB_HOST, DB_PORT, DB_NAME)"
```
Expected: `127.0.0.1 5433 DashboardproDBv1`.

Then the read that was failing, watching for the warnings:
```bash
PYTHONIOENCODING=utf-8 python -c "
from src.services.carry import carry_pct
print('USD/ZAR', carry_pct('USD/ZAR'))" 2>&1 | grep -c "authentication failed"
```
Expected: **0**. Any non-zero means the resolver is still landing on the native server.

- [ ] **Step 6: Prove the container still works**

```bash
docker compose build app && docker compose up -d
docker exec dashboard-pro-worker-1 python -c "
from src.data_backbone.config import DB_HOST, DB_PORT, DB_NAME
print(DB_HOST, DB_PORT, DB_NAME)"
```
Expected: `db 5432 DashboardproDBv1` — the compose env still winning. **If this prints `127.0.0.1 5433`, stop: the fix has broken production and env precedence is inverted.**

- [ ] **Step 7: Confirm the macro store actually fills now**

```bash
docker exec dashboard-pro-worker-1 python -c "
from src.services.fred_data import fred_series
s = fred_series('FEDFUNDS'); print(len(s), s.index.max() if len(s) else '-')"
```
Then check Postgres holds it:
```sql
SELECT count(*) FROM fred_series WHERE series_id = 'FEDFUNDS';
```
Expected: non-zero. Before this fix the write-back was discarded, so the table stayed empty — that is the defect's real cost, not the log noise.

- [ ] **Step 8: Full suite, bump, rebuild, `verify_deploy.py`, show the diff. Do not commit.**

---

## Out of scope, deliberately

- **`_DB_DEFAULTS` in `secrets.py`** also says `port 5432` / `dbname trading`. It is the last-resort default when neither secrets.toml nor env exist, and changing it touches every consumer. Worth its own look; not this plan.
- **Backfilling `fred_series`.** Once the writes land, the worker fills it on its normal schedule. Forcing a backfill is a separate operation.
- **The other `data_backbone` config** (`STALE_DAYS`, `PRICE_PERIOD`, watchlists). Untouched.

## Verification for the whole plan

- [ ] Host: `DB_HOST/PORT/NAME` = `127.0.0.1 / 5433 / DashboardproDBv1`.
- [ ] Host: zero `authentication failed` lines from a macro read.
- [ ] Container: `db / 5432 / DashboardproDBv1` — compose env still wins.
- [ ] `fred_series` rows present in Postgres after a worker read.
- [ ] Full suite: no failures beyond the 3 known.
- [ ] `verify_deploy.py` in sync at the new version.

---

Module map: [[Architecture]] · Docs index: [[README]]
