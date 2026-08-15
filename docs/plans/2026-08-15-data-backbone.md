# Data Backbone: Retire the Duplicate, Fix the Macro Store — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the `worker` container maintaining a price store nobody reads, and turn its broken FRED half into the shared macro store that pages currently bypass with per-page HTTP calls.

**Architecture:** `src/data_backbone` writes two tables. `price_bars` duplicates `market_bars`, which the OHLC spine already fills and everything already reads — that half is dead weight and goes. `fred_series` is empty only because of a spoofed User-Agent that makes FRED hang; fixing one header makes it work, and a thin `src/services/fred_data.py` then does for macro series exactly what `market_data.py` does for OHLC. Pages stop calling the FRED API directly, and a static guard keeps them off it.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), Postgres 18, SQLAlchemy 2.0 Core, APScheduler, requests, pytest + pure `ast`, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — the "Single sources of truth" section and the `market_data.py` bullet, whose rule this plan extends from price data to macro data.

## Global Constraints

- Never commit. Make changes only; the repo owner reviews and commits.
- **A plan gets its own bump too.** This plan took **1.9.1** on creation, so
  Task 1 lands on 1.9.2 and Task 2 on 1.9.3.
- **Bump the version on every completed task** (`python deploy/sync_version.py <next>`), and rebuild + `python deploy/verify_deploy.py` before calling a task done — a fix in git is not a fix in production.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`, scoped to `src/` pure logic + DB layer.
- Never remove an `@st.cache_data` decorator.
- Dropping a table is irreversible: `pg_dump` it first, into `backups/`.

---

## Measured starting state (2026-08-15)

| Table | Rows | Latest | Read by |
|---|---|---|---|
| `market_bars` | 206,216 | 2026-08-15 | the OHLC spine, i.e. everything |
| `price_bars` | 33,806 | 2026-08-15 | **nothing** |
| `fred_series` | **0** | — | **nothing** |

Three facts drive this plan:

1. **The worker is working, at nothing useful.** Its logs show `price AUDJPY=X: upserted 1301 bars` for every registry ticker, daily. Those bars duplicate `market_bars`, and `grep` finds no reader of `price_bars` outside `src/data_backbone/` itself.
2. **`fred_series` is empty for one reason.** Measured inside the running container:
   ```
   worker headers -> FAILED ReadTimeout after 20.1s
   no custom UA   -> 200  0.3s  423307 bytes
   ```
   `data_access.HEADERS = {"User-Agent": "Mozilla/5.0 (trading-dashboard)"}` makes `fred.stlouisfed.org` hang. It has been failing every run, at WARNING level, so nothing ever alerted.
3. **Pages hit FRED directly.** `pages/quant_models_tab.py::_fetch_fred`, `pages/disconnect_monitor_tab.py::fetch_fred` and `src/pages_lib/market_overview_lib.py` each call the FRED API themselves — no shared cache, no persistence, re-fetched per page per session. That is the same drift the OHLC spine just fixed, in a different data type.

## File Structure

- **Modify** `src/data_backbone/data_access.py` — drop the User-Agent that breaks FRED.
- **Create** `src/services/fred_data.py` — the macro spine. One responsibility: serve a FRED series from Postgres, falling back to HTTP and writing back. Mirrors `market_data.py`'s shape so the two read alike.
- **Create** `tests/test_fred_data.py`, `tests/test_fred_spine.py` — behaviour, and the static guard.
- **Modify** the three FRED callers to use it.
- **Modify** `src/data_backbone/worker.py` and `docker-compose.yml` — worker becomes FRED-only.
- **Delete** the `price_bars` half of `src/data_backbone/db.py` and the table.

---

### Task 1: Make FRED work again

**Files:**
- Modify: `src/data_backbone/data_access.py:26`
- Test: `tests/test_data_backbone_fred.py` (create)

**Interfaces:**
- Produces: `data_access.fetch_fred(series_id: str) -> pd.Series`, unchanged signature, now returning data instead of raising.

- [ ] **Step 1: Write the failing test**

```python
"""Guard: the FRED fetch must not spoof a browser User-Agent.

Measured 2026-08-15 inside the worker container: the same URL returned 200 in
0.3s with no custom header, and hung to a 20s ReadTimeout with
`Mozilla/5.0 (trading-dashboard)`. `fred_series` sat empty for that reason
alone, logged at WARNING so nothing alerted.
"""
from src.data_backbone import data_access


def test_fetch_fred_does_not_spoof_a_browser_user_agent():
    ua = (data_access.HEADERS or {}).get("User-Agent", "")
    assert "Mozilla" not in ua, (
        "A spoofed browser UA makes fred.stlouisfed.org hang: " + ua
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_data_backbone_fred.py -v --no-cov`
Expected: FAIL — `Mozilla/5.0 (trading-dashboard)`.

- [ ] **Step 3: Fix the header**

```python
# No User-Agent override. A spoofed browser UA
# ("Mozilla/5.0 (trading-dashboard)") makes fred.stlouisfed.org hang until the
# read timeout — measured 2026-08-15: 200 in 0.3s bare, ReadTimeout at 20s
# with it. requests' own UA is fine and is what actually works.
HEADERS: dict[str, str] = {}
```

- [ ] **Step 4: Run the test, then prove it end-to-end**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_data_backbone_fred.py -v --no-cov
docker compose build && docker compose up -d
docker exec dashboard-pro-worker-1 python -c "from src.data_backbone import data_access as da; s=da.fetch_fred('DFF'); print(len(s), s.index.max())"
```
Expected: test passes; the fetch prints a row count and a recent date.

- [ ] **Step 5: Confirm the store fills**

```bash
docker exec dashboard-pro-db-1 psql -U postgres -d DashboardproDBv1 -c "select count(*), max(date) from fred_series;"
```
Expected: non-zero, dated within a few days. Note the worker refreshes daily, so trigger it directly rather than waiting: `docker exec dashboard-pro-worker-1 python -c "from src.data_backbone.worker import refresh_fred; refresh_fred('DFF')"`.

- [ ] **Step 6: Bump, rebuild, verify sync**

```bash
python deploy/sync_version.py 1.9.2 && docker compose build && docker compose up -d && python deploy/verify_deploy.py
```

- [ ] **Step 7: Show the owner the diff. Do not commit.**

---

### Task 2: The macro spine

**Files:**
- Create: `src/services/fred_data.py`
- Test: `tests/test_fred_data.py`

**Interfaces:**
- Produces: `fred_series(series_id: str, start: str | None = None) -> pd.Series` — Postgres first, HTTP fallback with write-back, naive `DatetimeIndex`, `ttl` fixed here.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pytest
from src.services import fred_data


def test_serves_from_postgres_without_touching_the_network(monkeypatch):
    stored = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    monkeypatch.setattr(fred_data, "_read_stored", lambda sid, start: stored)
    def boom(sid):
        raise AssertionError("network hit despite fresh stored data")
    monkeypatch.setattr(fred_data, "_fetch_remote", boom)
    out = fred_data.fred_series("DFF")
    assert list(out.values) == [1.0, 2.0]


def test_falls_back_to_http_and_writes_back(monkeypatch):
    fetched = pd.Series([3.0], index=pd.to_datetime(["2026-01-03"]))
    written = {}
    monkeypatch.setattr(fred_data, "_read_stored", lambda sid, start: pd.Series(dtype=float))
    monkeypatch.setattr(fred_data, "_fetch_remote", lambda sid: fetched)
    monkeypatch.setattr(fred_data, "_write_back", lambda sid, s: written.setdefault(sid, s))
    out = fred_data.fred_series("DFF")
    assert list(out.values) == [3.0]
    assert "DFF" in written


def test_index_is_naive(monkeypatch):
    aware = pd.Series([1.0], index=pd.to_datetime(["2026-01-01"]).tz_localize("UTC"))
    monkeypatch.setattr(fred_data, "_read_stored", lambda sid, start: aware)
    assert fred_data.fred_series("DFF").index.tz is None


def test_a_dead_backend_returns_empty_rather_than_raising(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(fred_data, "_read_stored", boom)
    monkeypatch.setattr(fred_data, "_fetch_remote", boom)
    assert fred_data.fred_series("DFF").empty
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_fred_data.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: src.services.fred_data`.

- [ ] **Step 3: Implement**

```python
"""Canonical macro spine — one definition of a FRED series, for every page.

The OHLC equivalent is `src/services/market_data.py`. Same reasoning: a page
that fetches FRED itself picks its own window and cache policy, and two pages
then disagree about the same series. Postgres (`fred_series`, filled by
`src.data_backbone.worker`) is the source; HTTP is the fallback, and what it
fetches is written back so the next reader is served locally.
"""
from __future__ import annotations

import pandas as pd

CANONICAL_TTL = 6 * 3600  # seconds — one staleness policy for every series


def _read_stored(series_id: str, start: str | None) -> pd.Series:
    from src.data_backbone import db
    return db.read_fred_series(series_id, start=start)


def _fetch_remote(series_id: str) -> pd.Series:
    from src.data_backbone import data_access
    return data_access.fetch_fred(series_id)


def _write_back(series_id: str, s: pd.Series) -> None:
    from src.data_backbone import db
    db.upsert_fred(series_id, s)


def fred_series(series_id: str, start: str | None = None) -> pd.Series:
    """A FRED series, Postgres first. Never raises — an empty Series on failure."""
    try:
        s = _read_stored(series_id, start)
    except Exception:
        s = pd.Series(dtype=float)
    if s is None or s.empty:
        try:
            s = _fetch_remote(series_id)
            if s is not None and not s.empty:
                try:
                    _write_back(series_id, s)
                except Exception:
                    pass  # serving the caller matters more than caching it
        except Exception:
            return pd.Series(dtype=float)
    if s is None or s.empty:
        return pd.Series(dtype=float)
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    return s
```

- [ ] **Step 4: Run tests**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_fred_data.py -v --no-cov`
Expected: 4 passed.

- [ ] **Step 5: Add `read_fred_series` to `src/data_backbone/db.py`** if absent — `db.py` already has `upsert_fred` and a `read_price_bars`; mirror the latter:

```python
def read_fred_series(series_id: str, start=None) -> pd.Series:
    """Stored observations for one series, oldest first."""
    stmt = select(fred_series.c.date, fred_series.c.value).where(
        fred_series.c.series_id == series_id)
    if start is not None:
        stmt = stmt.where(fred_series.c.date >= pd.Timestamp(start).date())
    with get_engine().connect() as conn:
        rows = conn.execute(stmt.order_by(fred_series.c.date)).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series([r.value for r in rows],
                     index=pd.to_datetime([r.date for r in rows]), name=series_id)
```

- [ ] **Step 6: Bump to 1.9.3, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 3: Point the pages at it, and keep them there

**Files:**
- Modify: `pages/quant_models_tab.py` (`_fetch_fred`), `pages/disconnect_monitor_tab.py` (`fetch_fred`), `src/pages_lib/market_overview_lib.py:849`
- Create: `tests/test_fred_spine.py`

**Interfaces:**
- Consumes: `fred_data.fred_series(series_id, start)` from Task 2.

- [ ] **Step 1: Write the guard, modelled on `tests/test_ohlc_spine.py`**

```python
"""Guard: pages read FRED through src/services/fred_data.py, never directly.

Same rule as the OHLC spine, for macro data: a page that calls the FRED API
itself picks its own window and cache policy, and pages then disagree about
the same series. `MIGRATING` shrinks to empty; adding to it is a deferral.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("pages", "src/pages_lib")
FRED_HOSTS = ("stlouisfed.org", "fredgraph")

MIGRATING: set[str] = {
    "pages/quant_models_tab.py",
    "pages/disconnect_monitor_tab.py",
    "src/pages_lib/market_overview_lib.py",
}


def _py_files() -> Iterator[Path]:
    for rel in SEARCH_DIRS:
        yield from (REPO / rel).rglob("*.py")


def _hits_fred_directly(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(h in node.value for h in FRED_HOSTS):
                return True
    return False


def test_no_page_calls_the_fred_api_directly():
    offenders = {
        p.relative_to(REPO).as_posix() for p in _py_files()
        if p.relative_to(REPO).as_posix() not in MIGRATING
        and _hits_fred_directly(p.read_text(encoding="utf-8"))
    }
    assert not offenders, (
        "These call FRED directly instead of via src/services/fred_data.py: "
        + ", ".join(sorted(offenders))
    )


def test_migrating_only_names_files_that_still_offend():
    stale = {
        rel for rel in MIGRATING
        if not (REPO / rel).exists()
        or not _hits_fred_directly((REPO / rel).read_text(encoding="utf-8"))
    }
    assert not stale, "Migrated but still exempted: " + ", ".join(sorted(stale))
```

- [ ] **Step 2: Run it — expect PASS with 3 exemptions, then prove it bites** by removing `quant_models_tab.py` from `MIGRATING` and re-running (expect FAIL naming it). Restore.

- [ ] **Step 3: Migrate `pages/quant_models_tab.py`**

Replace the body of `_fetch_fred` with:

```python
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _fetch_fred(series_id: str, start: str) -> pd.Series:
    """FRED observations from `start`, through the canonical macro spine."""
    from src.services.fred_data import fred_series
    return fred_series(series_id, start=start)
```

Delete the now-unused `FRED_URL` constant and the `fred_api_key` import **only if** nothing else in the file uses them — check with `grep -n "FRED_URL\|fred_api_key" pages/quant_models_tab.py`.

- [ ] **Step 4: Delete its `MIGRATING` entry, run the guard and the page's AppTest smoke**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_fred_spine.py -v --no-cov
PYTEST_CURRENT_TEST=1 PYTHONIOENCODING=utf-8 python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/quant_models_tab.py", default_timeout=300)
at.run()
print("EXCEPTIONS:", [str(e.value) for e in at.exception])
EOF
```
Expected: guard passes with 2 exemptions; `EXCEPTIONS: []`.

- [ ] **Step 5: Repeat Steps 3-4 for `pages/disconnect_monitor_tab.py`** — same replacement, its function is `fetch_fred(series_id, start)` and already has that signature.

- [ ] **Step 6: Repeat for `src/pages_lib/market_overview_lib.py:849`.** It calls `fetch_fred_series(sid, fred_key)` from `src/core/data_provider.py`, which takes a `limit` rather than a `start`; replace with `fred_series(sid).tail(limit)` so the caller's contract is unchanged.

- [ ] **Step 7: Assert the list is closed**

```python
def test_migration_is_complete():
    assert MIGRATING == set(), "Still calling FRED directly: " + ", ".join(sorted(MIGRATING))
```

- [ ] **Step 8: Bump to 1.10.0, rebuild, `verify_deploy.py`, full suite, show the diff.**

---

### Task 4: Retire the duplicate price store

**Files:**
- Modify: `src/data_backbone/worker.py`, `src/data_backbone/config.py`, `src/data_backbone/db.py`
- Modify: `docker-compose.yml:93-112`

- [ ] **Step 1: Back the table up before touching it**

```bash
docker exec dashboard-pro-db-1 pg_dump -U postgres -d DashboardproDBv1 -t price_bars > backups/price_bars_20260815.sql
```

- [ ] **Step 2: Prove nothing reads it, one more time**

```bash
grep -rn "price_bars\|read_price_bars\|upsert_price_bars" --include=*.py src/ pages/ deploy/ | grep -v "^src/data_backbone/"
```
Expected: no output. If anything appears, STOP and report — the premise is wrong.

- [ ] **Step 3: Drop the price half from the worker**

In `worker.py`, delete `refresh_ticker` and its scheduler registration, leaving the FRED refresh. Update the module docstring to say the worker is the macro refresher and that price bars live in `market_bars` via `market_cache`.

- [ ] **Step 4: Drop the table and its helpers**

Remove `price_bars` Table, `_price_records`, `price_upsert_stmt`, `upsert_price_bars`, `read_price_bars` from `db.py`, then:

```bash
docker exec dashboard-pro-db-1 psql -U postgres -d DashboardproDBv1 -c "DROP TABLE price_bars;"
```

- [ ] **Step 5: Rename the compose service so it says what it is**

```yaml
  macro-worker:
    build: .
    image: dashboard-pro:${APP_VERSION:-dev}
    restart: unless-stopped
    command: ["python", "-m", "src.data_backbone.worker"]
```

- [ ] **Step 6: Verify**

```bash
docker compose up -d
docker logs dashboard-pro-macro-worker-1 --tail 20
PYTHONIOENCODING=utf-8 python -m pytest -q
```
Expected: logs show FRED refreshes and no `price ...: upserted` lines; suite green apart from the 3 known failures.

- [ ] **Step 7: Bump to 1.11.0, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 5: Record it

**Files:**
- Modify: `.claude/CLAUDE.md`, `.foglamp/scan.json`

- [ ] **Step 1: Add a "Single sources of truth" bullet for `src/services/fred_data.py`**, stating that `fred_series` is the only path to FRED, that `tests/test_fred_spine.py` enforces it, and that the store is filled by the macro worker.
- [ ] **Step 2: Update the `data_backbone` node in `scan.json`** to describe a macro-only worker, and delete any `price_bars` edge. Then `python .foglamp/introspect.py && python .foglamp/render.py`.
- [ ] **Step 3: Bump to 1.11.1, rebuild, `verify_deploy.py`, show the diff.**

---

## Alternative considered: delete `src/data_backbone` outright

Simpler — remove the package, the worker service and both tables, and let pages keep calling FRED over HTTP. It was rejected because `fred_series` is one header away from working and closes a real gap: three call sites currently re-fetch the same macro series per page per session with no shared cache or persistence. Deleting it would leave that duplication in place and throw away the only durable macro store. If you would rather take this route, Tasks 1, 4 and 5 still apply and Tasks 2-3 are dropped.

## Self-Review

- **Spec coverage:** the "single source of truth" rule is extended to macro data (Tasks 2-3), and the duplicate price store that violated it is removed (Task 4). Covered.
- **Placeholder scan:** every step carries the actual code or command. Task 3's `market_overview_lib` step names the exact signature mismatch (`limit` vs `start`) and its resolution rather than saying "adapt as needed".
- **Type consistency:** `fred_series(series_id: str, start: str | None) -> pd.Series` is used with those types in Tasks 2 and 3; `read_fred_series` returns the same shape `_read_stored` is expected to.
- **Known risk:** Task 4 drops a table. Step 1 dumps it first and Step 2 re-proves there is no reader, with an explicit STOP if there is.
- **Sequencing:** Task 1 must land before Task 2, or the spine reads an empty table and every page silently falls back to HTTP — working, but proving nothing.

---

Module map: [[Architecture]] · Docs index: [[README]]
