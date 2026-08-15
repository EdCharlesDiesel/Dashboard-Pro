# Pages Communicate: One OHLC Spine Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every page reads market data through the canonical spine, so two pages can never disagree about the same instrument because they fetched it separately.

**Architecture:** `src/services/market_data.py` already *is* the spine (`daily_ohlc` / `weekly_ohlc` / `h4_ohlc` / `hourly_ohlc`, one fixed window and TTL per timeframe, routed through `market_cache.cached_ohlc`). Pages have drifted off it. Rather than migrate blind and let drift return, Task 1 lands a static-analysis guard that makes the rule enforceable, seeded with today's offenders in a shrinking `MIGRATING` list; each later task migrates one page and deletes its entry. When `MIGRATING` is empty the rule is self-enforcing.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), pytest + pure `ast`, Streamlit, yfinance, pandas.

**Spec:** `.claude/CLAUDE.md` — "Single sources of truth", `src/services/market_data.py` bullet: *"Pages fetch OHLC through these functions — a page may widen `period`/`days` for chart depth, but interval/resample/TTL are fixed here and nowhere else. Never re-introduce per-page `cached_ohlc`/`yf.download` fetchers with private windows: an EMA over a private lookback is how pages drift out of sync."*

## Global Constraints

- Never commit. Make changes only; the repo owner reviews and commits.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest` (pages emit glyphs that crash cp1252).
- Coverage gate is `--cov-fail-under=80`, scoped to `src/` pure logic + DB layer.
- Widening a window is legal (`daily_ohlc(t, period="2y")`); changing interval, resample rule or TTL outside `market_data.py` is not.
- Never remove an `@st.cache_data` decorator.
- Behaviour must be preserved: a migrated page renders the same bars it did before, from the same ticker over at least the same span.

---

## Measured starting state (2026-08-14)

| OHLC access path | Files |
|---|---|
| Canonical spine (`market_data`) | 18 |
| `market_cache.cached_ohlc` direct | 19 |
| Direct `yf.download` / `yf.Ticker` | 10 |

The 10 direct-yfinance files, with the call that has to move:

| File | Current call | Target |
|---|---|---|
| `pages/abr_toolkit_tab.py:379` | `yf.download(ticker, period="2y", interval="1d")` | `daily_ohlc(ticker, period="2y")` |
| `pages/abr_toolkit_tab.py:382` | `yf.download(ticker, period="720d", interval="1h")` | `hourly_ohlc(ticker, days=720)` |
| `pages/holding_period_tab.py:67` | `yf.download(ticker, period=f"{years}y", interval="1d")` | `daily_ohlc(ticker, period=f"{years}y")` |
| `pages/quant_models_tab.py:50` | `yf.download(ticker, start=start, interval="1d")` | `daily_ohlc(ticker, period=<span>d)` |
| `pages/disconnect_monitor_tab.py:120` | `yf.download(ticker, start=start, interval="1d")` | `daily_ohlc(ticker, period=<span>d)` |
| `pages/event_week_vol_tab.py:149` | `yf.download(ticker, start=start, end=end, interval="1d")` | `daily_ohlc(ticker, period=<span>d)` |
| `pages/liquidity_hunt_tab.py:48` | `yf.download(ticker, period=period, interval=interval)` | spine fn chosen by `interval` |
| `pages/surprise_tab.py:264` | `yf.download(["GC=F","BZ=F"], period="1y")` | two `daily_ohlc` calls, `period="1y"` |
| `pages/overnight_drift_tab.py:53,70` | `ES=F`, `^VIX` | **exception** — not registry instruments |
| `pages/bonds_gold_dxy_app.py:71` | bonds/DXY basket | **exception** — not registry instruments |
| `src/pages_lib/commodity_cot_lib.py` | CFTC COT series | **exception** — not OHLC |

## File Structure

- **Create** `tests/test_ohlc_spine.py` — the AST guard. One responsibility: assert no page fetches OHLC outside the spine, except an explicit, documented allowlist.
- **Modify** one page per task, smallest possible diff: swap the fetch, delete the now-unused `import yfinance as yf`, leave scoring untouched.
- **No new runtime module.** The spine already exists; this plan removes callers that bypass it.

---

### Task 1: The guard test

**Files:**
- Create: `tests/test_ohlc_spine.py`

**Interfaces:**
- Produces: `LEGITIMATE_NON_SPINE: frozenset[str]` (permanent exceptions) and `MIGRATING: set[str]` (shrinks to empty across Tasks 2-8). Later tasks delete one entry from `MIGRATING`.

- [ ] **Step 1: Write the failing test**

```python
"""Guard: pages fetch OHLC through the canonical spine, never privately.

`src/services/market_data.py` fixes one window, resample rule and TTL per
timeframe. A page that calls `yf.download` itself picks a private lookback,
and an EMA over a private lookback is exactly how two pages come to disagree
about the same instrument at the same moment. Widening via the spine's own
`period`/`days` argument stays legal; reaching past it does not.

Pure `ast` — no Streamlit runtime, no network. Mirrors
`tests/test_no_credential_inputs.py`.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parent.parent
SEARCH_DIRS = ("pages", "src/pages_lib")

# Permanent, reasoned exceptions: these fetch series the spine does not model.
LEGITIMATE_NON_SPINE = frozenset({
    "pages/overnight_drift_tab.py",      # ES=F and ^VIX are not registry instruments
    "pages/bonds_gold_dxy_app.py",       # bond yields / DXY basket, not registry pairs
    "src/pages_lib/commodity_cot_lib.py",  # CFTC COT positioning, not OHLC
})

# Shrinks to empty as pages migrate. Adding to this set is not a fix.
MIGRATING = {
    "pages/abr_toolkit_tab.py",
    "pages/holding_period_tab.py",
    "pages/quant_models_tab.py",
    "pages/disconnect_monitor_tab.py",
    "pages/event_week_vol_tab.py",
    "pages/liquidity_hunt_tab.py",
    "pages/surprise_tab.py",
}


def _py_files() -> Iterator[Path]:
    for rel in SEARCH_DIRS:
        yield from (REPO / rel).rglob("*.py")


def _rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _fetches_directly(tree: ast.AST) -> bool:
    """True if the module calls yf.download / yf.Ticker."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith("yf.download") or name.startswith("yf.Ticker"):
            return True
        if name.startswith("yfinance.download") or name.startswith("yfinance.Ticker"):
            return True
    return False


def test_no_page_fetches_ohlc_outside_the_spine():
    offenders = set()
    for path in _py_files():
        rel = _rel(path)
        if rel in LEGITIMATE_NON_SPINE or rel in MIGRATING:
            continue
        if _fetches_directly(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.add(rel)
    assert not offenders, (
        "These fetch OHLC privately instead of via src/services/market_data.py: "
        + ", ".join(sorted(offenders))
        + ". Use daily_ohlc/weekly_ohlc/h4_ohlc/hourly_ohlc (widening "
          "period/days is fine), or add a reasoned entry to LEGITIMATE_NON_SPINE."
    )


def test_migrating_list_only_names_files_that_still_offend():
    """A page that has been migrated must be removed from MIGRATING."""
    stale = set()
    for rel in MIGRATING:
        path = REPO / rel
        if not path.exists():
            stale.add(rel)
            continue
        if not _fetches_directly(ast.parse(path.read_text(encoding="utf-8"))):
            stale.add(rel)
    assert not stale, (
        "Already migrated (or gone) but still exempted in MIGRATING — "
        "delete these entries so the guard covers them: " + ", ".join(sorted(stale))
    )


def test_exception_lists_do_not_overlap():
    assert not (LEGITIMATE_NON_SPINE & MIGRATING)
```

- [ ] **Step 2: Run to verify it passes and genuinely bites**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_ohlc_spine.py -v --no-cov
```
Expected: 3 passed.

Then prove it fails when it should — temporarily remove `"pages/abr_toolkit_tab.py"` from `MIGRATING` and re-run. Expected: `test_no_page_fetches_ohlc_outside_the_spine` FAILS naming that file. Restore the entry.

- [ ] **Step 3: Show the owner the diff. Do not commit.**

---

### Task 2: `pages/abr_toolkit_tab.py`

**Files:**
- Modify: `pages/abr_toolkit_tab.py:379-383`
- Modify: `tests/test_ohlc_spine.py` (delete its `MIGRATING` entry)

**Interfaces:**
- Consumes: `daily_ohlc(ticker: str, period: str) -> pd.DataFrame`, `hourly_ohlc(ticker: str, days: int) -> pd.DataFrame` from `src.services.market_data`.

- [ ] **Step 1: Read the current fetch and its callers**

```bash
sed -n '370,395p' pages/abr_toolkit_tab.py
```

- [ ] **Step 2: Replace the fetch**

```python
from src.services.market_data import daily_ohlc, hourly_ohlc

# 2y of daily bars, and 720d of hourly, both through the spine so the window
# is widened rather than privately redefined.
raw = daily_ohlc(ticker, period="2y") if interval == "1d" else hourly_ohlc(ticker, days=720)
```

Delete `import yfinance as yf` if nothing else in the file uses it:

```bash
grep -n "yf\." pages/abr_toolkit_tab.py
```

- [ ] **Step 3: Delete its entry from `MIGRATING` in `tests/test_ohlc_spine.py`**

- [ ] **Step 4: Verify**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_ohlc_spine.py -v --no-cov
PYTHONIOENCODING=utf-8 python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/abr_toolkit_tab.py", default_timeout=240)
at.run()
print("exceptions:", [str(e.value) for e in at.exception])
EOF
```
Expected: guard passes with one fewer exemption; AppTest prints `exceptions: []`.

- [ ] **Step 5: Show the owner the diff. Do not commit.**

---

### Tasks 3-8: remaining pages

Each follows Task 2 exactly — read the fetch, swap it for the spine call named in the table above, drop the `yf` import if unused, delete the `MIGRATING` entry, run the guard plus that page's AppTest smoke, show the diff.

- [ ] **Task 3:** `pages/holding_period_tab.py:67` → `daily_ohlc(ticker, period=f"{years}y")`
- [ ] **Task 4:** `pages/quant_models_tab.py:50` → `daily_ohlc(ticker, period=f"{(date.today() - start).days}d")`
- [ ] **Task 5:** `pages/disconnect_monitor_tab.py:120` → `daily_ohlc(ticker, period=f"{(date.today() - start).days}d")`
- [ ] **Task 6:** `pages/event_week_vol_tab.py:149` → `daily_ohlc(ticker, period=f"{(end - start).days}d")`, then slice to `[start:end]`
- [ ] **Task 7:** `pages/liquidity_hunt_tab.py:48` → branch on `interval`: `"1d"` → `daily_ohlc(ticker, period=period)`, `"1h"` → `hourly_ohlc(ticker, days=int(period.rstrip("d")))`
- [ ] **Task 8:** `pages/surprise_tab.py:264` → two `daily_ohlc(t, period="1y")` calls joined on the index, replacing the multi-ticker download

---

### Task 9: Close the rule

**Files:**
- Modify: `tests/test_ohlc_spine.py`
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Assert `MIGRATING` is empty**

```python
def test_migration_is_complete():
    """MIGRATING exists only to shrink. Empty means the rule is self-enforcing."""
    assert MIGRATING == set(), (
        "Still bypassing the spine: " + ", ".join(sorted(MIGRATING))
    )
```

- [ ] **Step 2: Run the full suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest -q
```
Expected: 3 pre-existing failures only (2 GARCH, 1 watchdog ordering), everything else green.

- [ ] **Step 3: Record the guard in `.claude/CLAUDE.md`** under the `market_data.py` bullet: one sentence noting `tests/test_ohlc_spine.py` now enforces the rule and that `LEGITIMATE_NON_SPINE` is where a reasoned exception goes.

- [ ] **Step 4: Show the owner the diff. Do not commit.**

---

## Out of scope (deliberately)

Two disconnects measured on 2026-08-14 are **not** addressed here, because each is its own subsystem and mixing them would produce a change nobody can review:

1. **`src/data_backbone` has zero readers outside itself**, while a `worker` container refreshes it daily. Either it becomes the store or it and the container go. Needs its own plan.
2. **27 pages write signals to `trade_setups`; only 2 read any back.** Pages showing their own stored history and the Source Scorecard's verdict is a UI feature, not a data-layer fix. Needs its own plan.

## Self-Review

- **Spec coverage:** the spec's rule has two halves — no private `yf.download` (Tasks 1-8) and no private window (widening via spine args, enforced by the target column of the table). Covered.
- **Placeholder scan:** the `<span>d` in Tasks 4-6 is computed in the step's own code line, not left to the reader.
- **Type consistency:** `daily_ohlc(ticker: str, period: str)` and `hourly_ohlc(ticker: str, days: int)` are used with those exact types in every task.
- **Known gap:** the 19 files calling `market_cache.cached_ohlc` directly are the softer half of the same rule. The guard does not yet flag them, because several are legitimate (`market_data.py` itself, the DB layer). Tightening that needs a per-file read and belongs in a follow-up plan once the loud offenders are gone.

---

Module map: [[Architecture]] · Docs index: [[README]]
