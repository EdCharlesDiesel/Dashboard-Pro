# Market Overview: One Spine, Four Horizons — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Show, per instrument, what has changed over the **previous month, week, day and 4 hours** — and make those numbers agree with every other page. Today Market Overview reads its own private windows through `data_provider.fetch_data`, gets 66 daily bars where the rest of the app gets 214, and reports only the change of the single latest bar of whichever tab you are on.

**Architecture:** Two problems, fixed in order, because the second is worthless without the first. (1) The OHLC-spine guard only detects literal `yf.download`/`yf.Ticker`, so `data_provider.fetch_data` walks straight past it — close that hole, then migrate the page. (2) The spine has no monthly window, so add `monthly_ohlc` alongside `weekly_ohlc` using the same resample-from-daily convention. Only then does a four-horizon change panel mean anything: every horizon is a resample of the same daily bars, so the numbers cannot disagree with each other or with any other page.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), Streamlit, pandas, Postgres 18, pytest + pure `ast` for the static guard, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — "Single sources of truth (never duplicate these)" and the `market_data.py` bullet. `tests/test_ohlc_spine.py` is the executable form of that rule; this plan repairs it and then obeys it.

## Global Constraints

- Never commit. Make changes only; the repository owner reviews and commits.
- **A plan gets its own bump too.** `VERSION` read **1.10.5**, so this plan takes **1.10.6** — the patch, plus one.
- **Every completed task bumps the version**: read `VERSION`, add one to the last number, `python deploy/sync_version.py <that>`, then rebuild and `python deploy/verify_deploy.py`. Expected 1.10.7 → 1.10.10, but read the file each time rather than trusting that. Never a minor bump, never a reserved block, never a skipped number.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`. `src/services/market_data.py` **is** in scope (real unit tests); `src/pages_lib/*` is omitted (static guard + screenshots).
- Never remove an `@st.cache_data` decorator.
- No raw hex in a page or `pages_lib` module — colours come from `BloombergTheme`.
- Interval, resample rule and TTL are fixed in `market_data.py` and nowhere else. Widening through the spine's own argument (`daily_ohlc(t, period="2y")`) is legal; reaching past it is not.

---

## Measured starting state (2026-08-15, v1.10.5)

**The sync defect, measured:**

| Series | Market Overview via `fetch_data` | Every other page via the spine |
|---|---|---|
| EUR/USD daily | **66 bars** (`1d`/`3mo`), 2026-05-18 → 08-14 | **214 bars** (`300d`), 2025-10-20 → 08-14 |
| GC=F daily | **63 bars** | **207 bars** |
| Weekly | native `1wk` bars | daily **resampled** to `W` |

Last closes match (EUR/USD 1.15727 both ways), so this is not a price bug — it is an *indicator* bug. An EMA50 over 66 bars is barely warmed; over 214 it is stable. ADX, Bollinger and RSI all shift with the window. The spine guard's own docstring names this failure: *"an EMA over a private lookback is exactly how two pages come to disagree about the same instrument at the same moment."*

**Why the guard did not catch it:**

```python
# tests/test_ohlc_spine.py — _DIRECT_FETCH_PREFIXES
    "yf.download", "yf.Ticker", "yfinance.download", "yfinance.Ticker",
```

Pure-`ast` detection of *literal* yfinance calls. Market Overview calls `fetch_data(symbol, interval, period)` → `_get_provider_lazy().get_data(...)` → yfinance. Same destination, one indirection, invisible to the check. `MIGRATING` is empty and `LEGITIMATE_NON_SPINE` holds only `commodity_cot_lib.py`, so this page is **not** a sanctioned exemption — it is an undetected one.

**Blast radius of closing the hole** — files importing `data_provider` fetch helpers:

| File | Status |
|---|---|
| `src/pages_lib/market_overview_lib.py` | migrate in Task 3 |
| `pages/quant_models_tab.py` | imports only the `FRED_SERIES` **dict** (line 125), calls no fetch helper — unaffected, verified 2026-08-15 |

**What the page shows now:** `change = df["Close"].pct_change().iloc[-1] * 100` (lines 437, 462) — the last bar of the currently-selected tab. There is no month/week/day/4h comparison anywhere, and `config.timeframes` has no monthly entry. The spine has `daily_ohlc`, `weekly_ohlc`, `h4_ohlc`, `hourly_ohlc` — **no `monthly_ohlc`**.

**Definition adopted for "what changed":** for each horizon, the close of the most recent completed period against the one before it —
`Δ = last_close / previous_close - 1`. Month-over-month, week-over-week, day-over-day, 4h-over-4h. Not period-to-date, which answers a different question and cannot be compared across horizons.

---

## File structure

- **Modify** `tests/test_ohlc_spine.py` — add the indirect-fetch detection. The guard is the spec; it changes first.
- **Modify** `src/services/market_data.py` — add `monthly_ohlc()`. One new function, same shape as `weekly_ohlc`.
- **Create** `tests/test_market_data_monthly.py` — unit tests for the new spine function.
- **Create** `src/core/horizons.py` — pure change-computation, no Streamlit, no I/O. Testable and reusable by any page that later wants the same panel.
- **Create** `tests/test_horizons.py` — unit tests for the change maths.
- **Modify** `src/pages_lib/market_overview_lib.py` — `load_all_market_data()` reads the spine; new panel renders `horizons`.

---

### Task 1: Close the guard hole

The guard is the spec. If it cannot see the bypass, everything after this is unverifiable.

**Files:**
- Modify: `tests/test_ohlc_spine.py` — `_DIRECT_FETCH_PREFIXES` (line 57) and `MIGRATING` (line 55)

- [x] **Step 1: Write the failing assertion**

Add to `tests/test_ohlc_spine.py`, as its own test so the failure names the cause:

```python
def test_indirect_fetch_helpers_are_also_caught():
    """`data_provider.fetch_data` reaches yfinance too.

    The guard detected only literal `yf.download`/`yf.Ticker`, so a page could
    bypass the spine through one indirection and stay invisible. Measured
    2026-08-15: market_overview_lib pulled 66 daily bars where the spine serves
    214, and this test passed throughout.
    """
    offenders = {_rel(p) for p in _py_files() if _offends(p)}
    assert "src/pages_lib/market_overview_lib.py" in offenders, (
        "the guard no longer detects data_provider.fetch_data")
```

- [x] **Step 2: Run it to verify it fails**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ohlc_spine.py -q --no-cov`
Expected: FAIL — the offenders set does not contain the page, because `_DIRECT_FETCH_PREFIXES` does not yet include the indirect helper.

- [x] **Step 3: Confirm nothing else is caught**

Run: `grep -rn "fetch_data" pages/ src/pages_lib/ --include=*.py`
Expected: only `src/pages_lib/market_overview_lib.py`. `pages/quant_models_tab.py` imports the `FRED_SERIES` dict, not a fetch helper, so it is unaffected — but re-check rather than trust this line, since the file may have moved on.

Do **not** add `fetch_fred_series` to the tuple: FRED macro series are not OHLC and the spine does not model them.

- [x] **Step 4: Widen the detection**

`tests/test_ohlc_spine.py` — extend `_DIRECT_FETCH_PREFIXES` (line 57), with the reason:

```python
_DIRECT_FETCH_PREFIXES = (
    "yf.download", "yf.Ticker",
    "yfinance.download", "yfinance.Ticker",
    # Reaches yfinance through src/core/data_provider.py, so it bypasses the
    # spine exactly as a literal call does -- just one indirection further out.
    # `fetch_fred_series` is deliberately absent: FRED macro series are not OHLC
    # and the spine does not model them.
    "fetch_data",
)
```

- [x] **Step 5: Add the offender to `MIGRATING`, temporarily**

```python
MIGRATING: set[str] = {
    # Removed in Task 3 of docs/plans/2026-08-15-market-overview-multi-horizon.md.
    "src/pages_lib/market_overview_lib.py",
}
```

`test_migrating_list_only_names_files_that_still_offend` will delete this entry for you the moment Task 3 lands — that is the intended mechanism, not a workaround.

- [x] **Step 6: Run the guard**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_ohlc_spine.py -q --no-cov`
Expected: PASS — the new test sees the offender, and `MIGRATING` covers it. **`test_migration_is_complete` will now fail**; that is correct and expected until Task 3. If it does not fail, the exemption is not being read and Step 5 is wrong.

- [x] **Step 7: Full suite, then bump, rebuild, `verify_deploy.py`, show the diff. Do not commit.**

Expected failures: the 3 known (2 GARCH, 1 watchdog order-dependence) **plus** `test_migration_is_complete` until Task 3. Note that in the run output so the next task knows it is expected.

---

### Task 2: Give the spine a monthly window

**Files:**
- Modify: `src/services/market_data.py`
- Test: `tests/test_market_data_monthly.py`

**Interfaces:**
- Produces: `monthly_ohlc(ticker: str, period: str = MONTHLY_PERIOD) -> pd.DataFrame` and `MONTHLY_PERIOD: str = "5y"`.

- [x] **Step 1: Write the failing tests**

Create `tests/test_market_data_monthly.py`:

```python
"""Monthly bars come from the same daily series as everything else.

Pulling native `1mo` bars would give a *different* series from the daily one
every other timeframe is resampled from, which is how a monthly change can
disagree with the sum of its weeks.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.services import market_data as md


def _daily(days=400, start="2025-01-01"):
    idx = pd.date_range(start=start, periods=days, freq="D")
    return pd.DataFrame({"Open": range(days), "High": range(days),
                         "Low": range(days), "Close": range(days),
                         "Volume": [1] * days}, index=idx)


class TestMonthlyOhlc:
    def test_resamples_from_daily_not_native_monthly(self, monkeypatch):
        seen = {}

        def fake(ticker, **kw):
            seen.update(kw)
            return _daily()

        monkeypatch.setattr(md, "cached_ohlc", fake, raising=False)
        md.monthly_ohlc("EURUSD=X")
        assert seen.get("interval") == "1d", "must fetch daily and resample"

    def test_returns_monthly_bars(self, monkeypatch):
        monkeypatch.setattr(md, "cached_ohlc", lambda t, **kw: _daily(), raising=False)
        out = md.monthly_ohlc("EURUSD=X")
        # ~13 months in 400 days
        assert 12 <= len(out) <= 14
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_ohlc_aggregation_is_correct(self, monkeypatch):
        monkeypatch.setattr(md, "cached_ohlc", lambda t, **kw: _daily(days=62,
                            start="2025-01-01"), raising=False)
        out = md.monthly_ohlc("EURUSD=X")
        january = out.iloc[0]
        assert january["Open"] == 0        # first daily open of the month
        assert january["Close"] == 30      # last daily close of January
        assert january["High"] == 30       # max over the month
        assert january["Low"] == 0         # min over the month

    def test_empty_input_degrades_to_empty(self, monkeypatch):
        monkeypatch.setattr(md, "cached_ohlc",
                            lambda t, **kw: pd.DataFrame(), raising=False)
        assert md.monthly_ohlc("EURUSD=X").empty

    def test_failure_never_raises(self, monkeypatch):
        def boom(t, **kw):
            raise RuntimeError("network")
        monkeypatch.setattr(md, "cached_ohlc", boom, raising=False)
        assert md.monthly_ohlc("EURUSD=X").empty
```

- [x] **Step 2: Run to verify they fail** — `AttributeError: module 'src.services.market_data' has no attribute 'monthly_ohlc'`.

- [x] **Step 3: Implement, mirroring `weekly_ohlc` exactly**

In `src/services/market_data.py`, after `weekly_ohlc`:

```python
MONTHLY_PERIOD = "5y"    # ~60 monthly bars — enough for a 12-month lookback


def monthly_ohlc(ticker: str, period: str = MONTHLY_PERIOD) -> pd.DataFrame:
    """Canonical monthly bars: daily bars resampled to month-end (default 5y).

    Resampled from daily rather than pulled as native ``1mo`` bars, the same
    convention ``weekly_ohlc`` follows. Native monthly bars are a *different*
    series from the daily one every other timeframe derives from, which is how
    a monthly change comes to disagree with the weeks inside it. The in-progress
    month is included as a partial bar, identically for every page.
    """
    from src.db.market_cache import cached_ohlc
    try:
        df = _ohlcv(cached_ohlc(ticker, period=period, interval="1d",
                                ttl=CANONICAL_TTL))
        if df.empty:
            return df
        return _resample(df, "ME")
    except Exception:
        return pd.DataFrame()
```

Check `_resample`'s signature before writing: it takes a pandas offset alias. Use `"ME"` (month-end) — `"M"` is removed in pandas 3.0.

- [x] **Step 4: Run the tests** — expect PASS.
- [x] **Step 5: Prove it against a real ticker**

```bash
PYTHONIOENCODING=utf-8 python -c "from src.services.market_data import monthly_ohlc; d=monthly_ohlc('EURUSD=X'); print(len(d), d.index.min().date(), d.index.max().date()); print(d.tail(3))"
```
Expected: ~60 bars, month-end index, last bar the in-progress month.

- [x] **Step 6: Full suite, then bump, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 3: Migrate Market Overview onto the spine

**Files:**
- Modify: `src/pages_lib/market_overview_lib.py` — `load_all_market_data()` (line 279) and the `fetch_data` import (line 35)
- Modify: `tests/test_ohlc_spine.py` — remove the `MIGRATING` entry

- [x] **Step 1: Map each config timeframe to its spine call**

| `config.timeframes` key | Today | Spine call |
|---|---|---|
| Weekly | `1wk` / `2y` | `weekly_ohlc(symbol)` |
| Daily | `1d` / `3mo` | `daily_ohlc(symbol)` |
| 4 Hour | `4h` / `1mo` | `h4_ohlc(symbol)` |
| Hourly | `1h` / `1mo` | `hourly_ohlc(symbol)` |
| 15 Minute | `15m` / `5d` | **none exists** |

**The 15-minute row is the open question.** The spine has no 15m function. Check `src/services/market_data.py` again at implementation time — if none exists, keep the 15-minute tab on `cached_ohlc(symbol, period="5d", interval="15m", ttl=CANONICAL_TTL)`, which is the spine's own cache rather than `fetch_data`, and note it in the module docstring. Do **not** invent a `fifteen_min_ohlc` in this task; that is a spine change and belongs in its own plan.

- [x] **Step 2: Rewrite the fetcher**

Replace the `_fetch` closure inside `load_all_market_data`. Keep `run_parallel` — the fan-out is still worth it — and keep the `@st.cache_data` decorator.

```python
    from src.services.market_data import (daily_ohlc, h4_ohlc, hourly_ohlc,
                                          weekly_ohlc)

    _SPINE = {
        "Weekly": weekly_ohlc,
        "Daily": daily_ohlc,
        "4 Hour": h4_ohlc,
        "Hourly": hourly_ohlc,
    }

        def _fetch(item):
            tf_name, _tf_cfg, _pair_name, symbol = item
            fn = _SPINE.get(tf_name)
            if fn is not None:
                return fn(symbol)
            # 15-minute: no spine function yet, but go through the spine's own
            # cache rather than data_provider so the TTL still matches.
            from src.db.market_cache import cached_ohlc
            from src.services.market_data import CANONICAL_TTL
            return cached_ohlc(symbol, period="5d", interval="15m",
                               ttl=CANONICAL_TTL)
```

- [x] **Step 3: Remove the now-dead import** of `fetch_data` at line 35 if nothing else in the file uses it. Run `grep -n "fetch_data" src/pages_lib/market_overview_lib.py` and remove only if the count reaches zero; `fetch_fred_series` stays.

- [x] **Step 4: Delete the `MIGRATING` entry** in `tests/test_ohlc_spine.py`, restoring `MIGRATING: set[str] = set()`.

- [x] **Step 5: Run the guard** — `test_migration_is_complete` must now PASS, and the Task 1 test must FAIL (the page no longer offends). **Update that test**: it asserted the offender was detected, which was only true while the page offended. Change it to prove the *detection* still works without depending on a specific offender:

```python
def test_indirect_fetch_helpers_are_also_caught():
    assert "fetch_data" in _DIRECT_FETCH_PREFIXES, (
        "the guard must still detect data_provider.fetch_data; "
        "market_overview_lib bypassed the spine through it for months")
```

- [x] **Step 6: Prove the numbers moved**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src.pages_lib.market_overview_lib import load_all_market_data
d = load_all_market_data()
for tf in ('Daily','Weekly'):
    df = d[tf]['EUR/USD']; print(tf, len(df), df.index.min().date(), df.index.max().date())"
```
Expected: Daily ~214 bars (was 66), matching `daily_ohlc('EURUSD=X')` exactly.

- [x] **Step 7: Full suite, then bump, rebuild, `verify_deploy.py`, screenshot `/market-overview`, show the diff.**

---

### Task 4: The four-horizon change panel

**Files:**
- Create: `src/core/horizons.py`, `tests/test_horizons.py`
- Modify: `src/pages_lib/market_overview_lib.py` — add the panel

**Interfaces:**
- Produces: `period_change(df: pd.DataFrame) -> float | None` — last completed period vs the one before, as a percentage.
- Produces: `horizon_row(pair: str, frames: dict[str, pd.DataFrame]) -> dict` — `{"Pair", "4H %", "1D %", "1W %", "1M %"}`.

- [x] **Step 1: Write the failing tests**

Create `tests/test_horizons.py`:

```python
"""Change over a horizon, computed identically for every timeframe."""
from __future__ import annotations

import pandas as pd
import pytest

from src.core import horizons as hz


def _closes(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"Close": values}, index=idx)


class TestPeriodChange:
    def test_rise_is_positive(self):
        assert hz.period_change(_closes([100.0, 110.0])) == pytest.approx(10.0)

    def test_fall_is_negative(self):
        assert hz.period_change(_closes([100.0, 90.0])) == pytest.approx(-10.0)

    def test_uses_only_the_last_two_closes(self):
        assert hz.period_change(_closes([1.0, 50.0, 100.0, 110.0])) == pytest.approx(10.0)

    def test_one_bar_has_no_previous_period(self):
        assert hz.period_change(_closes([100.0])) is None

    def test_empty_and_none_are_none(self):
        assert hz.period_change(pd.DataFrame()) is None
        assert hz.period_change(None) is None

    def test_zero_previous_close_does_not_divide_by_zero(self):
        assert hz.period_change(_closes([0.0, 100.0])) is None

    def test_nan_close_is_none_not_nan(self):
        # NaN would render as "nan%" and, if ever persisted, is invalid JSONB.
        assert hz.period_change(_closes([float("nan"), 100.0])) is None


class TestHorizonRow:
    def test_builds_all_four_columns(self):
        frames = {"4 Hour": _closes([100.0, 101.0]),
                  "Daily": _closes([100.0, 102.0]),
                  "Weekly": _closes([100.0, 105.0]),
                  "Monthly": _closes([100.0, 110.0])}
        row = hz.horizon_row("EUR/USD", frames)
        assert row["Pair"] == "EUR/USD"
        assert row["4H %"] == pytest.approx(1.0)
        assert row["1D %"] == pytest.approx(2.0)
        assert row["1W %"] == pytest.approx(5.0)
        assert row["1M %"] == pytest.approx(10.0)

    def test_a_missing_timeframe_is_none_not_an_exception(self):
        row = hz.horizon_row("EUR/USD", {"Daily": _closes([100.0, 102.0])})
        assert row["1D %"] == pytest.approx(2.0)
        assert row["1M %"] is None
```

- [x] **Step 2: Run to verify they fail** — no module `src.core.horizons`.

- [x] **Step 3: Implement `src/core/horizons.py`**

```python
"""Percentage change over a horizon, computed the same way for every timeframe.

The page previously reported `Close.pct_change().iloc[-1]` on whichever tab was
open, so "change" meant a different span depending on where you were standing.
These functions take the frames the spine already produces -- all resampled from
one daily series -- so a 1W figure and the 1D figures inside it are arithmetically
consistent with each other.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

# Column label -> the `config.timeframes` key whose frame supplies it.
HORIZONS = (("4H %", "4 Hour"), ("1D %", "Daily"),
            ("1W %", "Weekly"), ("1M %", "Monthly"))


def period_change(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Last close vs the previous one, in percent. None when unanswerable."""
    if df is None or not hasattr(df, "empty") or df.empty or "Close" not in df:
        return None
    closes = df["Close"].dropna()
    if len(closes) < 2:
        return None
    previous, latest = float(closes.iloc[-2]), float(closes.iloc[-1])
    if not previous or not math.isfinite(previous) or not math.isfinite(latest):
        return None
    return (latest / previous - 1.0) * 100.0


def horizon_row(pair: str, frames: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """One table row: the pair, then its change over each horizon."""
    row: Dict[str, Any] = {"Pair": pair}
    for label, timeframe in HORIZONS:
        row[label] = period_change(frames.get(timeframe))
    return row
```

- [x] **Step 4: Run the tests** — expect PASS.

- [x] **Step 5: Load the monthly frame in `load_all_market_data`**

Add `"Monthly": monthly_ohlc` to the `_SPINE` map from Task 3, and add a `"Monthly"` key to the `data` dict it builds. Do **not** add Monthly to `config.timeframes` — that dict drives the page's tab list, and this horizon feeds the panel, not a tab. If a Monthly tab is wanted later, that is a separate change.

- [x] **Step 6: Render the panel**

Above the existing per-timeframe tabs, so it is the first thing read. Follow the page's existing table idiom; use `BloombergTheme` colours, no raw hex.

```python
    import streamlit as st
    from src.core.horizons import HORIZONS, horizon_row

    st.markdown("### What changed")
    rows = [horizon_row(pair, {tf: data.get(tf, {}).get(pair)
                               for tf, _ in ((t, l) for l, t in HORIZONS)})
            for pair in config.assets]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("Each column is the latest completed period against the one "
               "before it. All four are resampled from the same daily series, "
               "so they agree with each other and with every other page.")
```

Sort by absolute 1D change, biggest mover first — the panel's job is to show *what moved*, and alphabetical order buries that.

- [x] **Step 7: Full suite.**
- [x] **Step 8: Screenshot `/market-overview`** at 1280×800, light and dark. Confirm the panel reads as part of the terminal (see `docs/plans/2026-08-15-terminal-ui.md` — if that plan has landed, use `readout()` rather than a bare `st.dataframe`).
- [x] **Step 9: Bump, rebuild, `verify_deploy.py`, show the diff. Do not commit.**

---

## Out of scope, deliberately

- **A 15-minute spine function.** Task 3 keeps that tab on `cached_ohlc` with the canonical TTL. Adding `fifteen_min_ohlc` changes the spine's public surface and deserves its own plan.
- **A Monthly tab.** The monthly frame feeds the panel only. Adding a tab means new chart/indicator code for a timeframe nothing else uses yet.
- **`config.timeframes` restructuring.** It drives several pages; changing its shape here would widen the blast radius well past Market Overview.
- **Period-to-date changes** (month-to-date, week-to-date). A different question, and mixing the two in one table makes both unreadable. Worth adding later as a toggle.

## Verification for the whole plan

- [x] `tests/test_ohlc_spine.py` green with `MIGRATING == set()` — the page is on the spine and the guard can see indirect bypasses.
- [x] `tests/test_market_data_monthly.py` and `tests/test_horizons.py` — all pass.
- [x] Market Overview's Daily frame for EUR/USD is ~214 bars and identical to `daily_ohlc('EURUSD=X')`.
- [x] Full suite: no failures beyond the 3 known.
- [x] `verify_deploy.py` in sync at the final version.
- [x] Screenshot of the "What changed" panel, light and dark.

---

Module map: [[Architecture]] · Docs index: [[README]]
