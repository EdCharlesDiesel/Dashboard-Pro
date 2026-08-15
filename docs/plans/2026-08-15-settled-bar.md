# Settled Bars: Don't Trust a Bar That Just Closed — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Stop `biased_pivots` reading a bar the data provider is still revising. Judging the last *closed* bar (rather than the forming one) removed 23 of 25 same-bar contradictions — but 2 survived, and both happened within two hours of the FX day break, with the **same `bar_time` and a different close**. A bar that closed minutes ago is closed by index, not settled in fact. Require it to have been closed for a settling period before trusting it, and fall back to the bar before it when it has not.

**Architecture:** One pure helper, `settled_period_index(df, settle_hours, now)`, in `src/core/biased_pivots.py` next to `read()`. It returns the positional index of the newest bar old enough to trust — normally `-2`, and `-3` when `-2` closed too recently. `read()` uses it for the period, the close, and `bar_time`, so all three stay on the same bar. No I/O, no new dependency, and the frame stays the only input, which keeps `read()` unit-testable exactly as it is now.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), pandas, pytest, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — "Plan first" and the version rule. The defect evidence is in this plan's *Measured starting state*.

## Global Constraints

- Never commit. Make changes only; the repository owner reviews and commits.
- **A plan gets its own bump too.** `VERSION` read **1.10.3**, so this plan takes **1.10.4** — the patch, plus one. Task 1 will take whatever `VERSION` reads at the moment it completes, plus one (expected **1.10.5**). Never a minor bump, never a reserved block, never a skipped number.
- **Bump the version on every completed task** (`python deploy/sync_version.py <next>`), then rebuild and `python deploy/verify_deploy.py` before calling a task done.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`. `src/core/` **is** in scope for coverage, so this task needs real unit tests, not a static guard.
- Never remove an `@st.cache_data` decorator.
- Do not change the pivot maths. `levels()` and the zone logic stay exactly as they are.

---

## Measured starting state (2026-08-15, v1.10.3)

Same-bar contradictions for `biased_pivots` — one instrument, one `bar_time`, both directions persisted:

| window | count |
|---|---|
| all time | 25 |
| since the closed-bar fix (v1.7.0, 2026-08-11 18:03) | **2** |

The two survivors, with their stored entry prices:

| pair | `bar_time` | first read | second read |
|---|---|---|---|
| AUD/ZAR | 2026-08-13 | 21:22:04 → 11.42590 **Short** | 23:12:03 → 11.39812 **Long** |
| USD/ZAR | 2026-08-13 | 21:12:05 → 16.19230 **Short** | 22:02:07 → 16.13787 **Long** |

Both reads name the same bar and disagree about its close. `FX_DAY_BREAK_HOUR = 21` (`src/core/volume_profile.py:49`), and both pairs are ZAR crosses. The reading: shortly after the daily rollover, yfinance has dated a new bar — making yesterday's `iloc[-2]` — while still revising that bar's close. `read()` is deterministic given a frame; the frame itself changed underneath it.

**Confidence:** n = 2. The timing is consistent and the mechanism is plausible, but this is a hypothesis, not an established fact. Step 1 of Task 1 tests the *behaviour* (a bar too fresh must not be used), which holds whatever the provider's reason. Step 9 checks whether the contradictions actually stop.

Current structure of `read()` (`src/core/biased_pivots.py`):

```python
    if df is None or df.empty or len(df) < 3:      # line 116
        return None
    period = df.iloc[-2]                            # line 121
    earlier = df.iloc[-3]                           # line 124
    ...
        bar_time=df.index[-2] if isinstance(df.index, pd.DatetimeIndex) else None,   # line 151
```

**Why 6 hours:** the daily rollover happens once every 24, and the observed revisions were 50–110 minutes after it. Six hours clears that comfortably while still letting the first sweep after a European morning use yesterday's bar. On weekly bars — which `biased_pivots` now reads after v1.10.3 — the exposure window is a few hours out of 168, so this mostly matters for any source still on daily.

---

## File structure

- **Modify** `src/core/biased_pivots.py` — add `settled_period_index()`, use it in `read()`. One responsibility: decide which bar is old enough to trust, then read that one bar consistently.
- **Test** `tests/test_biased_pivots_settled.py` — new file. Unit tests for the helper and for `read()` honouring it.
- **Unchanged:** `src/pages_lib/biased_pivots_page.py`. The page passes a frame and gets a read; nothing about its contract moves.

---

### Task 1: Require a bar to be settled before trusting it

**Files:**
- Modify: `src/core/biased_pivots.py` — new helper above `read()`; `read()` at lines 116, 121, 124, 151
- Test: `tests/test_biased_pivots_settled.py`

**Interfaces:**
- Produces: `settled_period_index(df: pd.DataFrame, settle_hours: float = SETTLE_HOURS, now: datetime | None = None) -> int | None` — the positional index of the newest trustworthy bar (`-2` or `-3`), or `None` when the frame is too short. Exported from `src.core.biased_pivots`.
- Produces: module constant `SETTLE_HOURS: float = 6.0`.
- Consumes: nothing new.

- [x] **Step 1: Write the failing tests**

Create `tests/test_biased_pivots_settled.py`:

```python
"""A bar that closed minutes ago is closed by index, not settled in fact.

Judging the last closed bar removed 23 of 25 same-bar contradictions. The two
survivors named the same bar_time and disagreed about its close -- AUD/ZAR at
11.42590 then 11.39812, USD/ZAR at 16.19230 then 16.13787 -- both within two
hours of the 21:00 UTC FX day break. `read()` is deterministic given a frame;
the frame changed underneath it while the provider revised a just-closed bar.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.core import biased_pivots as bp


def _frame(rows, freq="D", end=None):
    """rows: list of (high, low, close). Last row is the forming bar."""
    end = end or datetime(2026, 8, 14)
    idx = pd.date_range(end=end, periods=len(rows), freq=freq)
    return pd.DataFrame(
        {"High": [r[0] for r in rows], "Low": [r[1] for r in rows],
         "Close": [r[2] for r in rows], "Open": [r[2] for r in rows]},
        index=idx)


class TestSettledPeriodIndex:
    def test_an_old_closed_bar_is_used(self):
        df = _frame([(1, 1, 50.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])
        # forming bar opened 2026-08-14; a week later it is long settled
        now = datetime(2026, 8, 21)
        assert bp.settled_period_index(df, now=now) == -2

    def test_a_freshly_closed_bar_is_skipped(self):
        df = _frame([(1, 1, 50.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])
        # the forming bar opened only an hour ago, so -2 closed an hour ago
        now = datetime(2026, 8, 14, 1, 0)
        assert bp.settled_period_index(df, now=now) == -3

    def test_the_boundary_is_inclusive(self):
        df = _frame([(1, 1, 50.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])
        exactly = datetime(2026, 8, 14) + timedelta(hours=bp.SETTLE_HOURS)
        assert bp.settled_period_index(df, now=exactly) == -2

    def test_too_short_to_step_back_returns_none(self):
        # Falling back needs a 4th bar for the shft+1 close.
        df = _frame([(1, 1, 50.0), (110.0, 90.0, 95.0), (1, 1, 0.0)])
        now = datetime(2026, 8, 14, 1, 0)
        assert bp.settled_period_index(df, now=now) == -3
        short = _frame([(110.0, 90.0, 95.0), (1, 1, 0.0)])
        assert bp.settled_period_index(short, now=now) is None

    def test_a_non_datetime_index_cannot_be_aged_so_is_trusted(self):
        # No clock means no evidence of freshness; refusing to read would be a
        # worse failure than reading a bar that might be fresh.
        df = pd.DataFrame({"High": [1, 110.0, 1], "Low": [1, 90.0, 1],
                           "Close": [50.0, 95.0, 0.0], "Open": [0, 0, 0]})
        assert bp.settled_period_index(df) == -2


class TestReadHonoursIt:
    def _four_bars(self):
        # -4 close 40 (shft+1 for the fallback), -3 is the settled period,
        # -2 is the fresh one, -1 forming.
        return _frame([(1, 1, 40.0), (210.0, 190.0, 195.0),
                       (110.0, 90.0, 95.0), (1, 1, 0.0)])

    def test_read_uses_the_fresh_bar_when_it_is_old_enough(self):
        r = bp.read(self._four_bars(), now=datetime(2026, 8, 21))
        assert r is not None
        assert r.price == pytest.approx(95.0)
        assert r.pp == pytest.approx((110.0 + 90.0 + 195.0) / 3.0)

    def test_read_falls_back_when_the_bar_is_too_fresh(self):
        r = bp.read(self._four_bars(), now=datetime(2026, 8, 14, 1, 0))
        assert r is not None
        assert r.price == pytest.approx(195.0)          # the older bar's close
        assert r.pp == pytest.approx((210.0 + 190.0 + 40.0) / 3.0)

    def test_bar_time_names_the_bar_actually_read(self):
        df = self._four_bars()
        fresh = bp.read(df, now=datetime(2026, 8, 14, 1, 0))
        assert fresh.bar_time == df.index[-3]
        settled = bp.read(df, now=datetime(2026, 8, 21))
        assert settled.bar_time == df.index[-2]

    def test_two_reads_of_one_frame_at_different_clock_times_agree(self):
        # The regression, in one test: the same frame read 50 minutes apart --
        # the gap that produced the USD/ZAR flip -- must give the same answer,
        # provided both are outside the settling window.
        df = self._four_bars()
        a = bp.read(df, now=datetime(2026, 8, 21, 9, 0))
        b = bp.read(df, now=datetime(2026, 8, 21, 9, 50))
        assert (a.direction, a.price, a.bar_time) == (b.direction, b.price, b.bar_time)
```

- [x] **Step 2: Run to verify they fail**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_biased_pivots_settled.py -q --no-cov`
Expected: FAIL — `AttributeError: module 'src.core.biased_pivots' has no attribute 'settled_period_index'`, and `read()` rejecting the `now=` keyword.

- [x] **Step 3: Add the constant and the helper**

In `src/core/biased_pivots.py`, above `read()`:

```python
# How long a bar must have been closed before its OHLC is trusted.
#
# A bar is closed by *index* the moment the provider dates the next one, but its
# close can still be revised for a while afterwards. Measured 2026-08-13: two
# reads of the same daily bar 50 and 110 minutes after the 21:00 UTC FX day
# break returned different closes and therefore opposite pivot zones -- AUD/ZAR
# 11.42590 then 11.39812, USD/ZAR 16.19230 then 16.13787. Six hours clears the
# observed revision window with room to spare while still letting a European
# morning sweep use yesterday's bar.
SETTLE_HOURS: float = 6.0


def settled_period_index(df: pd.DataFrame, settle_hours: float = SETTLE_HOURS,
                         now: Optional[datetime] = None) -> Optional[int]:
    """Positional index of the newest bar old enough to trust: -2, or -3.

    The forming bar's own timestamp is when the previous bar closed, so the age
    of ``df.index[-1]`` is the age of the close at ``-2``. When that is inside
    the settling window, step back one bar; the caller still needs one more for
    the indicator's ``shft+1`` close, hence the length check.

    Returns ``None`` when the frame is too short to satisfy the choice made.
    """
    if df is None or df.empty or len(df) < 3:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        return -2          # no clock, so no evidence of freshness
    closed_at = df.index[-1]
    if getattr(closed_at, "tzinfo", None) is not None:
        closed_at = closed_at.tz_convert("UTC").tz_localize(None)
    reference = now or datetime.utcnow()
    if getattr(reference, "tzinfo", None) is not None:
        reference = reference.astimezone(timezone.utc).replace(tzinfo=None)
    age_hours = (reference - closed_at.to_pydatetime()).total_seconds() / 3600.0
    if age_hours >= settle_hours:
        return -2
    return -3 if len(df) >= 4 else None
```

**Imports:** the module has `from typing import Any, Dict, Optional` (line 40) and `import pandas as pd` (line 42), but **no datetime import**. Add exactly this above the pandas import:

```python
from datetime import datetime, timezone
```

- [x] **Step 4: Use it in `read()`**

Change the signature and the three reads. `read()` currently hardcodes `-2`/`-3`:

```python
def read(df: pd.DataFrame, zone_frac: float = DEFAULT_ZONE_FRAC,
         now: Optional[datetime] = None) -> Optional[Pivots]:
```

Replace the guard and the two row picks:

```python
    idx = settled_period_index(df, now=now)
    if idx is None:
        return None
    period = df.iloc[idx]          # newest settled period -> high/low, and the
                                   # close being judged (same bar it is stamped
                                   # with, so the read cannot drift intrabar)
    earlier = df.iloc[idx - 1]     # one older -> close (the source's shft+1)
    price = float(period["Close"])
```

and the stamp:

```python
        bar_time=df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else None,
```

Delete the now-redundant `len(df) < 3` guard at line 116 — `settled_period_index` owns that check and returns `None`.

- [x] **Step 5: Run the new tests**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_biased_pivots_settled.py -q --no-cov`
Expected: PASS.

- [x] **Step 6: Run the existing pivot tests**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_biased_pivots.py tests/test_biased_pivots_horizon.py -q --no-cov`
Expected: PASS, unchanged. These build 3-bar frames dated 2024/2026 with no `now=`, so real "now" is years later and every bar is settled — the `-2` path, same as today. **If any fail, the maths moved and Step 4 is wrong.**

- [x] **Step 7: Full suite**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: green apart from the 3 known failures — 2 GARCH (no `arch` wheel for Python 3.14) and `test_mt5_watchdog.py::TestIsolation::test_watchdog_never_imports_the_broker_module` (pre-existing order dependence; `tests/test_mt5_mcp.py` leaves `MetaTrader5` in `sys.modules`). Confirm the count does not grow.

- [x] **Step 8: Bump, rebuild, verify**

```bash
python deploy/sync_version.py <VERSION + 1>     # read VERSION first; expected 1.10.5
docker compose build app && docker compose up -d
python deploy/verify_deploy.py
```
Expected: `in sync`, all four containers on the new tag.

- [x] **Step 9: Prove it against the live board**

```bash
docker exec dashboard-pro-sweeper-1 python -m src.services.signal_sweep --only biased_pivots
```

Then re-run the contradiction check and record the number in this plan:

```sql
SELECT source, count(*) FROM (
  SELECT source, instrument, checks_detail->>'bar_time' bt
  FROM trade_setups WHERE logged_at > '<deploy timestamp>'
    AND checks_detail->>'bar_time' IS NOT NULL
  GROUP BY 1,2,3 HAVING count(DISTINCT direction) > 1) x
GROUP BY source;
```

Expected: no `biased_pivots` rows. **This is a weak test on the day it runs** — the sweep saves nothing at a weekend and the settling window only bites near a rollover. The real check is the count after the next few daily rollovers; note the date it was taken so the next session can compare rather than re-derive.

- [x] **Step 10: Show the owner the diff. Do not commit.**

---

## Out of scope, deliberately

- **Other sources that read `iloc[-1]` or `iloc[-2]`.** `market_structure` was moved to the closed bar in the same pass that produced this diagnosis, and the rest have not been audited. A repo-wide sweep for fresh-bar reads is its own plan; do not widen this one.
- **The 25 historical same-bar contradictions.** They stay in `trade_setups` and expire with their horizons. Rewriting stored history would destroy the Source Scorecard's evidence.
- **Tuning `SETTLE_HOURS` per timeframe.** Six hours is one number covering daily and weekly. If a 15-minute source ever needs this, it needs a period-relative window, and that is a change to make when there is a caller for it.

## Verification for the whole plan

- [x] `tests/test_biased_pivots_settled.py` — all pass.
- [x] `tests/test_biased_pivots.py` and `tests/test_biased_pivots_horizon.py` — unchanged and green.
- [x] Full suite: no new failures beyond the 3 known.
- [x] `verify_deploy.py` in sync at the new version.
- [x] Same-bar contradiction count for `biased_pivots` recorded, with its date, at the bottom of this plan.

---

Module map: [[Architecture]] · Docs index: [[README]]

---

## Result (recorded 2026-08-15, v1.10.5)

| Window | Same-bar contradictions |
|---|---|
| all time | biased_pivots 25 |
| since the closed-bar fix (v1.7.0) | biased_pivots **2**, predictive 1 |
| **since this fix (v1.10.5)** | **NONE** |

7 `biased_pivots` rows written by the verifying sweep, none contradicting.

**This is a weak measurement and should be read as such.** It is a Saturday: no
new bars are forming, so the settling window was never actually exercised. Zero
here means "nothing regressed", not "the fix works". The real check is the count
after the next few daily rollovers — re-run the Step 9 query with
`since = '2026-08-15 17:20:00'` and compare against this row.
