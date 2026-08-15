# Biased Pivots: Match the Horizon to the Period — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Stop `biased_pivots` fighting the swing system. It reads **daily** pivots — levels that are recomputed the moment a new daily bar closes — and persists them with a **5-trading-day** horizon. The signal outlives the levels it was derived from by four days. Move the pivot period to **weekly** so the 5-day horizon is honest, and the page becomes the swing construct the rest of the system already assumes it is.

**Architecture:** No maths changes. `src/core/biased_pivots.read()` is period-agnostic — it takes any OHLC frame, uses `iloc[-2]` as the pivot period and `iloc[-3]` for the close (the indicator's `shft`/`shft+1` quirk). Feeding it `weekly_ohlc()` instead of `daily_ohlc()` produces weekly pivots with no change to `src/core/`. The page declares its period explicitly and derives the horizon from it, so the two can never drift apart again. The MQL5 original is already period-parameterised (`timePeriodConverter()`); this is the Python port catching up, not a new invention.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), Streamlit, Postgres 18, pytest + pure `ast` for the static guard, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — the "Plan first" working agreement and the `market_data.py` single-source-of-truth rule. Evidence for the defect is in this plan's *Measured starting state*.

## Global Constraints

- Never commit. Make changes only; the repo owner reviews and commits.
- **A plan gets its own bump too.** VERSION was 1.10.1, so this plan takes **1.10.2** — patch, plus one. Always plus one, sequentially, from whatever VERSION currently reads. Not a minor bump because the change "feels" bigger, and never a reserved block of numbers.
- Task 1 lands on **1.10.3**.
- **Bump the version on every completed task** (`python deploy/sync_version.py <next>`), then rebuild and `python deploy/verify_deploy.py` before calling a task done.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`. `src/pages_lib/*` is omitted from coverage, so this task is guarded by a **static `ast` test**, not unit coverage.
- Never remove an `@st.cache_data` decorator.

---

## Measured starting state (2026-08-15, v1.10.1)

`src/pages_lib/biased_pivots_page.py`:

```python
from src.services.market_data import daily_ohlc          # line 22
_HORIZON_DAYS = 5   # pivot zones are a day-trade construct, not a swing one   # line 26
        r: Optional[Pivots] = read(daily_ohlc(inst.ticker))                    # line 50
                "horizon_days": _HORIZON_DAYS,                                 # line 112
```

The comment concedes the mismatch in writing. Measured facts around it:

| Fact | Value |
|---|---|
| Pivot period actually read | daily (`daily_ohlc`) |
| Horizon persisted with every signal | 5 trading days |
| How long daily pivot levels stay valid | **1 period — they move at the next daily close** |
| Signals `biased_pivots` wrote in the last 7 days | 130 at a 5-day horizon |
| Instruments it contradicted itself on (before the closed-bar fix) | 20 of 27 — the worst of any source |
| Shortest horizon anywhere else in the system | 3 trading days (`market_structure` on 4H) |

The repaint half of this bug is **already fixed** (v1.7.0): `read()` now judges the closed bar's close rather than the forming bar's live price, so one bar yields one answer. What remains is the shelf life.

**Why weekly and not "horizon = 1 day":** a 1-day horizon would also be internally consistent, and would make this the only intraday source in a system whose spine holds no intraday windows, whose stops are 1.5× *daily* ATR, and whose dedupe key admits one signal per pair per session-day. Weekly pivots keep the indicator and make it fit. See `docs/System_Guide.md` for the swing-system evidence.

**Note on files already on disk:** an aborted attempt (no plan — the reason this plan exists) left `tests/test_biased_pivots_horizon.py` written and `biased_pivots_page.py` reverted to its original state. Verify that revert before starting: `git diff src/pages_lib/biased_pivots_page.py` must be empty.

---

## File structure

- **Modify** `src/pages_lib/biased_pivots_page.py` — the import, the two module constants, the `_scan` call, and the sidebar caption. One responsibility: declare the period, read that period, publish a horizon derived from it.
- **Test** `tests/test_biased_pivots_horizon.py` — static `ast` guard that the declared period, the horizon, and the frame actually read all agree.
- **Unchanged:** `src/core/biased_pivots.py`. The maths is period-agnostic and stays that way.

---

### Task 1: Weekly pivots, horizon derived from the period

**Files:**
- Modify: `src/pages_lib/biased_pivots_page.py:22,26,50` and the sidebar caption near `:36-41`
- Test: `tests/test_biased_pivots_horizon.py`

**Interfaces:**
- Consumes: `src.services.market_data.weekly_ohlc(ticker: str, period: str = "2y") -> pd.DataFrame` — daily bars resampled to `W`, ~104 rows, already used by `weekly_ema` and `weekly_swing`.
- Produces: module constants `_PIVOT_PERIOD: str` and `_HORIZON_DAYS: int` on `src.pages_lib.biased_pivots_page`, read by the guard test and by `persist_signals`.

- [x] **Step 1: Write the failing test**

`tests/test_biased_pivots_horizon.py` (already on disk from the aborted attempt — read it, confirm it matches, do not rewrite it blind):

```python
_TRADING_DAYS_PER = {"weekly": 5, "daily": 1, "monthly": 21}


class TestHorizonMatchesPeriod:
    def test_the_period_is_declared(self):
        assert hasattr(page, "_PIVOT_PERIOD")
        assert page._PIVOT_PERIOD in _TRADING_DAYS_PER

    def test_horizon_equals_one_period(self):
        assert page._HORIZON_DAYS == _TRADING_DAYS_PER[page._PIVOT_PERIOD]

    def test_it_is_not_a_day_trade_construct(self):
        assert page._PIVOT_PERIOD != "daily"
        assert page._HORIZON_DAYS >= 5

    def test_the_scan_reads_the_declared_period(self):
        source = inspect.getsource(page.BiasedPivotsPage._scan)
        calls = {n.func.id for n in ast.walk(ast.parse(source.strip()))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "weekly_ohlc" in calls
        assert "daily_ohlc" not in calls
```

The fourth test is the one that matters: a declared period the scanner ignores is worse than no declaration, because the horizon would look justified while the levels came from somewhere else.

- [x] **Step 2: Run it to verify it fails**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_biased_pivots_horizon.py -q --no-cov`
Expected: 4 failures in `TestHorizonMatchesPeriod` — `AttributeError`-style failure on `_PIVOT_PERIOD` first. `TestStillReadsCorrectlyOnWeeklyBars` should already **pass**: the maths is period-agnostic and that class proves it.

- [x] **Step 3: Change the import**

`src/pages_lib/biased_pivots_page.py:22`

```python
from src.services.market_data import weekly_ohlc
```

- [x] **Step 4: Declare the period and derive the horizon**

Replace line 26:

```python
# Pivot levels are recomputed every period: a bar closes and PP/S1/R1/M2/M3 all
# move. A signal derived from them is therefore valid for about ONE period and
# no longer. This page used daily pivots with a 5-day horizon, so every signal
# outlived its own levels by four days -- and the comment here conceded it:
# "pivot zones are a day-trade construct, not a swing one".
#
# Weekly pivots resolve it without touching the maths. `read()` is
# period-agnostic, and the MQL5 original is period-parameterised
# (`timePeriodConverter()`), so this is the port catching up. The horizon is
# derived from the period rather than written next to it, because two numbers
# that must agree will not stay in agreement if a human maintains both.
_PIVOT_PERIOD = "weekly"
_TRADING_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}
_HORIZON_DAYS = _TRADING_DAYS[_PIVOT_PERIOD]
```

- [x] **Step 5: Read the declared period in `_scan`**

`src/pages_lib/biased_pivots_page.py:50`

```python
                r: Optional[Pivots] = read(weekly_ohlc(inst.ticker))
```

- [x] **Step 6: Tell the user what they are looking at**

The sidebar still says these are the chart's levels without saying which period. Add one caption after the existing `⚠` line:

```python
        st.caption("Levels come from the last **completed week**, so a read is "
                   "good for the week ahead. Daily pivots move every night — "
                   "too fast for a 5-day swing horizon.")
```

- [x] **Step 7: Run the tests**

Run: `PYTHONIOENCODING=utf-8 python -m pytest tests/test_biased_pivots_horizon.py tests/test_biased_pivots.py -q --no-cov`
Expected: all PASS. `tests/test_biased_pivots.py` must stay green untouched — if it fails, the maths was changed, which this task must not do.

- [x] **Step 8: Full suite**

Run: `PYTHONIOENCODING=utf-8 python -m pytest -q`
Expected: green apart from the 2 known GARCH failures (no `arch` wheel for Python 3.14) and `test_mt5_watchdog.py::TestIsolation::test_watchdog_never_imports_the_broker_module`, a pre-existing order dependence — `tests/test_mt5_mcp.py` leaves `MetaTrader5` in `sys.modules`. Neither is caused by this task; confirm the count does not grow.

- [x] **Step 9: Prove it end-to-end**

```bash
python deploy/sync_version.py 1.10.3
docker compose build app && docker compose up -d
docker exec dashboard-pro-sweeper-1 python -m src.services.signal_sweep --only biased_pivots
```

Then confirm the horizon actually persisted, and that the levels moved to a weekly cadence:

```sql
SELECT instrument, checks_detail->>'horizon_days' AS horizon,
       checks_detail->>'bar_time' AS bar
FROM trade_setups
WHERE source = 'biased_pivots' AND logged_at > now() - interval '10 minutes'
ORDER BY instrument;
```

Expected: `horizon = 5`, and every `bar_time` on a **week boundary** (a Sunday, from the `W` resample) rather than a weekday. A weekday `bar_time` means Step 5 did not take.

- [x] **Step 10: `python deploy/verify_deploy.py`** — host and all four containers must report the same fingerprint.

- [x] **Step 11: Show the owner the diff. Do not commit.**

---

## Out of scope, deliberately

- **The existing 130 daily-derived signals in `trade_setups`.** They keep their stored `horizon_days = 5` and expire naturally within 7 calendar days. Rewriting stored history would destroy the evidence the Source Scorecard needs, and the Scorecard cannot score them anyway until they resolve.
- **`src/core/biased_pivots.py`.** Period-agnostic by design; changing it would couple the maths to one timeframe.
- **Other sources' horizons.** `market_structure` was fixed in the same pass that produced this defect's diagnosis (4H → 3 days). `daily_macd`, `weekly_ema` and `daily_trend` declare none and inherit the 10-day default — that is a separate audit with its own plan.

## Verification for the whole plan

- [x] `tests/test_biased_pivots_horizon.py` — all pass.
- [x] `tests/test_biased_pivots.py` — unchanged and green (the maths did not move).
- [x] Full suite: no new failures beyond the 3 known.
- [x] A live sweep writes `horizon_days = 5` with week-boundary `bar_time`.
- [x] `verify_deploy.py` reports in sync at 1.10.3.

---

Module map: [[Architecture]] · Docs index: [[README]]
