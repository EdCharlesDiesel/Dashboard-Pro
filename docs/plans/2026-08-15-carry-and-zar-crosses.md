# Carry in the Signals, ZAR Crosses Out — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two changes from the institutional review. (1) Retire the four ZAR crosses whose spreads consume a fifth of their own risk budget. (2) Make **carry** — the dominant driver at a 5–20 day horizon — an input to signal scoring instead of a number stranded in a research tab.

**Architecture:** The removal is a registry edit plus the three places that hardcode the universe's shape. The carry work is a new `src/services/carry.py`: it maps a registry pair to its two policy rates via `FRED_SERIES`, reads them through the existing `fred_data.fred_series()` (Postgres-first, so no new I/O path), and returns the annualised differential for being **long** the pair. `src/core/signals.py` then gains one scoring check and one detail field. No page changes — the number flows into `checks_detail` and the existing renderers pick it up.

**Tech Stack:** Python 3.14 venv (deploy target 3.12), pandas, Postgres 18, FRED, pytest, Docker Compose.

**Spec:** `.claude/CLAUDE.md` — "Single sources of truth" (the registry is the universe; `fred_data` is the macro spine). The evidence for both changes is in this plan's *Measured starting state*.

## Global Constraints

- Never commit. Make changes only; the repository owner reviews and commits.
- **A plan gets its own bump too.** `VERSION` read **1.10.10**, so this plan takes **1.10.11** — the patch, plus one.
- **Every completed task bumps the version**: read `VERSION`, add one to the last number, `python deploy/sync_version.py <that>`, then rebuild and `python deploy/verify_deploy.py`. Never a minor bump, never a reserved block, never a skipped number.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- Use type hints on new code.
- Coverage gate is `--cov-fail-under=80`; `src/services/` and `src/core/` are both in scope, so both tasks need real unit tests.
- Never remove an `@st.cache_data` decorator.
- **Do not delete stored data.** Rows in `trade_setups` and `market_bars` for retired instruments stay: they are the Source Scorecard's evidence, and deleting them would rewrite history to make a decision look better than it was.

---

## Measured starting state (2026-08-15, v1.10.10)

### Why the crosses go

Spread as a fraction of daily ATR14, measured live:

| pair | spread | ATR14 | spread/ATR | verdict |
|---|---|---|---|---|
| NZD/ZAR | 296 pips | 0.079 | **37.5%** | retire |
| EUR/ZAR | 551 | 0.157 | 35.1% | **keep** — see below |
| AUD/ZAR | 246 | 0.083 | **29.5%** | retire |
| CHF/ZAR | 430 | 0.155 | **27.7%** | retire |
| GBP/ZAR | 460 | 0.169 | 27.3% | **keep** |
| ZAR/JPY | 2.2 | 0.098 | **22.3%** | retire |
| USD/ZAR | 332 | 0.151 | 22.0% | **keep** |
| AUD/USD | 0.6 | 0.0041 | 1.5% | — |
| EUR/USD | 0.6 | 0.0050 | 1.2% | — |

On a 1.5×ATR stop, a 30% spread/ATR hands **~20% of the risk budget** to the spread before the market moves. G10 majors cost under 1%.

**Retire: CHF/ZAR, NZD/ZAR, AUD/ZAR, ZAR/JPY.** These are synthetic crosses with no natural interbank liquidity. **Keep USD/ZAR, EUR/ZAR, GBP/ZAR** — genuine depth, and the owner holds live positions in EUR/ZAR and GBP/ZAR.

> **Measure again before deleting.** These are Saturday, market-closed readings and overstate the true cost. Task 1 Step 1 re-measures during London and records the numbers. If a cross comes in under 15% of ATR on a weekday, keep it and say so — the decision follows the evidence, not this table.

### Why carry goes in

`uip_carry_analysis()` exists in `src/core/quant_models.py:271` and is reachable only from `pages/quant_models_tab.py`. It votes on nothing.

`FRED_SERIES` (`src/core/data_provider.py`) already carries a `Rates` series for **all nine currencies** in the universe:

```
USD FEDFUNDS · EUR ECBDFR · GBP BOERUKM · JPY IRSTCI01JPM156N · CHF ...
ZAR IRSTCI01ZAM156N · AUD IRSTCI01AUM156N · NZD IRSTCI01NZM156N · CAD IRSTCI01CAM156N
```

and `src/services/fred_data.fred_series(series_id, start=None)` already reads them Postgres-first.

This matters beyond completeness: the GBM drift estimated from price history came out **statistically indistinguishable from zero on every instrument** (t = 0.13 to 1.64, measured 2026-08-15). Carry is the one component of expected return that does **not** have to be inferred from noisy prices — it can be read off the policy rates. A short USD/ZAR position pays the differential every night, and nothing in the signal path currently knows that.

`src/core/signals.py` scores direction as `direction_score / direction_max` over a dict of boolean checks (line 556), and records a human-readable string per check in `details` (line 536 shows the `Spread/ATR` pattern). Carry follows that exact shape.

### The universe today

26 instruments — 15 pure G10, 7 ZAR, 4 commodities. After Task 1: **22**.

`tests/test_registry.py::test_len_is_full_universe` hardcodes 26 and must move to 22 — that test exists precisely so a silent change fails loudly.

---

## File structure

- **Modify** `src/instruments/registry.py` — drop 4 rows; shrink the `ZAR pairs` correlation group; drop 4 `TYPICAL_SPREAD_PIPS` entries; remove ZAR/JPY from `JPY crosses`.
- **Modify** `tests/test_registry.py` — the hardcoded count and any ZAR-cross references.
- **Create** `src/services/carry.py` — pair → annualised carry. Pure apart from the `fred_data` read.
- **Create** `tests/test_carry.py`.
- **Modify** `src/core/signals.py` — one check in the direction dict, one `details` entry.
- **Modify** `tests/test_signals.py` — cover the new check.

---

### Task 1: Retire the four ZAR crosses

**Files:**
- Modify: `src/instruments/registry.py:62-64,71,94,108-109,123-125`
- Modify: `tests/test_registry.py:36`

- [ ] **Step 1: Re-measure spreads during London hours, and record them here**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src.instruments.registry import INSTRUMENTS
from src.services.market_data import daily_ohlc
from src.indicators.technical import TechnicalIndicators as TI
# fill from a live mcp__mt5__get_quote call
quotes = {}   # {'CHF/ZAR': (bid, ask), ...}
for p,(b,a) in quotes.items():
    inst = INSTRUMENTS.get(p); df = daily_ohlc(inst.ticker)
    atr = float(TI.atr(df['High'], df['Low'], df['Close'], 14).iloc[-1])
    print('%-9s %6.1f%% of ATR' % (p, (a-b)/atr*100))"
```

**Gate:** retire only those still above **15% of ATR** on a weekday. Write the measured numbers into this plan under *Result* before proceeding. If all four come in tight, stop and tell the owner — the premise was wrong.

- [ ] **Step 2: Write the failing test**

`tests/test_registry.py` — change the count and add the reason:

```python
    def test_len_is_full_universe(self):
        # Deliberately a hard number: the registry drives what every scanner
        # scans, so a silent addition or removal should fail loudly here.
        # 22 -> 25 (CHF/NZD/AUD ZAR), 26 (ZAR/JPY), then back to 22 on
        # 2026-08-15 when the four synthetic ZAR crosses were retired for
        # spreads of 22-37% of daily ATR (see
        # docs/plans/2026-08-15-carry-and-zar-crosses.md).
        assert len(INSTRUMENTS) == 22

    def test_the_illiquid_zar_crosses_are_gone(self):
        for pair in ("CHF/ZAR", "NZD/ZAR", "AUD/ZAR", "ZAR/JPY"):
            assert pair not in INSTRUMENTS, (
                "%s costs a fifth of its own risk budget in spread" % pair)

    def test_the_liquid_zar_pairs_remain(self):
        for pair in ("USD/ZAR", "EUR/ZAR", "GBP/ZAR"):
            assert pair in INSTRUMENTS
```

- [ ] **Step 3: Run to verify it fails** — `assert 26 == 22`.

- [ ] **Step 4: Remove the four rows** from the `INSTRUMENTS` table (lines 62-64 for CHF/NZD/AUD ZAR, line 71 for ZAR/JPY). Keep the block comment explaining the 0.62 pip value — it still applies to the three survivors.

- [ ] **Step 5: Shrink the correlation groups**

`"ZAR pairs (same dir = stacked ZAR risk)"` → `frozenset({"USD/ZAR", "EUR/ZAR", "GBP/ZAR"})`.
`"JPY crosses"` → remove `"ZAR/JPY"`.

`tests/test_registry.py::test_corr_groups_only_reference_known_instruments` will fail if either is missed — that is the safety net, but do not rely on it.

- [ ] **Step 6: Remove the four `TYPICAL_SPREAD_PIPS` entries** (lines 123-125 for the crosses, 127 for ZAR/JPY — keep `XAU/USD` and `XAG/USD` on that line).

- [ ] **Step 7: Run the registry tests, then the full suite**

Watch for anything else hardcoding 26 or naming a retired pair:
`grep -rn "CHF/ZAR\|NZD/ZAR\|AUD/ZAR\|ZAR/JPY" --include=*.py src/ pages/ tests/`

- [ ] **Step 8: Confirm the retired pairs stop being scanned**

```bash
docker exec dashboard-pro-sweeper-1 python -c "
from src.instruments.registry import INSTRUMENTS
print(len(INSTRUMENTS), sorted(p for p in INSTRUMENTS if 'ZAR' in p))"
```
Expected: `22 ['EUR/ZAR', 'GBP/ZAR', 'USD/ZAR']`.

Stored `trade_setups` rows for retired pairs **stay**. They are scorecard evidence.

- [ ] **Step 9: Bump, rebuild, `verify_deploy.py`, show the diff. Do not commit.**

---

### Task 2: A carry service

**Files:**
- Create: `src/services/carry.py`
- Test: `tests/test_carry.py`

**Interfaces:**
- Produces: `carry_pct(pair: str, *, now: date | None = None) -> float | None` — annualised carry in **percent** for being **long** the pair: `rate(base) - rate(quote)`. `None` when either rate is unavailable.
- Produces: `RATE_SERIES: dict[str, str]` — currency → FRED series id, derived from `FRED_SERIES` so there is one source of truth.
- Consumes: `src.services.fred_data.fred_series`.

- [ ] **Step 1: Write the failing tests**

```python
"""Carry: the one component of expected return you do not have to infer.

GBM drift estimated from price history was statistically indistinguishable from
zero on every instrument in this universe (t = 0.13 to 1.64, 2026-08-15). The
rate differential is not inferred -- it is read off the policy rates, and at a
5-20 day horizon it is the dominant driver.

Sign convention: `carry_pct` is what you EARN for being LONG the pair. Long
USD/ZAR earns rate(USD) - rate(ZAR), which is deeply negative; the short earns
the mirror. Getting this backwards would invert the trade on every EM pair, so
it has its own test.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.services import carry


def _series(value):
    return pd.Series([value], index=pd.to_datetime(["2026-08-01"]))


@pytest.fixture
def rates(monkeypatch):
    table = {"FEDFUNDS": 4.0, "IRSTCI01ZAM156N": 7.5, "ECBDFR": 2.0}

    def fake(series_id, start=None):
        return _series(table[series_id]) if series_id in table else pd.Series(dtype=float)

    monkeypatch.setattr(carry, "fred_series", fake)
    return table


class TestCarryPct:
    def test_long_a_low_yielder_against_a_high_yielder_is_negative(self, rates):
        # Long USD/ZAR = long USD (4.0), short ZAR (7.5) -> -3.5
        assert carry.carry_pct("USD/ZAR") == pytest.approx(-3.5)

    def test_the_sign_flips_with_the_quote_order(self, rates):
        # EUR/USD: long EUR (2.0), short USD (4.0) -> -2.0
        assert carry.carry_pct("EUR/USD") == pytest.approx(-2.0)

    def test_a_missing_rate_is_none_not_zero(self, rates):
        # Zero would read as "no carry", which is a claim. None is the truth.
        assert carry.carry_pct("XAU/USD") is None

    def test_an_unknown_pair_is_none(self, rates):
        assert carry.carry_pct("FOO/BAR") is None

    def test_commodities_have_no_carry(self, rates):
        for pair in ("XAU/USD", "XAG/USD", "WTI/USD"):
            assert carry.carry_pct(pair) is None

    def test_a_fred_failure_never_raises(self, monkeypatch):
        def boom(series_id, start=None):
            raise RuntimeError("FRED down")
        monkeypatch.setattr(carry, "fred_series", boom)
        assert carry.carry_pct("USD/ZAR") is None

    def test_an_empty_series_is_none(self, monkeypatch):
        monkeypatch.setattr(carry, "fred_series",
                            lambda s, start=None: pd.Series(dtype=float))
        assert carry.carry_pct("USD/ZAR") is None


class TestFavoursDirection:
    def test_long_is_favoured_when_carry_is_positive(self, rates):
        # ZAR/USD does not exist, so use the real high-yielder case in reverse:
        # short USD/ZAR earns +3.5.
        assert carry.favours("USD/ZAR", "Short") is True
        assert carry.favours("USD/ZAR", "Long") is False

    def test_unknown_carry_neither_favours_nor_opposes(self, rates):
        assert carry.favours("XAU/USD", "Long") is None

    def test_a_negligible_differential_is_neutral(self, monkeypatch):
        monkeypatch.setattr(carry, "fred_series",
                            lambda s, start=None: _series(4.0))
        # Both legs 4.0 -> 0.0 differential, inside the dead band.
        assert carry.favours("EUR/USD", "Long") is None


class TestRateSeriesTable:
    def test_every_currency_in_the_universe_has_a_rate(self):
        from src.instruments.registry import INSTRUMENTS
        missing = set()
        for pair in INSTRUMENTS:
            base, _, quote = pair.partition("/")
            for ccy in (base, quote):
                if ccy in ("XAU", "XAG", "XPT", "WTI"):
                    continue
                if ccy not in carry.RATE_SERIES:
                    missing.add(ccy)
        assert not missing, "no policy-rate series for: %s" % sorted(missing)
```

- [ ] **Step 2: Run to verify they fail** — no module `src.services.carry`.

- [ ] **Step 3: Implement `src/services/carry.py`**

```python
"""Annualised carry per registry pair, from policy rates.

`carry_pct` is what you EARN for being LONG the pair: rate(base) - rate(quote).
Long USD/ZAR earns the US rate and pays the South African one, so it is deeply
negative and the short earns the mirror. This sign is the whole point of the
module, and getting it backwards would invert the read on every EM pair.

Rates come through `src/services/fred_data.py`, which is Postgres-first, so a
page asking for carry does not add a network round-trip.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Optional

from src.core.data_provider import FRED_SERIES
from src.services.fred_data import fred_series

logger = logging.getLogger("ForexDashboard")

# Currency -> FRED policy-rate series, derived from FRED_SERIES so the mapping
# is not duplicated. Metals and oil are absent by construction: they have no
# policy rate, and `carry_pct` returns None for them rather than a misleading 0.
RATE_SERIES: Dict[str, str] = {
    ccy: block["Rates"] for ccy, block in FRED_SERIES.items() if block.get("Rates")
}

# Differentials smaller than this are noise, not an edge: policy rates move in
# 25bp steps and the series are monthly.
DEAD_BAND_PCT = 0.25


def _latest_rate(currency: str) -> Optional[float]:
    series_id = RATE_SERIES.get(currency)
    if not series_id:
        return None
    try:
        series = fred_series(series_id)
    except Exception as exc:                      # noqa: BLE001 — never break a page
        logger.warning("[carry] %s (%s) unavailable: %s", currency, series_id, exc)
        return None
    if series is None or len(series) == 0:
        return None
    try:
        return float(series.dropna().iloc[-1])
    except Exception:                             # noqa: BLE001
        return None


def carry_pct(pair: str, *, now: Optional[date] = None) -> Optional[float]:
    """Annualised carry, in percent, for being LONG ``pair``."""
    base, _, quote = str(pair or "").partition("/")
    if not base or not quote:
        return None
    base_rate, quote_rate = _latest_rate(base), _latest_rate(quote)
    if base_rate is None or quote_rate is None:
        return None
    return base_rate - quote_rate


def favours(pair: str, direction: str) -> Optional[bool]:
    """Does carry pay this direction? ``None`` when unknown or negligible."""
    value = carry_pct(pair)
    if value is None or abs(value) < DEAD_BAND_PCT:
        return None
    long_side = str(direction or "").strip().upper().startswith(("L", "B"))
    return (value > 0) if long_side else (value < 0)
```

- [ ] **Step 4: Run the tests** — expect PASS.

- [ ] **Step 5: Prove it against live FRED data**

```bash
PYTHONIOENCODING=utf-8 python -c "
from src.services.carry import carry_pct
for p in ('USD/ZAR','EUR/ZAR','GBP/ZAR','EUR/USD','USD/JPY','AUD/USD','XAU/USD'):
    print('%-9s %s' % (p, carry_pct(p)))"
```
Expected: USD/ZAR strongly negative (SA policy rate well above the Fed's), `XAU/USD` `None`. **If every value is `None`, `fred_series` has no stored data — run the data_backbone worker before concluding the module is broken.**

- [ ] **Step 6: Bump, rebuild, `verify_deploy.py`, show the diff.**

---

### Task 3: Carry as a scoring check

**Files:**
- Modify: `src/core/signals.py` — the direction-check dict (~line 533-556) and `details`
- Test: `tests/test_signals.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_signals.py`, mirroring how the existing `Spread/ATR` check is tested:

```python
class TestCarryCheck:
    def test_carry_against_the_trade_fails_the_check(self, monkeypatch):
        monkeypatch.setattr("src.core.signals.carry_favours",
                            lambda pair, direction: False)
        result = _score_direction(...)      # use the file's existing helper
        assert result["details"]["Carry"].startswith("❌")

    def test_carry_with_the_trade_passes(self, monkeypatch):
        monkeypatch.setattr("src.core.signals.carry_favours",
                            lambda pair, direction: True)
        result = _score_direction(...)
        assert result["details"]["Carry"].startswith("✅")

    def test_unknown_carry_neither_helps_nor_hurts(self, monkeypatch):
        # Metals have no policy rate. The check must not silently count as a
        # failure, or every gold signal loses a point for a question that does
        # not apply to it.
        monkeypatch.setattr("src.core.signals.carry_favours",
                            lambda pair, direction: None)
        result = _score_direction(...)
        assert result["details"]["Carry"] == "—"
```

**Read `tests/test_signals.py` first** and match its existing fixture style — the helper that builds a scored signal already exists there; do not invent a new one.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement**

Import at the top of `src/core/signals.py`:

```python
from src.services.carry import favours as carry_favours
```

**Verified 2026-08-15: this is not circular.** `src/core/data_provider.py` already imports `src.services.fred_data`, and `fred_data` imports only `src.data_backbone` — so `core → services` is an established direction here, and `src.core.signals` + `src.services.fred_data` + `src.core.data_provider` import together cleanly. A module-level import is correct; no function-local workaround needed.

In the direction-check block, beside `Spread/ATR`:

```python
    # Carry is the one expected-return component that is not inferred from
    # price history -- and price-history drift was indistinguishable from zero
    # across this whole universe (t = 0.13 to 1.64, 2026-08-15). A trade
    # fighting a meaningful differential is paying to hold every night.
    carry_ok = carry_favours(pair, direction)
    if carry_ok is None:
        details["Carry"] = "—"          # no policy rate (metals, oil) or negligible
    else:
        direction_items["carry"] = carry_ok
        details["Carry"] = f"{'✅' if carry_ok else '❌'} carry {'pays' if carry_ok else 'costs'}"
```

**Note the asymmetry and keep it:** an unknown carry is left *out* of `direction_items` entirely rather than scored as False. Adding it as a failure would dock every metal signal a point for a question that does not apply. That changes `direction_max` per instrument, which the existing `pct` calculation already handles (`direction_score / direction_max`).

- [ ] **Step 4: Run the tests.**
- [ ] **Step 5: Full suite.** Watch `tests/test_signals.py` for count-based assertions on `direction_max` — adding a conditional check changes it for FX pairs but not metals.
- [ ] **Step 6: Sweep and confirm carry reaches the board**

```bash
docker exec dashboard-pro-sweeper-1 python -m src.services.signal_sweep --only setup_ranker
```
```sql
SELECT instrument, direction, checks_detail->'details'->>'Carry'
FROM trade_setups WHERE logged_at > now() - interval '10 minutes' LIMIT 10;
```

- [ ] **Step 7: Bump, rebuild, `verify_deploy.py`, show the diff. Do not commit.**

---

## Out of scope, deliberately

- **Carry-weighted position sizing.** Scoring first; sizing off carry is a separate decision with its own risk profile.
- **Forward points / swap rates from the broker.** Policy-rate differentials approximate carry well at this horizon; broker swap rates are the exact figure and belong in a later plan if the approximation proves too coarse.
- **Deleting stored rows for retired instruments.** Scorecard evidence.
- **The other institutional-review items** — spread as a hard gate, event-risk gating, vol-regime sizing. Each is its own plan.

## Verification for the whole plan

- [ ] `len(INSTRUMENTS) == 22`, and the sweeper agrees.
- [ ] London-hours spreads recorded in *Result* below, justifying each retirement.
- [ ] `carry_pct('USD/ZAR')` strongly negative; `carry_pct('XAU/USD')` is `None`.
- [ ] A swept signal carries a `Carry` entry in `checks_detail`.
- [ ] Full suite: no failures beyond the 3 known.
- [ ] `verify_deploy.py` in sync at the final version.

---

Module map: [[Architecture]] · Docs index: [[README]]
