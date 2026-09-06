# Gate Limits Are Scale-Relative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the pre-trade gate rejecting legitimate gold trades by replacing
three absolute point-count limits with limits expressed relative to what the
instrument actually does — stop distance and ATR — so one config is correct
across a $1.16 pair and a $4,430 one.

**Architecture:** `GateConfig` gains three scale-relative limits. Two need no
new data (spread is measured against the signal's own stop distance). The third
needs ATR, so `MarketSnapshot` gains an `atr` field the executor fills from D1
candles. Every ATR-based check falls back to today's absolute limit when `atr`
is unavailable, so behaviour is unchanged wherever ATR is not plumbed.

**Tech Stack:** Python 3.14, pytest. `src/execution/gate.py` is pure — no MT5,
no DB, no I/O — and must stay that way.

**Spec:** The owner's request, 2026-09-06 ("go"), following the sync audit in
this session. Measurements below are from the live Exness terminal and the
board, taken 2026-09-06.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.66**.
- **The gate is never weakened to let a blocked signal through.** This plan
  changes the *units* of three limits, not their intent. Every task must show
  that the stub case in `looks_synthetic`'s docstring is still blocked.
- **`gate.py` stays pure.** ATR arrives as a field on `MarketSnapshot`, computed
  by the caller. No imports of MT5, pandas or services into `gate.py`.
- **Fail closed on missing data.** When `snap.atr <= 0`, ATR-based checks fall
  back to the existing absolute point limits rather than skipping.
- **The executor stays disarmed.** It is not wired into `docker-compose.yml` and
  defaults to `dry_run`. Nothing in this plan arms it.
- `looks_synthetic` is **out of scope** — see "Deliberately out of scope".

---

## Context

The board says **XAU/USD BULLISH 0.67**. A reasonable gold long is rejected four
times over. Measured, 2026-09-06, with the live spec (`digits=3`, `point=0.001`,
bid 4430.194 / ask 4430.376):

```
pullback limit-buy entry=4420 stop=4400 tp=4460 -> ok=False
   BLOCK: stop 20000pts above maximum 5000
   BLOCK: entry 10376pts from market (max 300) - stale or stub
   BLOCK: levels look synthetic (round grid + integer R)
   BLOCK: spread 182pts above 40
```

Every one of these is a **units artifact**, not a risk judgment. The limits are
absolute point counts, and a "point" means wildly different things across the
instruments the gate is whitelisted for.

### The same limit, three meanings

`max_stop_points = 5000` as a fraction of price:

| Symbol | price | point | max stop | as % of price |
|---|---|---|---|---|
| EUR/USD | 1.1628 | 0.00001 | 0.05 | 4.61% |
| XAG/USD | 68.21 | 0.001 | $5.00 | 7.55% |
| **XAU/USD** | **4479.88** | **0.001** | **$5.00** | **0.113%** |

Gold and silver share `point=0.001`, but gold trades 66× higher. Gold gets a
limit ~40× tighter than everything else, purely as an accident of quoting.

### Percent-of-price does not fix it either

ATR14 from D1 candles (20 bars, via the broker terminal):

| Symbol | ATR14 | ATR as % of price | 3×ATR as % of price |
|---|---|---|---|
| XAU/USD | 104.37 | 2.330% | 6.99% |
| XAG/USD | 2.5539 | 3.744% | 11.23% |
| EUR/USD | 0.0042 | 0.365% | 1.09% |

ATR-as-%-of-price differs 10× between metals and FX, so a flat percentage cap
would block metals or be meaningless for FX. **ATR multiple is the only measure
that means the same thing on all three.**

### The spread limit is backwards

`max_spread_points = 40` blocks gold's *normal* spread. This is not a weekend
artifact — the D1 candles carry a `spread` field, and gold reads 168–182 on
every session including 2026-08-24 with 458,379 ticks:

| Symbol | spread (pts) | as % of price | as % of its own ATR |
|---|---|---|---|
| XAU/USD | 182 | 0.0041% | **0.17%** |
| XAG/USD | 21 | 0.0308% | 0.82% |
| EUR/USD | 6 | 0.0052% | 1.43% |

Relative to the move it is trying to capture, **gold is the cheapest of the
three to trade** — and it is the only one the gate blocks.

### Why this went unnoticed

The gate's whitelist is `("XAUUSD", "XAGUSD")`: it is built *for* metals. Silver
passes every check, because at $68 its point-to-price ratio happens to resemble
FX. Only gold, at 66× silver's price with the same `point`, falls outside. One
of the two whitelisted symbols works, so the config looks calibrated.

Nothing is being lost today: there are no `EXECUTOR_*` variables in
`docker-compose.yml`, so the executor is not running as a service and defaults
to dry-run. This bites the moment it is armed.

### The constants were validated before this plan was written

A prototype of the three proposed checks was run against the live specs and the
measured ATR14 above, so an implementer is not discovering the calibration is
wrong at Task 5. Results:

| Case | XAU/USD | XAG/USD | EUR/USD |
|---|---|---|---|
| realistic trade at market, ordinary stop | passes | passes | passes |
| the Context pullback (4420 / 4400) | passes | — | — |
| absurd stop (4× ATR) | blocked | blocked | blocked |
| the stub (1.10000 vs 1.16279 market) | — | — | blocked, 14.95× ATR |
| fallback with `atr=0` | blocked (20000pts) | — | — |

The fallback row is the important one: with ATR unavailable the gold trade is
blocked exactly as it is today. The change fails closed.

**Measured spread headroom** before `max_spread_frac_of_stop = 0.10` fires,
against each instrument's ordinary spread and stop: gold 11×, EUR/USD 5×,
silver 4.8×. A 10× gold blowout (1820 points against a $20 stop) sits at 9.1%
and still passes; ~11× blocks. That is deliberate — see the notes at the end.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/execution/gate.py` | pure gate + sizing | add `atr` to `MarketSnapshot`; add 3 scale-relative limits to `GateConfig`; rewrite 3 checks in `run_gate` |
| `src/execution/mt5_executor.py` | terminal I/O, snapshot construction | compute ATR14 in `snapshot()` |
| `tests/test_execution_gate.py` | existing gate unit tests | extend |
| `tests/test_gate_scale_calibration.py` | **new** — cross-instrument guard | create |

---

## Task 1: Spread relative to the stop it protects

The spread check needs no new data. What matters economically is not the raw
point count but how much of the trade's risk the spread consumes.

**Files:**
- Modify: `src/execution/gate.py` (`GateConfig` ~line 113, `run_gate` ~line 260)
- Test: `tests/test_execution_gate.py`

**Interfaces:**
- Produces: `GateConfig.max_spread_frac_of_stop: float = 0.10`. The existing
  `max_spread_points` stays as a hard ceiling for absurd spreads.

- [ ] **Step 1: Write the failing tests**

```python
def _gold_snap(bid=4430.194, ask=4430.376):
    return MarketSnapshot(
        symbol="XAUUSD", bid=bid, ask=ask, point=0.001, digits=3,
        tick_value=1.0, tick_size=0.001, volume_min=0.01, volume_step=0.01,
        volume_max=200.0, trade_allowed=True, stops_level_points=0.0)


def _acct():
    return AccountState(balance=10_000, equity=10_000, free_margin=9_000,
                        open_positions=0, enabled=True, dry_run=True)


def test_gold_normal_spread_does_not_block_a_normal_stop():
    """182 points is gold's ordinary spread and 0.9% of a $20 stop."""
    snap = _gold_snap()
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4430.376, stop=4410.376, tp1=4470.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert not any("spread" in r for r in res.reasons), res.reasons


def test_spread_blocks_when_it_eats_the_stop():
    """The same 182-point spread against a $1 stop is 18% of risk."""
    snap = _gold_snap()
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4430.376, stop=4429.376, tp1=4432.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert any("spread" in r for r in res.reasons), res.reasons
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_execution_gate.py -k spread -v --no-cov`
Expected: `test_gold_normal_spread_does_not_block_a_normal_stop` FAILS with
`spread 182pts above 40` present in reasons.

- [ ] **Step 3: Add the config field**

In `GateConfig`, under `# --- execution conditions ---`:

```python
    # --- execution conditions ----------------------------------------------
    # An absolute point count means something different on every instrument:
    # 40 points is 6.7x EUR/USD's typical spread but a quarter of gold's
    # ordinary 182, so this limit blocked every gold trade while passing
    # silver. What matters is how much of the trade's own risk the spread
    # consumes, which is scale-free.
    max_spread_frac_of_stop: float = 0.10
    # Kept as a backstop for a genuinely broken quote, not as the primary
    # test. Raised so it cannot fire on a normal metals spread.
    max_spread_points: float = 100_000.0
```

- [ ] **Step 4: Rewrite the check**

Replace the `--- execution conditions ---` block in `run_gate`:

```python
    # --- execution conditions ----------------------------------------------
    if sig.stop_distance > 0 and snap.point > 0:
        spread_price = snap.ask - snap.bid
        frac = spread_price / sig.stop_distance
        if frac > cfg.max_spread_frac_of_stop:
            res.block(f"spread {snap.spread_points:.0f}pts is {frac:.1%} of the "
                      f"stop (max {cfg.max_spread_frac_of_stop:.0%})")
    if snap.spread_points > cfg.max_spread_points:
        res.block(f"spread {snap.spread_points:.0f}pts above "
                  f"{cfg.max_spread_points:.0f}")
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS, including the pre-existing spread test at line 103 — check it;
if it asserted on `max_spread_points=40` it must be updated to construct
`GateConfig(max_spread_points=40.0)` explicitly and assert the backstop, not the
primary path.

---

## Task 2: A maximum stop measured in ATR

**Files:**
- Modify: `src/execution/gate.py` (`MarketSnapshot` ~line 58, `GateConfig`
  ~line 107, `run_gate` ~line 237)
- Test: `tests/test_execution_gate.py`

**Interfaces:**
- Consumes: `_gold_snap()` from Task 1.
- Produces: `MarketSnapshot.atr: float = 0.0` (price units, 0 = unknown);
  `GateConfig.max_stop_atr_mult: float = 3.0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_normal_gold_stop_passes_when_atr_is_known():
    """$20 is 0.19 ATR on gold (ATR14 = $104). Nowhere near a 3x cap."""
    snap = replace(_gold_snap(), atr=104.37)
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4430.376, stop=4410.376, tp1=4470.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert not any("stop" in r and "maximum" in r for r in res.reasons), res.reasons


def test_absurd_stop_still_blocked_in_atr_terms():
    """$400 is 3.8 ATR - past the 3x cap."""
    snap = replace(_gold_snap(), atr=104.37)
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4430.376, stop=4030.376, tp1=5230.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert any("ATR" in r for r in res.reasons), res.reasons


def test_falls_back_to_points_when_atr_unknown():
    """atr=0 must not disable the check - it reverts to today's limit."""
    snap = _gold_snap()          # atr defaults to 0.0
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4430.376, stop=4410.376, tp1=4470.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert any("above maximum" in r for r in res.reasons), res.reasons
```

Add `from dataclasses import replace` to the test module's imports.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_execution_gate.py -k "atr or gold_stop" -v --no-cov`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'atr'`.

- [ ] **Step 3: Add the `atr` field**

In `MarketSnapshot`, after `margin_per_lot`:

```python
    margin_per_lot: float = 0.0
    #: ATR14 on D1 in *price* units, or 0.0 when unknown. Supplied by the
    #: caller - gate.py is pure and computes no indicators. Limits expressed
    #: as ATR multiples mean the same thing on a $1.16 pair and a $4,430 one;
    #: absolute point counts do not.
    atr: float = 0.0
```

- [ ] **Step 4: Add the config field**

In `GateConfig`, replacing the `max_stop_points` line:

```python
    min_stop_points: float = 50.0
    # 3x ATR is a wide-but-sane ceiling on every instrument: $313 on gold,
    # $7.66 on silver, 126 pips on EUR/USD. Used whenever snap.atr > 0.
    max_stop_atr_mult: float = 3.0
    # Fallback only, for snap.atr <= 0. Absolute points are wrong across
    # instruments (5000 = 4.61% of EUR/USD but 0.113% of gold), so this is
    # deliberately *not* the primary test.
    max_stop_points: float = 5000.0
```

- [ ] **Step 5: Rewrite the stop check**

Replace the `--- stop distance vs instrument ---` block in `run_gate`:

```python
    # --- stop distance vs instrument ---------------------------------------
    if snap.point > 0:
        stop_pts = sig.stop_distance / snap.point
        if stop_pts < cfg.min_stop_points:
            res.block(f"stop {stop_pts:.0f}pts below minimum {cfg.min_stop_points:.0f}")
        if snap.atr > 0:
            mult = sig.stop_distance / snap.atr
            if mult > cfg.max_stop_atr_mult:
                res.block(f"stop {mult:.2f}x ATR above maximum "
                          f"{cfg.max_stop_atr_mult:.1f}x")
        elif stop_pts > cfg.max_stop_points:
            res.block(f"stop {stop_pts:.0f}pts above maximum "
                      f"{cfg.max_stop_points:.0f}")
        if snap.stops_level_points and stop_pts < snap.stops_level_points:
            res.block(f"stop {stop_pts:.0f}pts inside broker stops level "
                      f"{snap.stops_level_points:.0f}")
```

- [ ] **Step 6: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS.

---

## Task 3: Entry deviation measured in ATR

`max_entry_deviation_points = 300` is $0.30 on gold, so every pullback limit
entry is rejected as "stale or stub". Expressed in ATR the same limit correctly
passes a $10 pullback (0.096 ATR) while still catching the motivating stub —
EUR/USD entry 1.10000 against a 1.16279 market is 14.9 ATR away.

**Files:**
- Modify: `src/execution/gate.py` (`GateConfig` ~line 101, `run_gate` ~line 248)
- Test: `tests/test_execution_gate.py`

**Interfaces:**
- Consumes: `MarketSnapshot.atr` from Task 2.
- Produces: `GateConfig.max_entry_deviation_atr_mult: float = 1.0`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gold_pullback_entry_is_not_called_stale():
    snap = replace(_gold_snap(), atr=104.37)
    sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                 entry=4420.376, stop=4400.376, tp1=4461.376)
    res = run_gate(sig, snap, _acct(), GateConfig())
    assert not any("from market" in r for r in res.reasons), res.reasons


def test_the_stub_entry_is_still_blocked():
    """The motivating case from looks_synthetic's docstring: 1.10000 while
    EUR/USD trades at 1.16279 - 14.9 ATR away."""
    snap = MarketSnapshot(
        symbol="EURUSD", bid=1.16269, ask=1.16279, point=0.00001, digits=5,
        tick_value=1.0, tick_size=0.00001, volume_min=0.01, volume_step=0.01,
        volume_max=200.0, trade_allowed=True, stops_level_points=0.0,
        atr=0.0042)
    sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                 entry=1.10000, stop=1.09500, tp1=1.11000)
    # The default whitelist is metals-only, so EUR/USD would also collect a
    # whitelist block. Name it explicitly - otherwise this test could pass on
    # the wrong reason, which is how three guards in this repo already went
    # green while checking nothing.
    cfg = GateConfig(symbol_whitelist=("EURUSD",))
    res = run_gate(sig, snap, _acct(), cfg)
    assert any("from market" in r for r in res.reasons), res.reasons
    assert not any("whitelist" in r for r in res.reasons), res.reasons
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_execution_gate.py -k "pullback or stub_entry" -v --no-cov`
Expected: `test_gold_pullback_entry_is_not_called_stale` FAILS with
`entry 10376pts from market (max 300)`.

- [ ] **Step 3: Add the config field**

```python
    # --- sanity on the levels themselves -----------------------------------
    # 300 points is $0.30 on gold, so every pullback limit entry read as a
    # stub. 1x ATR passes a $10 gold pullback (0.096 ATR) and still rejects
    # the 1.10000-vs-1.16279 stub (14.9 ATR).
    max_entry_deviation_atr_mult: float = 1.0
    max_entry_deviation_points: float = 300.0   # fallback when atr <= 0
    reject_suspicious_round: bool = True
```

- [ ] **Step 4: Rewrite the deviation check**

```python
    # --- is this signal anywhere near the actual market? --------------------
    if snap.point > 0:
        ref = snap.ask if sig.is_buy else snap.bid
        dev_price = abs(sig.entry - ref)
        if snap.atr > 0:
            dev_atr = dev_price / snap.atr
            if dev_atr > cfg.max_entry_deviation_atr_mult:
                res.block(f"entry {dev_atr:.2f}x ATR from market "
                          f"(max {cfg.max_entry_deviation_atr_mult:.1f}x) "
                          f"— stale or stub")
        else:
            dev = dev_price / snap.point
            if dev > cfg.max_entry_deviation_points:
                res.block(f"entry {dev:.0f}pts from market "
                          f"(max {cfg.max_entry_deviation_points:.0f}) — "
                          f"stale or stub")
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS.

---

## Task 4: Give the executor a real ATR

**Files:**
- Modify: `src/execution/mt5_executor.py` (`snapshot()` at line 99)
- Test: `tests/test_execution_gate.py`

**Interfaces:**
- Consumes: `MarketSnapshot.atr` from Task 2.
- Produces: `atr14_from_rates(rates) -> float` and `build_gate_config() ->
  GateConfig` in `mt5_executor`, so both are testable without a terminal.

> **Found during Task 1 review — this task is load-bearing, not cosmetic.**
> `mt5_executor.py:409` constructs its `GateConfig` with
> `max_spread_points=float(os.environ.get("EXECUTOR_MAX_SPREAD_PTS", "40"))`.
> That override re-imposes the exact limit Task 1 removed. Task 1's unit tests
> pass because they call `GateConfig()` directly, while the running executor
> would still block every gold trade. Until Step 6 below lands, **Task 1 is a
> no-op in production.** The same trap applies to any future limit added to
> that constructor.

- [ ] **Step 1: Write the failing test**

```python
def test_atr14_from_rates_matches_a_hand_computed_value():
    """Three bars, no gaps: TR is just high-low, so ATR is their mean."""
    rates = [
        {"high": 10.0, "low": 8.0, "close": 9.0},
        {"high": 11.0, "low": 9.0, "close": 10.0},
        {"high": 12.0, "low": 10.0, "close": 11.0},
    ]
    from src.execution.mt5_executor import atr14_from_rates
    assert atr14_from_rates(rates) == pytest.approx(2.0)


def test_atr14_counts_the_gap_not_just_the_range():
    """A bar that gaps up has a true range larger than its own high-low."""
    rates = [
        {"high": 10.0, "low": 9.0, "close": 9.5},
        {"high": 20.0, "low": 19.0, "close": 19.5},
    ]
    from src.execution.mt5_executor import atr14_from_rates
    assert atr14_from_rates(rates) == pytest.approx(10.5)


def test_atr14_returns_zero_on_insufficient_data():
    from src.execution.mt5_executor import atr14_from_rates
    assert atr14_from_rates([]) == 0.0
    assert atr14_from_rates([{"high": 1.0, "low": 0.5, "close": 0.8}]) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_execution_gate.py -k atr14 -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'atr14_from_rates'`.

- [ ] **Step 3: Implement the helper**

Add to `src/execution/mt5_executor.py`, above `snapshot()`:

```python
def atr14_from_rates(rates, period: int = 14) -> float:
    """Mean true range over the last ``period`` bars, in price units.

    Takes anything indexable by ``high``/``low``/``close`` - MT5 returns a
    numpy recarray, tests pass dicts. Returns 0.0 when there is not enough
    data, which the gate reads as "unknown" and falls back to point limits.
    """
    if rates is None or len(rates) < 2:
        return 0.0
    trs = []
    for i in range(1, len(rates)):
        cur, prev = rates[i], rates[i - 1]
        hi, lo = float(cur["high"]), float(cur["low"])
        pc = float(prev["close"])
        trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
    if not trs:
        return 0.0
    window = trs[-period:]
    return sum(window) / len(window)
```

- [ ] **Step 4: Fill the field in `snapshot()`**

Inside `snapshot()`, after the `margin_per_lot` block and before the
`MarketSnapshot(...)` construction:

```python
    # ATR14 on D1, in price units. Best-effort: the gate treats 0.0 as
    # "unknown" and falls back to absolute point limits, so a terminal that
    # will not serve rates degrades to today's behaviour rather than failing.
    atr = 0.0
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 21)
        atr = atr14_from_rates(rates)
    except Exception:
        log.warning("no D1 rates for %s - gate falls back to point limits", symbol)
```

Then pass `atr=atr` into the `MarketSnapshot(...)` call.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS.

- [ ] **Step 6: Stop the executor re-imposing the old limits**

Write this test first:

```python
def test_the_executor_config_does_not_reimpose_the_old_point_limits(monkeypatch):
    """The runtime config must not undo the scale-relative limits.

    mt5_executor built its GateConfig with
    `max_spread_points=os.environ.get("EXECUTOR_MAX_SPREAD_PTS", "40")`.
    That silently restored the limit this plan removed - unit tests calling
    GateConfig() directly passed while the running executor still blocked
    every gold trade.
    """
    for var in ("EXECUTOR_MAX_SPREAD_PTS", "EXECUTOR_MAX_SPREAD_FRAC"):
        monkeypatch.delenv(var, raising=False)
    from src.execution.mt5_executor import build_gate_config
    cfg = build_gate_config()
    assert cfg.max_spread_points > 1_000, cfg.max_spread_points
    assert cfg.max_spread_frac_of_stop == pytest.approx(0.10)


def test_the_executor_config_still_honours_an_explicit_override(monkeypatch):
    monkeypatch.setenv("EXECUTOR_MAX_SPREAD_FRAC", "0.05")
    from src.execution.mt5_executor import build_gate_config
    assert build_gate_config().max_spread_frac_of_stop == pytest.approx(0.05)
```

Run it: expect `ImportError: cannot import name 'build_gate_config'`.

Then extract the inline `cfg = GateConfig(...)` in `main()` (around line 402)
into a module-level function, changing only the spread lines:

```python
def build_gate_config() -> GateConfig:
    """The executor's runtime GateConfig, from the environment.

    Extracted from main() so it can be asserted on without a terminal or a
    database. Any limit added here must stay consistent with GateConfig's
    own defaults - an override that restores a superseded limit defeats the
    gate's calibration while every unit test still passes.
    """
    return GateConfig(
        symbol_whitelist=tuple(
            s.strip().upper()
            for s in os.environ.get("EXECUTOR_SYMBOLS", "XAUUSD,XAGUSD").split(",")
            if s.strip()),
        default_risk_pct=float(os.environ.get("EXECUTOR_RISK_PCT", "0.5")),
        max_concurrent_positions=int(os.environ.get("EXECUTOR_MAX_POSITIONS", "3")),
        max_daily_loss_r=float(os.environ.get("EXECUTOR_MAX_DAILY_LOSS_R", "3")),
        # Spread is now judged against the stop it protects. The absolute
        # points value stays only as a broken-quote backstop, so its default
        # must match GateConfig's, not the superseded 40.
        max_spread_points=float(os.environ.get("EXECUTOR_MAX_SPREAD_PTS", "100000")),
        max_spread_frac_of_stop=float(
            os.environ.get("EXECUTOR_MAX_SPREAD_FRAC", "0.10")),
    )
```

Replace the inline construction in `main()` with `cfg = build_gate_config()`.

- [ ] **Step 7: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS.

---

## Task 5: The cross-instrument guard

The bug's whole character was that one config was silently correct for silver
and wrong for gold. A guard that only ever checks one instrument would not have
caught it, and would not catch the next one.

**Files:**
- Create: `tests/test_gate_scale_calibration.py`

- [ ] **Step 1: Write the guard**

```python
"""One GateConfig must be correct across every price scale it whitelists.

The gate's limits were absolute point counts. `max_stop_points = 5000` is
4.61% of EUR/USD's price, 7.55% of silver's, and 0.113% of gold's - so the
same config passed silver and rejected every realistic gold trade. Because
one of the two whitelisted symbols worked, the config looked calibrated.

Live specs and ATR14 (D1, 20 bars) from the Exness terminal, 2026-09-06.
"""
from dataclasses import replace

import pytest

from src.execution.gate import (
    AccountState, GateConfig, MarketSnapshot, Signal, run_gate,
)

# symbol -> (bid, ask, point, digits, atr14, a realistic stop in price units)
INSTRUMENTS = {
    "XAUUSD": (4430.194, 4430.376, 0.001, 3, 104.37, 20.0),
    "XAGUSD": (66.201, 66.222, 0.001, 3, 2.5539, 1.00),
    "EURUSD": (1.16269, 1.16279, 0.00001, 5, 0.0042, 0.0050),
}


def _snap(sym):
    bid, ask, point, digits, atr, _ = INSTRUMENTS[sym]
    return MarketSnapshot(
        symbol=sym, bid=bid, ask=ask, point=point, digits=digits,
        tick_value=1.0, tick_size=point, volume_min=0.01, volume_step=0.01,
        volume_max=200.0, trade_allowed=True, stops_level_points=0.0, atr=atr)


def _acct():
    return AccountState(balance=10_000, equity=10_000, free_margin=9_000,
                        open_positions=0, enabled=True, dry_run=True)


def _cfg(sym):
    """Default config, but whitelisting the symbol under test.

    The shipped whitelist is metals-only. Without this, every EUR/USD case
    would carry a whitelist block, and a test asserting "no stop reason" would
    pass while the signal was in fact rejected - a guard green for the wrong
    reason.
    """
    return GateConfig(symbol_whitelist=(sym,))


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_a_realistic_trade_passes_on_every_scale(sym):
    """Entry at market, an ordinary stop, 2R target - must not be blocked
    for any reason involving stop size, spread or distance from market."""
    snap = _snap(sym)
    stop_dist = INSTRUMENTS[sym][5]
    sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                 entry=snap.ask, stop=snap.ask - stop_dist,
                 tp1=snap.ask + 2 * stop_dist)
    res = run_gate(sig, snap, _acct(), _cfg(sym))
    scale_reasons = [r for r in res.reasons
                     if any(k in r for k in ("stop", "spread", "from market"))]
    assert not scale_reasons, f"{sym}: {scale_reasons}"


@pytest.mark.parametrize("sym", sorted(INSTRUMENTS))
def test_an_absurd_stop_is_blocked_on_every_scale(sym):
    """4x ATR is past the 3x ceiling regardless of instrument."""
    snap = _snap(sym)
    sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                 entry=snap.ask, stop=snap.ask - 4 * snap.atr,
                 tp1=snap.ask + 8 * snap.atr)
    res = run_gate(sig, snap, _acct(), _cfg(sym))
    assert any("ATR" in r for r in res.reasons), f"{sym}: {res.reasons}"


def test_the_limits_carry_no_absolute_price_assumption():
    """No primary limit may be an absolute point count.

    This is the regression that would have caught the original bug: a point
    count cannot be correct on two instruments whose price differs 66x while
    sharing a `point`.
    """
    cfg = GateConfig()
    assert cfg.max_stop_atr_mult > 0
    assert cfg.max_entry_deviation_atr_mult > 0
    assert cfg.max_spread_frac_of_stop > 0
    # The absolute forms survive only as fallbacks, and must be loose enough
    # never to fire on a normal metals spread (gold's is 182 points).
    assert cfg.max_spread_points > 1_000


def test_gold_and_silver_are_treated_alike_despite_a_66x_price_gap():
    """The exact asymmetry that hid the bug."""
    results = {}
    for sym in ("XAUUSD", "XAGUSD"):
        snap = _snap(sym)
        stop_dist = INSTRUMENTS[sym][5]
        sig = Signal(signal_id="s", symbol=sym, direction="BUY",
                     entry=snap.ask, stop=snap.ask - stop_dist,
                     tp1=snap.ask + 2 * stop_dist)
        results[sym] = run_gate(sig, snap, _acct(), GateConfig()).ok
    assert results["XAUUSD"] == results["XAGUSD"], results
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_gate_scale_calibration.py -v --no-cov`
Expected: PASS, **8 tests** (two parametrised over three symbols, plus two
standalone).

- [ ] **Step 3: Prove it would have caught the original bug**

**Restoring the old *config values* is not enough — you must restore the old
*state*.** Setting `max_stop_atr_mult=0.0` does not revert to the point-based
path: `run_gate` prefers the ATR branch whenever `snap.atr > 0`, so a 0.0
ceiling blocks every instrument and the asymmetry disappears into universal
failure. Before Task 2, ATR was not plumbed at all — the true pre-fix state is
`atr = 0.0`.

In a scratch script **outside the repo** (the repo's own `scratchpad/` is not
gitignored), run Task 5's realistic-trade signals with `atr` forced to `0.0`
against `GateConfig(max_spread_points=40.0, max_stop_points=5000.0,
symbol_whitelist=(sym,))`. Delete the script afterwards.

Measured 2026-09-06 — the asymmetry that defines the bug:

```
EURUSD: ok=True   reasons=[]
XAGUSD: ok=True   reasons=[]
XAUUSD: ok=False  reasons=['stop 20000pts above maximum 5000',
                           'spread 182pts above 40']
```

Gold blocked, silver and EUR/USD passing, with the exact block messages quoted
in Context. That is the regression this guard now prevents.

---

## Deliberately out of scope

**`looks_synthetic` is not touched by this plan**, though it is the fourth block
on the gold trade above. Two defects are recorded here for a separate decision:

1. **Its grid is scaled by `point`, not price.** `grid = snap.point * 1000` is
   $1.00 on gold, so any entry and stop on a $0.50 boundary is flagged. Round
   dollar levels are *real* levels on gold, not stub defaults. The docstring
   also mis-states the arithmetic — it says "for gold at 0.01 point it is $10",
   but gold's point is `0.001`, making the grid $1.00.

2. **The R-multiple condition is dead code.** The docstring says levels are
   flagged when they are round **and** the R-multiple is a clean integer, but
   both branches `return True`, so roundness alone is sufficient.

These are left alone because the standing constraint is that **the gate is never
weakened to let a blocked signal through**, and `looks_synthetic` is a safety
heuristic rather than a units bug — narrowing it is a judgment call for the
owner, not a mechanical fix. Note that Task 3 substantially reduces the need for
it: the motivating stub (1.10000 against a 1.16279 market) is caught by the
ATR-relative deviation check at 14.9× ATR.

---

## Verification

Evidence before claims.

1. **The four blocks become zero.** Re-run the exact signal from Context —
   `entry=4420, stop=4400, tp1=4460` on the live gold snapshot with
   `atr=104.37` — and record `res.reasons`. Expect only the `looks_synthetic`
   block to remain, and say so explicitly.
2. **Silver and EUR/USD are unchanged for realistic trades** — Task 5's
   parametrised test, all three instruments.
3. **The stub is still blocked** — `test_the_stub_entry_is_still_blocked`.
4. **The fallback path is exercised** — `test_falls_back_to_points_when_atr_unknown`
   proves `atr=0` reverts to today's limits rather than skipping the check.
5. **The runtime config matches the gate's own defaults.** Print
   `build_gate_config()` with no `EXECUTOR_*` variables set and confirm
   `max_spread_points` is the backstop, not 40. A limit fixed in `GateConfig`
   and overridden in the executor is fixed nowhere — this is the check that
   distinguishes the two.
6. **The guard would have caught the original bug** — Task 5 Step 3's output.
7. **Full suite:** `python -m pytest -q` — the 2 known GARCH failures, no
   third. Baseline was 2073 before this plan; Task 1 took it to **2075**, and
   each later task adds its own tests, so compare the *set* of failures, never
   the count.
8. **`gate.py` is still pure:** `grep -nE "^(import|from)" src/execution/gate.py`
   shows only `math`, `dataclasses`, `datetime`.
9. Show the owner the diff. **Never commit.**

## Notes the owner must act on

- **The executor is still disarmed** and unwired from `docker-compose.yml`.
  This plan does not arm it; that is a separate decision.
- **`max_spread_frac_of_stop = 0.10` is a first calibration.** Normal spreads
  sit at 0.9% of stop (gold, $20), 2.0% (EUR/USD, 50 pips) and 2.1% (silver,
  $1.00), giving 11× / 5× / 4.8× headroom respectively. Measured: a 10× gold
  blowout still passes at 9.1%; ~11× blocks. If news spikes need catching
  sooner, tighten toward 0.05 — but measure the actual spread during an NFP
  print first rather than guessing. Note this limit scales with the *stop*, so
  a tight scalp is protected more aggressively than a wide swing, which is the
  intended behaviour.
- **ATR14 on D1 is a choice.** A 4H setup may want 4H ATR. The field is a plain
  number on the snapshot, so changing which timeframe fills it is a one-line
  change in `snapshot()`.
