# looks_synthetic Is Scale-Relative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the stub detector judge roundness against the instrument's price
magnitude rather than the broker's digit convention, and require both the
conditions its docstring has always claimed — so it stops rejecting the desk's
own round-number gold levels while still catching the placeholder signal
currently sitting in `pending_signals`.

**Architecture:** Two changes inside `looks_synthetic`, nothing else. The grid
is derived from the entry's order of magnitude instead of `point * 1000`, and
the dead `return True` that made the R-multiple test irrelevant is removed.

**Tech Stack:** Python 3.14, pytest. `src/execution/gate.py` is pure and stays
that way; `math` is already imported.

**Spec:** The owner's instruction, 2026-09-06: "fix looks_synthetic too",
following the scoping note in `2026-09-06-gate-limits-are-scale-relative.md`
which recorded both defects and deferred them for this decision.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.67**.
- **This narrows a safety check, which the standing rule normally forbids.** The
  owner has explicitly asked for it. The narrowing is therefore bounded by one
  non-negotiable requirement: **the stub in `pending_signals` must still be
  blocked**, and every task must show it. A change that lets that row through
  is a failure, not a trade-off.
- **`gate.py` stays pure** — no imports beyond `math`, `dataclasses`,
  `datetime`.
- `run_gate`'s call site and the `reject_suspicious_round` flag are unchanged.
  This plan changes what `looks_synthetic` *decides*, not how it is used.

---

## Context

`looks_synthetic` is the last remaining block on a legitimate gold trade after
1.10.66 fixed the three units bugs. It carries two defects.

### Defect 1 — the grid follows the broker, not the instrument

```python
grid = snap.point * 1000.0        # docstring: "for gold at 0.01 point it is $10"
```

The docstring's arithmetic assumes gold quotes at 2 digits. On this broker gold
has **`digits=3`, `point=0.001`**, so the grid is **$1.00** and the half-grid
**$0.50** — ten times finer than intended. The same instrument gets a different
verdict purely from how a broker chooses to quote it.

### Defect 2 — the R-multiple test is dead code

```python
    if sig.tp1 is not None and sig.stop_distance > 0:
        rr = abs(sig.tp1 - sig.entry) / sig.stop_distance
        if abs(rr - round(rr)) < 1e-6:
            return True
    return True          # <- reached whenever the branch above is not
```

Both paths return `True`, so roundness of entry and stop is sufficient on its
own. The docstring says "Both round **AND** the R-multiple to tp1 is a clean
integer". The code has never done that.

### What the real data says

The queue (`pending_signals`) holds exactly two rows:

| symbol | entry | stop | tp1 |
|---|---|---|---|
| XAUUSD | 4604.31 | 4589.77 | 4633.62 |
| EURUSD | 1.100000 | 1.095000 | 1.110000 |

The second is the placeholder from the docstring's own motivating case — it is
real, it is live, and it must keep being caught.

The journal (`trade_setups`) shows the false-positive side. Real gold entry
prices logged by the desk include `4482`, `4649` and `4626.5` — all of which sit
on the current $0.50 grid and would be rejected if their stop were also round.

Measured against those signals, current vs proposed:

| Signal | current | proposed | correct |
|---|---|---|---|
| EUR/USD stub (in queue) | flagged | flagged | flagged |
| XAU/USD real (in queue) | clean | clean | clean |
| XAU/USD 4482 (journal) | **flagged** | clean | clean |
| XAU/USD 4649 (journal) | **flagged** | clean | clean |
| XAU/USD 4626.5 (journal) | **flagged** | clean | clean |
| XAU/USD round levels, R = 2.15 | **flagged** | clean | clean |
| XAU/USD round levels, R = 2.00 exactly | flagged | flagged | flagged |
| XAG/USD measured | clean | clean | clean |
| EUR/USD measured | clean | clean | clean |

**current 5/9 correct, proposed 9/9.** The proposal is more *precise*, not more
permissive: both genuinely generated setups stay blocked.

### The replacement grid

```python
grid = 10.0 ** (math.floor(math.log10(abs(entry))) - 2)
```

Two orders of magnitude below the entry's leading digit, so "round" means the
same thing on every price scale:

| Instrument | price | grid | half-grid |
|---|---|---|---|
| EUR/USD | 1.16 | 0.01 | 0.005 |
| XAG/USD | 66 | 0.1 | 0.05 |
| XAU/USD | 4604 | 10 | 5 |

For EUR/USD this reproduces today's grid exactly, which is why the stub is still
caught.

---

## Task 1: Make the stub detector scale-relative

**Files:**
- Modify: `src/execution/gate.py` (`looks_synthetic`, lines 143–177)
- Test: `tests/test_execution_gate.py`

**Interfaces:**
- `looks_synthetic(sig, snap) -> bool` keeps its exact signature and call site.

- [ ] **Step 1: Write the failing tests**

Add a new class, matching the file's existing class-based style. `_gold_snap()`
and `_acct()` already exist — reuse them, do not redefine.

```python
class TestTheStubDetectorIsScaleRelative:
    """Roundness must mean the same thing on a $1.16 pair and a $4,600 one.

    The grid was `point * 1000`, whose docstring assumed gold quotes at 2
    digits. This broker quotes it at 3, making the grid $1.00 instead of $10
    and rejecting the desk's own round entries (4482, 4649, 4626.5 all appear
    in trade_setups).
    """

    def _snap(self, symbol, bid, ask, point, digits):
        return MarketSnapshot(
            symbol=symbol, bid=bid, ask=ask, point=point, digits=digits,
            tick_value=1.0, tick_size=point, volume_min=0.01,
            volume_step=0.01, volume_max=200.0, trade_allowed=True,
            stops_level_points=0.0)

    def test_the_live_stub_is_still_caught(self):
        """pending_signals holds EURUSD 1.100000/1.095000/1.110000 right now.

        This is the whole reason the check exists. If this ever goes green
        the narrowing has gone too far.
        """
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=1.100000, stop=1.095000, tp1=1.110000)
        assert looks_synthetic(sig, snap) is True

    def test_the_live_gold_signal_is_not_a_stub(self):
        """pending_signals also holds XAUUSD 4604.31/4589.77/4633.62."""
        snap = self._snap("XAUUSD", 4604.19, 4604.31, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4604.31, stop=4589.77, tp1=4633.62)
        assert looks_synthetic(sig, snap) is False

    @pytest.mark.parametrize("entry", [4482.0, 4649.0, 4626.5])
    def test_real_round_gold_entries_are_not_stubs(self, entry):
        """Every one of these is a real entry_price from trade_setups."""
        snap = self._snap("XAUUSD", entry - 0.1, entry, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=entry, stop=entry - 20.0, tp1=entry + 40.0)
        assert looks_synthetic(sig, snap) is False

    def test_round_levels_with_a_non_integer_r_are_not_stubs(self):
        """The docstring always said BOTH conditions were required."""
        snap = self._snap("XAUUSD", 4599.9, 4600.0, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4600.0, stop=4580.0, tp1=4643.0)   # R = 3.15
        assert looks_synthetic(sig, snap) is False

    def test_a_generated_gold_setup_is_still_caught(self):
        """Round to the $5 grid AND exactly 3R - the narrowing must not
        reach this."""
        snap = self._snap("XAUUSD", 4599.9, 4600.0, 0.001, 3)
        sig = Signal(signal_id="s", symbol="XAUUSD", direction="BUY",
                     entry=4600.0, stop=4580.0, tp1=4660.0)   # R = 3.00
        assert looks_synthetic(sig, snap) is True

    def test_a_missing_tp1_cannot_be_a_stub(self):
        """With no tp1 there is no R to be suspicious about."""
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=1.100000, stop=1.095000, tp1=None)
        assert looks_synthetic(sig, snap) is False

    def test_a_zero_entry_does_not_raise(self):
        """log10(0) is -inf; a malformed signal must not crash the gate."""
        snap = self._snap("EURUSD", 1.16269, 1.16279, 0.00001, 5)
        sig = Signal(signal_id="s", symbol="EURUSD", direction="BUY",
                     entry=0.0, stop=0.0, tp1=0.0)
        assert looks_synthetic(sig, snap) is False
```

Ensure `looks_synthetic` is imported in the test module's import list.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_execution_gate.py -k StubDetector -v --no-cov`

Expected: the three `test_real_round_gold_entries_are_not_stubs` cases FAIL
(`assert True is False`), and `test_round_levels_with_a_non_integer_r_are_not_stubs`
FAILS. `test_the_live_stub_is_still_caught` PASSES already — it must never have
failed, and must still pass at the end.

- [ ] **Step 3: Rewrite the function body**

Replace everything from `if snap.point <= 0:` to the closing `return True`:

```python
    # A stub is a level a human typed into a config, and the tell is that it
    # is round at a coarseness a *measured* level would essentially never hit.
    # The old grid was `snap.point * 1000`, whose docstring assumed gold quotes
    # at 2 digits; this broker quotes it at 3, so the grid came out $1.00
    # instead of $10 and rejected real entries (4482, 4649 and 4626.5 all
    # appear in trade_setups). Anchoring to the entry's order of magnitude
    # makes "round" mean the same thing on a $1.16 pair and a $4,600 one:
    #
    #     EUR/USD 1.16  -> grid 0.01   half 0.005   (identical to before)
    #     XAG/USD 66    -> grid 0.1    half 0.05
    #     XAU/USD 4604  -> grid 10     half 5
    if sig.entry == 0 or sig.stop_distance <= 0:
        return False
    grid = 10.0 ** (math.floor(math.log10(abs(sig.entry))) - 2)

    def on_grid(x: float, g: float) -> bool:
        return abs(x / g - round(x / g)) < 1e-6

    half_grid = grid / 2.0
    if not (on_grid(sig.entry, half_grid) and on_grid(sig.stop, half_grid)):
        return False

    # BOTH conditions, which is what the docstring has always claimed. The
    # previous version fell through to a bare `return True`, so the R test
    # never affected the answer and roundness alone was enough.
    if sig.tp1 is None:
        return False
    rr = abs(sig.tp1 - sig.entry) / sig.stop_distance
    return abs(rr - round(rr)) < 1e-6
```

Also correct the docstring: the grid is derived from the entry's magnitude, and
the worked example should read *"XAU/USD near 4600 gives a $10 grid"*. Keep the
existing "deliberately conservative" note — it is still true, just now measured
against price rather than the broker's digits.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_execution_gate.py -v --no-cov`
Expected: PASS, including every pre-existing test.

`TestTheGateRefuses::test_round_grid_levels_are_treated_as_placeholders`
(`entry=2400.0, stop=2390.0, tp1=2420.0`) was checked against the new grid
before this plan was written and **still flags**: grid 10, half-grid 5, both
levels on it, R exactly 2.00. It must pass unchanged — **do not edit it.** If it
fails, the implementation is wrong, not the test.

- [ ] **Step 5: Confirm the live queue verdicts**

Run this and paste the output into the report:

```python
python - <<'PY'
from src.execution.gate import Signal, MarketSnapshot, looks_synthetic
def snap(sym, bid, ask, point, digits):
    return MarketSnapshot(symbol=sym, bid=bid, ask=ask, point=point,
        digits=digits, tick_value=1.0, tick_size=point, volume_min=0.01,
        volume_step=0.01, volume_max=200.0, trade_allowed=True,
        stops_level_points=0.0)
rows = [("XAUUSD", snap("XAUUSD",4604.19,4604.31,0.001,3), 4604.31,4589.77,4633.62),
        ("EURUSD", snap("EURUSD",1.16269,1.16279,0.00001,5), 1.1,1.095,1.11)]
for name, s, e, st, tp in rows:
    sig = Signal(signal_id="x", symbol=name, direction="BUY", entry=e, stop=st, tp1=tp)
    print(f"{name}: looks_synthetic={looks_synthetic(sig, s)}")
PY
```

Expected exactly: `XAUUSD: looks_synthetic=False`, `EURUSD: looks_synthetic=True`.

---

## Verification

Evidence before claims.

1. **The live stub is still blocked** — Step 5's output must show
   `EURUSD: looks_synthetic=True`. This is the gate-keeping check for the whole
   plan.
2. **The gold trade from 1.10.66's Context is now clean.** Re-run it end-to-end
   and record `res.reasons`:
   `entry=4600.0, stop=4580.0, tp1=4643.0` (R = 3.15) on the gold snapshot with
   `atr=104.37` must return `ok=True` with an empty reason list.
3. **A generated gold setup is still caught** —
   `test_a_generated_gold_setup_is_still_caught`.
4. **Full suite:** `python -m pytest -q --no-cov` — the 2 known GARCH failures
   in `tests/test_quant_models.py`, no third. Compare the *set* of failures,
   never the count.
5. **`gate.py` is still pure:** `grep -nE "^(import|from)" src/execution/gate.py`
   shows only `math`, `dataclasses`, `datetime`.
6. **Version:** `python deploy/sync_version.py 1.10.67`, then
   `python deploy/sync_version.py --check`.
7. Show the owner the diff. **Never commit.**

## Notes the owner must act on

- **This is a real narrowing.** A round-numbered setup with a non-integer R is
  now allowed through where it previously was not. That is the intended effect
  and it is what makes the desk's own gold levels tradeable, but it is a
  reduction in what the gate rejects. The compensating controls are unchanged:
  the ATR-relative entry-deviation check from 1.10.66 catches levels far from
  market, which is the shape the original placeholder actually had.
- **The stub row is still sitting in `pending_signals`** (`EURUSD 1.100000 /
  1.095000 / 1.110000`, created 2026-08-23). It is blocked, not removed. Worth
  purging so it stops being re-evaluated on every executor poll.
- **`looks_synthetic` remains a heuristic.** It cannot distinguish a trader who
  genuinely wants a round entry and a 2R target from a generated one. It is
  deliberately biased toward rejecting; the docstring's "far cheaper to
  hand-check a false positive than to auto-execute a stub" still governs.
