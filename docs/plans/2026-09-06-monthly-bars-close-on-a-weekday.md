# Monthly Bars Close On A Business Day Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `monthly_ohlc` the same determinism `weekly_ohlc` got at 1.10.58 —
a settled monthly bar's Close must not depend on a weekend row that may never
arrive.

**Architecture:** Resample `BME` (business month end) instead of `ME` — the
direct analogue of the `W-FRI` fix, moving the bin boundary rather than dropping
rows.

**Tech Stack:** Python 3.14, pandas, pytest.

**Spec:** The owner's request, 2026-09-06: *"fix monthly too"*, after 1.10.58
left `monthly_ohlc` carrying the same flaw.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.59**.
- **Weekly is not revisited.** `W-FRI` already fixed it and keeps the Sunday
  session as the next week's open; monthly cannot use that mechanism.
- **The in-progress month stays included** as a partial bar — existing,
  documented, unrelated.

---

## Context

`weekly_ohlc` was fixed by moving the *bin boundary*: `W-FRI` puts FX's
flickering Sunday session at the start of the following week, so it stops being
any week's Close while remaining in the data.

**The same mechanism does work for months** — `BME` (business month end) moves
the bin boundary to the last business day, so a month-ending Sunday falls into
the *next* month. Constructed settled months with month-ending weekend rows:

```
ME   settled months that move when the weekend row vanishes: ['2026-05-31']
BME  settled months that move: none        (label becomes Fri 2026-05-29)
```

**A wrong turn worth recording.** Measured against live data, both rules showed
"1/61 moved", which read as *BME does not help* and led to a proposal to drop
weekend rows entirely. That measurement was confounded: the one month moving was
**2026-09-30, the in-progress month**, whose last row is Sunday 09-06. A partial
month's Close moving as days arrive is by design, not a repaint. Testing the
*settled* case explicitly reversed the conclusion — and the discarded proposal
would have thrown away weekend price data for no benefit.

**Monthly is not currently broken.** The live 5-year window holds exactly one
weekend row (2026-09-06), so no settled month-end has ever carried one. This
change closes a latent exposure rather than an active fault — the same fault
weekly *was* actively suffering.

---

## Task 1: Weekday-only monthly bars

**Files:**
- Modify: `src/services/market_data.py`
- Test: `tests/test_weekly_resample_stability.py`

- [ ] **Step 1: Write the failing test** — resample a daily frame with weekend
      rows and again without them; every settled month's Close must match. Under
      today's code it does not.
- [ ] **Step 2: Run, watch it fail.**
- [ ] **Step 3: Implement** — filter `dayofweek < 5` before `_resample(df, "ME")`,
      with the reason recorded at the call site.
- [ ] **Step 4: Green**, and assert monthly labels still land on month-ends.

---

## Verification

1. **0 of N months move** when weekend rows are removed.
2. **The measured migration cost** is reported per instrument, not assumed.
3. **Full suite**, known GARCH failures and no third. Any other test that moves
   is a real consequence and is reported, not silently updated.
4. Show the owner the diff. **Never commit.**
