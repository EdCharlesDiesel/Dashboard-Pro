# Biased Pivots Repaint Hunt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find and close the path by which `biased_pivots` produced two opposite
signals from the *same* bar 46 minutes apart, and add the guard that would have
caught it — the module was already fixed for this once and it has come back.

**Architecture:** Diagnose before changing anything. The page and `read()` both
look correct on inspection, so the fault is in the data they are handed:
`weekly_ohlc()` resamples cached daily bars, so a "closed" weekly bar's Close is
*derived*, not stored, and can move when the daily window underneath it changes.
The fix follows the evidence; the guard does not.

**Tech Stack:** Python 3.14, pandas, yfinance-backed cache, Postgres, pytest.

**Spec:** The owner's question, 2026-09-06: *"why is biased_pivots reversing on 4
pairs"*, and the answer found in the data — 3 were legitimate, 1 was a repaint.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.57**.
- **Diagnose first.** No fix is written before a reproduction exists. The
  module's docstring already records one fix for this symptom; a second guess
  would be the third.
- **Do not "correct" the pivot maths.** The `shft`/`shft+1` close pairing is a
  deliberate port of `BiasedPivots.mq5` and is documented as not-a-bug.
- **The executor can act on this source.** A signal that says Long at 10:18 and
  Short at 11:04 off one bar is one the queue would happily accept, so this is
  not cosmetic.

---

## Context

Asked why `biased_pivots` reversed on 4 pairs this week, the data separates into
two unrelated things:

| Pair | 1st read | 2nd read | `bar_time` | Verdict |
|---|---|---|---|---|
| GBP/CAD | 09-05 10:18 Short | 09-06 08:30 Long | 08-23 → **08-30** | new bar — legitimate |
| GBP/ZAR | 09-05 10:18 Long | 09-06 08:30 Short | 08-23 → **08-30** | new bar — legitimate |
| WTI/USD | 09-05 10:18 Short | 09-06 08:41 Long | 08-23 → **08-30** | new bar — legitimate |
| **GBP/AUD** | 09-05 **10:18 Long** | 09-05 **11:04 Short** | **08-23 → 08-23** | **repaint** |

A weekly pivot giving a different answer on a new week is the indicator working.
GBP/AUD is not that: **same `bar_time`, opposite direction, 46 minutes apart** —
and two different entry prices off a bar that closed a fortnight earlier.

```
GBP/AUD  Long   09-05 10:18  bar_time 2026-08-23  entry 1.90280  PP 1.90946
GBP/AUD  Short  09-05 11:04  bar_time 2026-08-23  entry 1.91631  PP 1.90956
```

**This is the bug the module was already fixed for.** Its own docstring:

> *"it repainted the direction instead of the levels: the sweep re-reads every 5
> minutes, so as price drifted across a zone boundary the same bar produced
> opposite calls. Measured 2026-08-11 — AUD/USD long at 23:43 and short 40
> minutes later, both stamped `bar_time = 2026-08-09`… `biased_pivots`
> contradicted itself on **20 of 27 instruments** this way, more than any other
> source… One bar, one answer."*

That fix made the judged price the **closed bar's Close**. Yet one bar produced
two prices, so the closed bar's Close was not constant between the two reads.

**Where it cannot be.** Inspected and clean:

- `biased_pivots_page.py:122-123` persists `r["direction"]` and `r["price"]` —
  both straight from `read()`, no live quote.
- `read()` sets `price = period["Close"]` where `period = df.iloc[idx]`.

**Where it must be: the frame.** `weekly_ohlc()` does not fetch weekly bars — it
pulls **daily** bars from `cached_ohlc(..., ttl=CANONICAL_TTL)` and resamples to
`W`, deliberately, and *"the in-progress week is included as a partial bar"*. So
a weekly bar's Close is derived from whatever daily rows the cache held at that
moment. If that daily window changes — a refresh, a revision, a short fetch —
a *past* week's Close can move even though its label does not.

That would produce exactly this signature: stable `bar_time`, moving price,
flipped direction — and it would be invisible to every test that feeds `read()`
a fixed DataFrame.

---

## Task 1: Reproduce it

**Files:** none modified — investigation only

- [ ] **Step 1:** Fetch `weekly_ohlc("GBPAUD=X")` and record the full row for
      `2026-08-23`: Open/High/Low/Close, plus `settled_period_index`'s choice.
- [ ] **Step 2:** Print the **daily** rows the cache holds for 2026-08-17..23 and
      confirm whether their last Close equals the weekly Close. A mismatch here
      is the answer on its own.
- [ ] **Step 3:** Force a cache refresh, re-read, and diff the same weekly row.
      **A changed Close on a fortnight-old bar is the reproduction.**
- [ ] **Step 4:** If the frame is stable, the fault is upstream of it — walk
      `cached_ohlc` for a short/partial fetch path instead, and record that.
- [ ] **Step 5:** Write down which hypothesis the evidence supports **before**
      touching code.

---

## Task 2: Guard the invariant, whatever the cause

**Files:** Create `tests/test_biased_pivots_stability.py`

The guard is worth having regardless of root cause, and must be written before
the fix so it is seen to fail.

- [ ] **Step 1: Write the failing test** — for a fixed daily frame, two
      successive `read()` calls return the same direction *and* the same price;
      and appending a **new forming day** to the frame does not change the answer
      for an already-settled bar.
- [ ] **Step 2: Run, watch it fail** if the second case reproduces; if it passes,
      the instability is in the cache, not `read()`, and Task 3 targets there.
- [ ] **Step 3: Green after the fix.**

---

## Task 3: Fix at the layer the evidence names

**Files:** decided by Task 1 — `src/services/market_data.py` or
`src/db/market_cache.py` or `src/core/biased_pivots.py`

- [ ] **Step 1:** Implement the smallest change that makes a settled bar's OHLC
      immutable for a given `bar_time`.
- [ ] **Step 2:** Re-run the Task 2 guard.
- [ ] **Step 3:** Re-run the persisted-signal check: no `(pair, bar_time)` may
      carry two directions.

---

## Task 4: Make the weekly page tell these apart

**Files:** Modify `src/core/weeks_trades.py`, `src/pages_lib/weeks_trades_page.py`

- [ ] **Step 1:** Split "self-reversed" into **new-bar** (legitimate) and
      **same-bar** (repaint), keyed on `checks_detail->>'bar_time'`.
- [ ] **Step 2:** Surface same-bar reversals as the alarm; new-bar as
      information.

This is the reporting half of the same lesson: one real defect sat among three
innocuous reversals and was indistinguishable from them, exactly as 17 cases of
routine source disagreement had previously hidden all four.

---

## Verification

1. **A reproduction exists** and is recorded, before any fix.
2. **The guard fails before the fix and passes after.**
3. **No `(pair, bar_time)` in `trade_setups` carries two directions** for
   `biased_pivots`, checked in SQL over the last 30 days.
4. **The weekly page separates** same-bar from new-bar reversals.
5. **Full suite**, known GARCH failures and no third.
6. Show the owner the diff. **Never commit.**

## Note

The historical count matters for scope: this source contradicted itself on **20
of 27 instruments** in August. Today it is 1 of 27, so either the earlier fix is
mostly working and one path escapes it, or the conditions that trigger it are
rarer this week. The SQL check in Verification 3 over 30 days will say which,
and that determines whether this is a leak or a regression.
