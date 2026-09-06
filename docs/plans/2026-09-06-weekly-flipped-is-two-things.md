# Weekly "Flipped" Conflates Two Different Things Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `flipped` flag on This Week's Trades, which reports 22 of
27 pairs as "changed side" when the real number is 5, with two measures that say
what is actually happening: how the sources split, and which sources reversed
themselves.

**Architecture:** `PairWeek` gains `reversing_sources`; `flipped` is redefined to
mean same-source reversal, and the source split is surfaced as counts the page
renders directly. Pure aggregation only — `src/core/weeks_trades.py` and its
renderer.

**Tech Stack:** Python 3.14, Streamlit, pytest.

**Spec:** The owner's question, 2026-09-06: *"why are 22 of 27 pairs flipping
this week"* — asked of a number this page produced at 1.10.52.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.54**.
- **No change to what is stored or queried.** This is a reporting defect.
- `src/core/weeks_trades.py` stays pure and at 100% coverage.

---

## Context

The page reported **22 of 27 pairs "Changed side"**. The real count is **5**.

`flipped` was defined as `longs > 0 and shorts > 0` across the whole week,
ignoring which source spoke and when. Measured against this week's data:

```
pairs with cross-source disagreement : 22
pairs with same-source reversal      :  5
```

Those are different events. EUR/USD this week:

```
Long   daily_trend, market_structure, weekly_ema
Short  daily_macd, seasonality
```

Three sources long, two short, **at the same time**. Eighteen independent
indicators disagreeing is the normal state of a multi-source system — it is the
condition `consensus()` exists to resolve, not a warning. Reporting it as the
system changing its mind invents an alarm out of routine behaviour, and on a
trading dashboard a fabricated alarm is worse than no column at all.

**A boolean also destroys the informative part.** `AUD/USD 10/3` and
`EUR/USD 3/3` were both rendered "flipped: yes". The first is conviction with
one dissenter; the second is a coin toss. The split is the number worth reading.

**The genuine reversals are concentrated.** Of the 5 same-source reversals, 4 are
`biased_pivots` — one indicator producing 80% of them in a week. That is a real
observation about a source, and it was invisible under a per-pair boolean.

---

## Task 1: Measure the two things separately

**Files:**
- Modify: `src/core/weeks_trades.py`
- Test: `tests/test_weeks_trades.py`

**Interfaces:**
- `PairWeek` gains `reversing_sources: set[str]`
- `flipped` now means **a source reversed itself**, not "both sides appeared"

- [ ] **Step 1: Write the failing tests** — two sources disagreeing at the same
      time is **not** flipped; one source taking both sides **is**, and names
      that source; a pair with several reversing sources names them all; the
      long/short counts are unchanged so the split stays readable.
- [ ] **Step 2: Run, watch them fail** — the current definition reports the
      disagreement case as flipped, which is the defect.
- [ ] **Step 3: Implement** — track direction per source, mark reversal when one
      source shows both.
- [ ] **Step 4: Green**, coverage still 100%.

---

## Task 2: Say it on the page

**Files:** Modify `src/pages_lib/weeks_trades_page.py`

- [ ] **Step 1:** Replace the "Flipped" column with **Split** (`10/3`) and
      **Reversed** (the naming sources), so agreement and reversal are separate.
- [ ] **Step 2:** Change the headline metric from "Changed side" to
      "Self-reversed", counting the honest 5 rather than the misleading 22.
- [ ] **Step 3:** Caption stating plainly that sources disagreeing is normal and
      that only a source contradicting *itself* is flagged.
- [ ] **Step 4:** AppTest — no exception, non-zero widget count.

---

## Verification

1. **The number the owner asked about changes from 22 to 5**, against live data.
2. **Unit tests green**, including the simultaneous-disagreement case that the
   old definition got wrong.
3. **Coverage stays 100%** on `weeks_trades.py`.
4. **Full suite**, known GARCH failures and no third.
5. **Deployed**, and the page read back from the running container.
6. Show the owner the diff. **Never commit.**
