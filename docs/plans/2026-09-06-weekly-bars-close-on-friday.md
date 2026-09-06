# Weekly Bars Close On Friday Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a settled weekly bar's Close deterministic, by resampling FX
weeklies week-ending-Friday instead of week-ending-Sunday — closing the repaint
that let one bar produce two opposite signals.

**Architecture:** One rule change in `weekly_ohlc()`, the layer all five
affected sources share. Everything else follows from it.

**Tech Stack:** Python 3.14, pandas, pytest.

**Spec:** The owner's decision, 2026-09-06, choosing option A after the
measurement below.

## Global Constraints

- **Never commit.** The owner reviews and commits.
- This plan takes **1.10.58**.
- **This changes values, not just behaviour.** Every weekly bar's label and Close
  moves. That is the point, and it is why the before/after is recorded rather
  than asserted.
- **The in-progress week stays included** as a partial bar — the documented
  convention in `weekly_ohlc`, and unrelated to this fault.
- **Do not touch `monthly_ohlc`.** It has the same flaw and is out of scope; see
  below.

---

## Context

`biased_pivots` produced Long at 10:18 and Short at 11:04 on 2026-09-05, both
stamped `bar_time = 2026-08-23`. The prices tell the whole story:

```
2026-08-21 (Fri)  Close 1.91631   <- the 11:04 read
2026-08-23 (Sun)  Close 1.90280   <- the 10:18 read
```

Both used the weekly bar **labelled 2026-08-23**. Pandas `"W"` means
week-ending-**Sunday**, so FX's partial Sunday session lands at the *end* of the
week and becomes its Close. When that row is present the week closes at 1.90280;
when absent, at Friday's 1.91631 — opposite sides of a pivot zone.

**The row genuinely flickers.** The daily history holds **4 Saturday and 4 Sunday
bars in 220**, present only in recent weeks and not consistently even there
(2026-08-09, 08-16, 08-23 and 09-06 have Sundays; 08-30 does not). So a *settled*
weekly bar can still gain or lose its final day, while its label never changes —
which is why `bar_time` looked stable and every downstream dedupe key was
satisfied.

**Measured, both rules, same data:**

```
W      weeks whose Close changes if weekend bars vanish: 5 of 43
W-FRI  weeks whose Close changes if weekend bars vanish: 0 of 43
```

`W-FRI` puts the Sunday session at the *start* of the following week, where it
belongs by FX convention — the week closes Friday, the new week opens Sunday
evening — and the instability goes to zero.

**Why an end-of-window coverage check was the wrong fix.** That was the first
proposal, and the evidence refutes it: the frame is not truncated. The missing
row sits *inside* the history, so a recency test on `df.index.max()` would pass
and change nothing. Recorded because the plan it came from
(`2026-09-06-biased-pivots-repaint-hunt.md`) still names it.

**Blast radius.** `weekly_ohlc` has **10 consumers**, including
`setup_ranker`, `bias_service` and `background_scanner` — the live signal path.
Every weekly bar's label and Close changes:

```
W      2026-08-30 -> 1.88986        W-FRI  2026-09-04 -> 1.87853
```

Charts the desk compares against will disagree until they are re-read on the same
convention. That is a migration, not a quiet fix.

**Out of scope, deliberately:** `monthly_ohlc` resamples to month-end and has the
same flaw — **1 of 11 monthly closes** moves if weekend bars vanish. It needs the
same treatment and its own plan; bundling it would double the surface of a
values-changing review.

---

## Task 1: The guarantee, then the rule

**Files:**
- Modify: `src/services/market_data.py`
- Test: `tests/test_weekly_resample_stability.py`

- [ ] **Step 1: Write the failing test** — build a daily frame spanning several
      weeks *with* weekend rows, resample, then drop the weekend rows and
      resample again: **every settled week's Close must be identical**. Under
      `"W"` it is not.
- [ ] **Step 2: Run, watch it fail** on the weeks whose Sunday row was removed.
- [ ] **Step 3: Implement** — `_resample(df, "W-FRI")` in `weekly_ohlc`, with the
      reason recorded at the call site.
- [ ] **Step 4: Green.** Also assert weekly labels fall on Fridays.

---

## Task 2: Record what moved

**Files:** none modified

- [ ] **Step 1:** For a sample of live instruments, print the last 3 weekly
      closes under both rules, before/after, and save it beside the plan.
- [ ] **Step 2:** Re-read `biased_pivots` for GBP/AUD and confirm one bar now
      yields one answer regardless of the weekend row.

---

## Verification

1. **0 of N weeks move** when weekend bars are removed — the number that was 5.
2. **Weekly labels are Fridays.**
3. **The repaint case is closed:** GBP/AUD's 2026-08-23 week resolves to a single
   Close whether or not the Sunday row is present.
4. **Full suite**, known GARCH failures and no third. Any *other* test that moves
   is a real consequence of the migration and is reported, not silently updated.
5. Show the owner the diff. **Never commit.**
