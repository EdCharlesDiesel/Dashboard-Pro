---
name: confirm-bias
description: Use when a shortlist exists and each candidate needs its timeframes checked - confirms weekly to daily to 4H alignment and filters the day's tradeable set.
---

# Confirm Bias

Runbook steps 3–5. Takes the shortlist and asks, per candidate, whether the
timeframes actually agree.

## Steps

1. **Second opinion.** Cross-check the candidate against the house view / MTF
   matrix. Where the board disagrees with the ranker, that disagreement is the
   finding — report it rather than picking the one you prefer.
2. **Filter the day.** Session, liquidity, and the news calendar. A setup an
   hour before high-impact data is not tradeable yet, whatever it scored.
3. **Alignment, in order: weekly → daily → 4H.** The higher timeframe sets the
   permission; the lower one only times it.

## The refusal that matters

**If the timeframes disagree, the setup does not pass — and you must name the
dissenter.** "Weekly and daily are long, 4H structure is against it" is useful.
"Mixed signals" is not. A dissenting 4H on a strong weekly is a *wait*; a
dissenting weekly is a *no*.

## Report

Per candidate: pass or fail, which timeframes agree, which dissents, and any
news blackout window. Then the filtered set that survives.

## Stop

Confirmed candidates carry no size yet. `size-risk` is next.
