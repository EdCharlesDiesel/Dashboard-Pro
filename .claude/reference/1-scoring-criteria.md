# 1 — Setup Scoring Criteria

**Source of truth: `src/core/signals.py` (`score_setup`, `_QUALITY_CRITERIA`).**
This file names the criteria so an agent can talk about them. It deliberately
does not restate a single threshold — those live in code, and a copy here would
be wrong the first time anyone tuned one.

## Two scores, not one

The scorer returns a **direction** score and a **quality** score, and they
answer different questions.

### Direction — 9 criteria: is the read right?

| Timeframe | Criteria |
|---|---|
| Weekly | Weekly EMA · Weekly RSI · Weekly Structure |
| Daily | Daily Trend · Daily Structure · Daily MACD · Daily 200MA |
| 4H | 4H Structure |
| Cross-market | Currency Strength |

Reported as `n/9` and as a percentage.

### Quality — 3 criteria: is it tradeable?

`_QUALITY_CRITERIA` = **ATR Volatile · 4H Zone · Spread/ATR**

## Grades

| Grade | Direction score |
|---|---|
| **A** | >= 80% |
| **B** | 60–79% |
| **C** | 40–59% |
| **D** | < 40% |

## The rule that actually matters

**A Grade A that fails the quality gate is a good read on an untradeable
instrument.** The direction can be perfect while the spread eats the edge, or
the ATR says the stop has to sit so far away that no sane size fits the risk
budget. Never rank such a setup first because its grade is high — report the
grade *and* the quality gate, and say which one is the problem.

The ZAR crosses are the standing example: they score direction well and fail
`Spread/ATR` regularly. That is not a bug in the score, it is the score doing
its job.

## Related

- Scoring one instrument: dispatch the `setup-scorer` agent — one per dispatch.
- Turning a grade into a size: `4-execution-handoff.md`.
