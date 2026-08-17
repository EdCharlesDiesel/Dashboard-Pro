---
name: run-checklist
description: Use when a sized setup is ready and needs the final GO decision - runs the daily checklist gate, where a partial pass is a no.
---

# Run Checklist

Runbook step 8. The gate. Everything upstream produces candidates; this decides
whether any of them trades tonight.

## The gate

Every item passes, or the setup does not go. **A partial pass is a no.** The
checklist exists precisely for the evening when the setup looks good and one
item does not clear — that is the case it was written for.

| # | Item | Fails when |
|---|---|---|
| 1 | Direction score and grade | Grade C or D |
| 2 | Quality gate | Any of ATR Volatile / 4H Zone / Spread/ATR fails |
| 3 | Timeframe alignment | Weekly, daily or 4H dissents (`confirm-bias`) |
| 4 | Risk budget | `actual_risk` above the per-trade ceiling |
| 5 | Correlated exposure | Adds a second position on a currency leg already held |
| 6 | Margin level | Below the warning threshold, or a new position takes it there |
| 7 | News blackout | High-impact event inside the window |
| 8 | Stop defined | No ATR or structural stop, or a stop the account cannot survive |
| 9 | R:R | Below 1:2 |
| 10 | Data freshness | Balance or book older than 15 minutes |

Dispatch the `risk-auditor` agent for items 4–8; it returns PASS/FAIL per rule
with the offending number.

## Report

The table, with the failing number quoted for each FAIL. Then a plain verdict:
**GO** or **NO GO**, and for NO GO, the single item that was decisive.

## When nothing clears

Say so plainly: "Nothing reaches GO tonight." That is a successful run of this
skill, not a failure of it. Do not soften a NO GO into a smaller position — the
checklist is not a scale.

## Stop

A GO verdict is permission to *present* the trade, not to place it.
