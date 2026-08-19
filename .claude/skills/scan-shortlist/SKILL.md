---
name: scan-shortlist
description: Use when starting a trading session and you need the day's candidate setups - reads the macro backdrop and the stored Setup Ranker scores into a ranked shortlist.
---

# Scan Shortlist

Runbook steps 1–2 (`docs/Live_Session_Runbook.md`). Establishes the backdrop,
then produces the shortlist everything downstream works from.

## Steps

1. **Macro backdrop and risk regime first.** Risk-on or risk-off, the session
   about to open, and what is on the calendar. A shortlist read without the
   backdrop is a list of chart patterns.
2. **Read the stored scores — do not rescan.** The `scanner` container writes
   the board every 300s. Re-running the scan from a session costs minutes and
   returns the same numbers.
3. Rank by grade, then by whether the quality gate passed. See
   `.claude/reference/1-scoring-criteria.md`.
4. Check the age of what you read. If the board or the book is stale, say so
   with the timestamp rather than presenting stale numbers as current.

## Report

Grade counts (A/B/C/D), the shortlist with direction and score, which entries
failed the quality gate, and the backdrop in two lines.

## Stop

The shortlist is a list of candidates, not decisions. Nothing here is confirmed
and nothing is sized — `confirm-bias` comes next.
