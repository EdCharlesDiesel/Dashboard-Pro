---
name: execute-and-log
description: Use when a GO verdict exists and the owner has explicitly approved this specific trade - places it through the MT5 gates and records the decision for review.
---

# Execute and Log

Runbook steps 9–10. **The only skill on this desk that moves money.**

## Preconditions — all of them, no exceptions

- [ ] `run-checklist` returned **GO**
- [ ] The owner has seen the plan and approved **this specific trade**
- [ ] The owner has supplied `confirm=true` in words for this order
- [ ] Symbol, direction, volume and stop stated back and agreed
- [ ] The four gates hold (`.claude/reference/2-mt5-tooling.md`)

If any is missing, stop and name it. **Never supply `confirm=true` on the
owner's behalf**, and never treat an earlier general "go ahead" as approval for
a later specific order.

## Execute

1. State symbol, direction, volume, stop and target back one final time.
2. Place through the MT5 trade tools. `order_check` dry-runs first; a request
   the broker would reject raises rather than sending.
3. Read the result. Report the fill price and compare it to the arrival price —
   the mid when the trade was first discussed. That difference is the execution
   quality, and it is the only number that measures this step.

## Log

- The book updates itself: the MT5 sync writes `open_positions.json`.
- Record the **score at decision time** to `setup_ranker_score_history.json`.
  Without it, a post-mortem can only argue about hindsight.
- `logs/mt5_tool_audit.jsonl` records every call automatically, via the
  PostToolUse hook. Nothing to do.

## Review — step 10

Same evening or next morning, dispatch `trade-reviewer` on anything that
closed. It returns one of `thesis-correct` / `thesis-wrong` / `execution-error`
/ `invalidated-by-news`. The label is the point: it separates a bad trade from
bad luck, and only one of those is worth changing the process over.
