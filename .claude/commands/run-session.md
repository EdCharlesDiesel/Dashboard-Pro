---
description: Run the desk's live session end to end - scan, confirm, size, gate, present - stopping before anything is placed.
argument-hint: [optional: instruments or "all"] [optional: risk %, default 1.00]
---

# Run Session

Live session for: $ARGUMENTS

This is `docs/Live_Session_Runbook.md` executed. Read it alongside this command
— if the two ever disagree, the runbook is right and this file is stale.

1. **Scan** — `scan-shortlist`. Macro backdrop and risk regime, then the ranked
   shortlist from the stored board. Report grade counts and the backdrop before
   going further; if nothing is Grade A or B, say so and stop.

2. **Confirm** — `confirm-bias` per surviving candidate. Weekly → daily → 4H.
   Name every dissenting timeframe. Drop what does not align.

3. **Size** — `size-risk` per confirmed setup. Check the balance's age first.
   Report `actual_risk`, not the target risk.

4. **Audit** — dispatch `risk-auditor` on each sized setup. Read every FAIL
   with its number.

5. **Gate** — `run-checklist`. GO or NO GO, and the decisive item for each NO GO.

6. **Plan** — `trade-plan-writer` for each GO, five headers with conviction and
   counter-indicators.

7. **STOP AND PRESENT.** Show the owner: the backdrop, the shortlist with
   grades, what failed and why, the surviving plans, and the total risk if
   every GO were taken. Ask plainly whether to place any of them.

8. **Only on an explicit yes, for a specific trade:** `execute-and-log`.

## Never

- Skip step 7. A GO verdict is permission to present, not to place.
- Supply `confirm=true` on the owner's behalf, or treat an earlier general
  "go ahead" as approval for a specific order.
- Soften a NO GO into a smaller position. The checklist is a gate, not a scale.
- Present stale numbers as current. If the balance or the book is over 15
  minutes old, say so with the timestamp.

## When nothing clears

"Nothing reaches GO tonight" is a complete and successful run. The runbook has
a section for exactly this. Do not go looking for a marginal setup to justify
the session.
