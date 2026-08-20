# 4 — Execution Handoff

How a scored setup becomes a live position. Five stages; the pipeline stops at
any one that fails.

## 1. Read the score

Grade **and** quality gate, from `1-scoring-criteria.md`. A failed quality gate
ends it here — write it up, do not trade it.

## 2. Size it

**`RiskService.compute()` in `src/services/risk_service.py` is the source of
this arithmetic.** Call it. Never re-derive lot size in prose.

```python
RiskService.compute(account_balance, risk_pct, pip_value, sl_pips,
                    tp1_pips, tp2_pips)
# -> RiskBreakdown(risk_amount, lot_size, rr_tp1, rr_tp2, actual_risk)
```

Two things that bite:

- **`actual_risk`, not `risk_amount`, is what you are risking.** Lot size rounds
  to 2dp, and on a wide-stop metals trade the rounding moves real money.
- **Pip value comes from `src/instruments/registry.py`.** Gold is 100 oz per lot;
  a JPY cross is not a EUR cross. Hardcoding one number for all of them is how a
  gold position ends up sized like an FX major.

Check the balance's age before sizing. A stale balance sizes every trade wrong
and says nothing while doing it — a stale $5,182 against a real $1,989 once
produced sizes 2.6x too large.

## 3. Check what is already open

`src/services/exposure.py` — `net_currency_exposure()` and `check_stack()`.
A second position sharing a currency leg is not a second trade; it is one
bigger trade with a hidden correlation. The Setup Ranker's held-book panel
shows the same netting.

Also read the margin level. `mt5_link.margin_warning()` speaks up below 200%.

## 4. Pass the four gates

See `2-mt5-tooling.md`. All four, then `order_check`. **`confirm=true` comes
from the owner in words, for that order.**

## 5. Record the decision

| Store | What it holds |
|---|---|
| `open_positions.json` | The live book, written by the MT5 sync |
| `setup_ranker_score_history.json` | The score **at the time of the decision** |
| `logs/mt5_tool_audit.jsonl` | Every tool call, written by the PostToolUse hook |

The score history is what makes the scorer accountable later: without the score
as it stood when the trade was taken, a post-mortem can only argue about what
the setup looked like in hindsight.
