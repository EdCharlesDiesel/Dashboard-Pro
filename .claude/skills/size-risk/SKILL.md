---
name: size-risk
description: Use when a setup has passed bias confirmation and needs a position size - computes lot size from the live balance, the ATR stop and the risk budget.
---

# Size Risk

Runbook step 7. Sizing is arithmetic, not judgement, and the arithmetic already
exists: `RiskService.compute()` in `src/services/risk_service.py`. Call it.
Never re-derive lot size in prose — a second implementation of this maths is
how a position ends up 2.6x too large.

## Inputs

- Live balance from `account_state.get_balance()` — **check its age first**
- Stop distance in pips, from the setup's ATR stop
- Risk % per trade (default 1.00%; never above 2%)

## Steps

1. Read the balance **and its `updated_at`**. Older than 15 minutes: stop and
   say so. See `logs/mt5_sync.log`. A stale balance sizes every trade wrong and
   is silent about it.
2. Get pip value from `src/instruments/registry.py`. Never hardcode one — gold
   is 100 oz per lot and JPY crosses are not EUR crosses.
3. Call `RiskService.compute(account_balance, risk_pct, pip_value, sl_pips,
   tp1_pips, tp2_pips)`.
4. Report **`actual_risk`**, not just the target. Lot rounding moves real money
   on wide-stop metals trades.
5. Check correlated exposure with `src/services/exposure.py`. A second position
   sharing a currency leg is one bigger trade, not two.

## Sanity check before reporting

Compare the risk to the balance. If the stop distance and the minimum legal lot
already exceed the risk budget, **the account is too small for that stop** —
say exactly that. The honest answer is sometimes that the trade does not fit.

## Stop

Report lot size, `actual_risk` in account currency, R:R, and the correlated
exposure this would add. Place nothing.
