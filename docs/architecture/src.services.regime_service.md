---
layer: services
lines: 429
generated: true
---

# `src/services/regime_service.py`

Streamlit-facing wrapper over the statistical jump model

Source: [[Src/services/regime_service.py]] · [open on disk](../src/services/regime_service.py)

## Imports (5)

- [[src.core.jump_model]] — jump_model.py
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.alert_service]] — Shared email-alert plumbing for the scanner pages
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.tool_log]] — Shared usage logging for the interactive tool pages (R:R Calculator,

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`regime_inputs`, `RegimeRead`, `read_from_fit`, `annualised_sharpe`, `regime_strategy_returns`, `PenaltyTuning`, `tune_jump_penalty`, `get_regime`, `get_regime_history`, `log_regime`

Back to [[Architecture]].
