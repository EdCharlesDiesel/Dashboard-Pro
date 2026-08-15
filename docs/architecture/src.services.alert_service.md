---
layer: services
lines: 301
generated: true
---

# `src/services/alert_service.py`

Shared email-alert plumbing for the scanner pages

Source: [[Src/services/alert_service.py]] · [open on disk](../src/services/alert_service.py)

## Imports (2)

- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.services.tool_log]] — Shared usage logging for the interactive tool pages (R:R Calculator,

## Imported by (5)

- [[src.pages_lib.commodity_cot_lib]]
- [[src.pages_lib.correlations]]
- [[src.pages_lib.market_overview_lib]]
- [[src.services.regime_service]]
- [[src.services.signal_store]]

## Public surface

`email_configured`, `email_recipient`, `send_email`, `NotifyCache`

Back to [[Architecture]].
