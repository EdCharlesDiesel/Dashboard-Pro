---
layer: services
lines: 191
generated: true
---

# `src/services/open_positions.py`

Durable store for the positions you are actually holding

Source: [[Src/services/open_positions.py]] · [open on disk](../src/services/open_positions.py)

## Imports (1)

- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres

## Imported by (5)

- [[src.pages_lib.setup_ranker]]
- [[src.pages_lib.todays_trades_page]]
- [[src.services.mt4_import]]
- [[src.services.mt5_link]]
- [[src.services.position_risk]]

## Public surface

`make_row`, `save`, `account_snapshot`, `load`, `saved_at`, `clear`, `unstopped`

Back to [[Architecture]].
