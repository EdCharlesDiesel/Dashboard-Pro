---
layer: services
lines: 124
generated: true
---

# `src/services/market_data.py`

Canonical market-data spine — the one OHLC feed every page reads from

Source: [[Src/services/market_data.py]] · [open on disk](../src/services/market_data.py)

## Imports (1)

- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres

## Imported by (8)

- [[src.pages_lib.biased_pivots_page]]
- [[src.pages_lib.fibo_ribbon_page]]
- [[src.pages_lib.setup_ranker]]
- [[src.pages_lib.todays_trades_page]]
- [[src.services.background_scanner]]
- [[src.services.bias_service]]
- [[src.services.position_risk]]
- [[src.services.regime_service]]

## Public surface

`daily_ohlc`, `weekly_ohlc`, `h4_ohlc`, `hourly_ohlc`, `data_asof`, `asof_caption`

Back to [[Architecture]].
