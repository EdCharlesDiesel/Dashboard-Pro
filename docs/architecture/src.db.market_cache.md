---
layer: db
lines: 519
generated: true
---

# `src/db/market_cache.py`

Streamlit read-through cache for market data, backed by Postgres

Source: [[Src/db/market_cache.py]] · [open on disk](../src/db/market_cache.py)

## Imports (3)

- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_data_repository]] — Market-data persistence via PostgreSQL
- [[src.db.trade_repository]] — Trade persistence via PostgreSQL

## Imported by (23)

- [[src.core.data_provider]]
- [[src.indicators.trend_signal]]
- [[src.pages_lib.correlations]]
- [[src.pages_lib.currency_strength]]
- [[src.pages_lib.dxy_gold]]
- [[src.pages_lib.fib_entry]]
- [[src.pages_lib.instrument_predictor]]
- [[src.pages_lib.todays_trades_page]]
- [[src.services.account_state]]
- [[src.services.alert_service]]
- [[src.services.atr_service]]
- [[src.services.backtest_service]]
- [[src.services.market_data]]
- [[src.services.mt4_watch]]
- [[src.services.mt5_trade_import]]
- [[src.services.open_positions]]
- [[src.services.position_risk]]
- [[src.services.precomputed]]
- [[src.services.score_history]]
- [[src.services.signal_store]]
- [[src.services.signal_sweep]]
- [[src.services.swing_playbook_service]]
- [[src.services.tool_log]]

## Public surface

`pooled_market_repository`, `cached_ohlc`, `cached_blob`, `cached_closes`, `clear_market_caches`

Back to [[Architecture]].
