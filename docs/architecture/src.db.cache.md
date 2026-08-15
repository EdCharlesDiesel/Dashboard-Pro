---
layer: db
lines: 168
generated: true
---

# `src/db/cache.py`

Connection pooling for the Postgres trade repository

Source: [[Src/db/cache.py]] · [open on disk](../src/db/cache.py)

## Imports (1)

- [[src.db.trade_repository]] — Trade persistence via PostgreSQL

## Imported by (11)

- [[src.db.connection]]
- [[src.db.market_cache]]
- [[src.pages_lib.daily_trading.checklist]]
- [[src.pages_lib.daily_trading.sidebar]]
- [[src.pages_lib.todays_trades_page]]
- [[src.services.mt5_trade_import]]
- [[src.services.position_risk]]
- [[src.services.signal_store]]
- [[src.services.signal_sweep]]
- [[src.services.swing_playbook_service]]
- [[src.services.tool_log]]

## Public surface

`pooled_repository`, `cached_load_setups`, `cached_load_open`, `cached_daily_losses`, `cached_performance_stats`, `cached_realized_pnl`, `cached_get_state`, `set_state`, `clear_read_caches`

Back to [[Architecture]].
