---
layer: services
lines: 341
generated: true
---

# `src/services/mt5_trade_import.py`

Import closed MT5 deals into ``trade_setups`` as graded, closed rows

Source: [[Src/services/mt5_trade_import.py]] · [open on disk](../src/services/mt5_trade_import.py)

## Imports (5)

- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.broker_symbols]] — Broker symbol → registry instrument mapping
- [[src.services.mt4_import]] — Parse an MT4 'Save as Report' / 'Detailed Report' HTML statement into 

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`parse_close_level`, `pair_deals`, `trips_to_journal_rows`, `broker_utc_offset`, `import_closed_trades`

Back to [[Architecture]].
