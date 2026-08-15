---
layer: pages_lib
lines: 362
generated: true
---

# `src/pages_lib/todays_trades_page.py`

Today's Trades — the morning answer, on one screen

Source: [[Src/pages_lib/todays_trades_page.py]] · [open on disk](../src/pages_lib/todays_trades_page.py)

## Imports (10)

- [[src.core.todays_trades]] — Today's tradeable ideas — consensus across every signal source, sized
- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.indicators.technical]] — Static technical-indicator helpers
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.pages_lib.base]] — Base class for every Bloomberg-styled Streamlit page
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.open_positions]] — Durable store for the positions you are actually holding
- [[src.services.position_risk]] — What the open book is actually risking, priced by the stochastic engin
- [[src.ui.theme]] — Bloomberg-terminal palette + global CSS injection

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`TodaysTradesPage`

Back to [[Architecture]].
