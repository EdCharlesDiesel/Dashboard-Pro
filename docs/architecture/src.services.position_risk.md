---
layer: services
lines: 488
generated: true
---

# `src/services/position_risk.py`

What the open book is actually risking, priced by the stochastic engine

Source: [[Src/services/position_risk.py]] · [open on disk](../src/services/position_risk.py)

## Imports (9)

- [[src.core.stochastic]] — Stochastic calculus toolkit for price series
- [[src.core.todays_trades]] — Today's tradeable ideas — consensus across every signal source, sized
- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.indicators.technical]] — Static technical-indicator helpers
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.account_state]] — Persistent store for the live account balance
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.open_positions]] — Durable store for the positions you are actually holding

## Imported by (1)

- [[src.pages_lib.todays_trades_page]]

## Public surface

`Position`, `PositionRisk`, `contract_units`, `usd_pnl`, `barrier_view`, `assess`, `cone`, `positions_from_store`, `verdict`, `position_from_idea`, `screen`, `load_book`, `calibrate`, `load_candidates`, `main`

Back to [[Architecture]].
