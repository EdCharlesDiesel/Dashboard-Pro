---
layer: services
lines: 348
generated: true
---

# `src/services/signal_sweep.py`

Run every signal page headlessly so its own code persists its signals

Source: [[Src/services/signal_sweep.py]] · [open on disk](../src/services/signal_sweep.py)

## Imports (3)

- [[src.core.observability]] — Central observability: structured logging + event recording
- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`run_page`, `sync_broker_state`, `sweep_once`, `run_forever`, `main`

Back to [[Architecture]].
