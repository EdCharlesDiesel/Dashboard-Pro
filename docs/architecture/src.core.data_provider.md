---
layer: core
lines: 272
generated: true
---

# `src/core/data_provider.py`

_No module docstring._

Source: [[Src/core/data_provider.py]] · [open on disk](../src/core/data_provider.py)

## Imports (3)

- [[src.core.config]] — —
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.services.fred_data]] — Canonical macro spine — one definition of a FRED series, for every pag

## Imported by (2)

- [[src.core.reporting]]
- [[src.pages_lib.market_overview_lib]]

## Public surface

`MarketDataProvider`, `QuantConnectProvider`, `FallbackProvider`, `get_provider`, `fetch_data`, `get_macro_data`, `fetch_fred_series`

Back to [[Architecture]].
