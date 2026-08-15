---
layer: pages_lib
lines: 270
generated: true
---

# `src/pages_lib/currency_strength.py`

Currency Strength Meter — rank the 9 base/quote currencies strong to weak

Source: [[Src/pages_lib/currency_strength.py]] · [open on disk](../src/pages_lib/currency_strength.py)

## Imports (6)

- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.pages_lib.base]] — Base class for every Bloomberg-styled Streamlit page
- [[src.services.signal_store]] — Shared trade-signal persistence — save any page's signals to ``trade_s
- [[src.ui.components]] — Object-oriented Bloomberg-terminal UI primitives
- [[src.ui.theme]] — Bloomberg-terminal palette + global CSS injection

## Imported by (2)

- [[src.pages_lib.instrument_predictor]]
- [[src.pages_lib.setup_ranker]]

## Public surface

`CurrencyStrengthPage`

Back to [[Architecture]].
