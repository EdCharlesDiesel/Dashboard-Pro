---
layer: services
lines: 98
generated: true
---

# `src/services/bias_service.py`

Streamlit-facing wrapper over the canonical bias engine

Source: [[Src/services/bias_service.py]] · [open on disk](../src/services/bias_service.py)

## Imports (5)

- [[src.core.bias]] — Canonical directional bias — the single definition of BULLISH / BEARIS
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.precomputed]] — Precomputed score board — the worker's output the UI reads instead of
- [[src.ui.components]] — Object-oriented Bloomberg-terminal UI primitives

## Imported by (6)

- [[src.pages_lib.daily_trading.checklist]]
- [[src.pages_lib.daily_trading.trend]]
- [[src.pages_lib.dxy_gold]]
- [[src.pages_lib.instrument_predictor]]
- [[src.pages_lib.market_overview_lib]]
- [[src.pages_lib.setup_ranker]]

## Public surface

`get_house_view`, `show_house_view`

Back to [[Architecture]].
