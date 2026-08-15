---
layer: pages_lib
lines: 309
generated: true
---

# `src/pages_lib/instrument_predictor.py`

Instrument Predictor — one composite directional read per instrument

Source: [[Src/pages_lib/instrument_predictor.py]] · [open on disk](../src/pages_lib/instrument_predictor.py)

## Imports (12)

- [[src.core.quant_models]] — quant_models.py
- [[src.core.signals]] — —
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.indicators.trend_signal]] — Trend-following signal evaluator — preserves logic from daily-trading-
- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.pages_lib.base]] — Base class for every Bloomberg-styled Streamlit page
- [[src.pages_lib.currency_strength]] — Currency Strength Meter — rank the 9 base/quote currencies strong to w
- [[src.services.bias_service]] — Streamlit-facing wrapper over the canonical bias engine
- [[src.services.cot_fetcher]] — cot_fetcher.py
- [[src.services.prediction_service]] — Instrument Predictor's composite-signal aggregator — pure logic, no I/
- [[src.services.signal_store]] — Shared trade-signal persistence — save any page's signals to ``trade_s
- [[src.ui.components]] — Object-oriented Bloomberg-terminal UI primitives

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`gather_prediction`, `InstrumentPredictorPage`

Back to [[Architecture]].
