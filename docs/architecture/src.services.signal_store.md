---
layer: services
lines: 160
generated: true
---

# `src/services/signal_store.py`

Shared trade-signal persistence — save any page's signals to ``trade_setups``

Source: [[Src/services/signal_store.py]] · [open on disk](../src/services/signal_store.py)

## Imports (6)

- [[src.core.signals]] — —
- [[src.core.volume_profile]] — Fixed-range volume profile — TradingView's FRVP, anchored to session b
- [[src.db.cache]] — Connection pooling for the Postgres trade repository
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres
- [[src.services.alert_service]] — Shared email-alert plumbing for the scanner pages
- [[src.services.session_service]] — Trading-session detection — Kill Zone classifier

## Imported by (9)

- [[src.pages_lib.biased_pivots_page]]
- [[src.pages_lib.currency_strength]]
- [[src.pages_lib.daily_trading.trend]]
- [[src.pages_lib.dxy_gold]]
- [[src.pages_lib.fib_entry]]
- [[src.pages_lib.fibo_ribbon_page]]
- [[src.pages_lib.instrument_predictor]]
- [[src.pages_lib.market_overview_lib]]
- [[src.pages_lib.setup_ranker]]

## Public surface

`default_dedupe_key`, `persist_signals`

Back to [[Architecture]].
