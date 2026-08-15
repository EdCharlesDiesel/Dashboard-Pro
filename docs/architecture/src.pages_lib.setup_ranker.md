---
layer: pages_lib
lines: 1599
generated: true
---

# `src/pages_lib/setup_ranker.py`

Setup Ranker page — Bloomberg-terminal version

Source: [[Src/pages_lib/setup_ranker.py]] · [open on disk](../src/pages_lib/setup_ranker.py)

## Imports (14)

- [[src.core.bias]] — Canonical directional bias — the single definition of BULLISH / BEARIS
- [[src.core.secrets]] — Single source of truth for all sensitive configuration
- [[src.core.signals]] — —
- [[src.pages_lib.base]] — Base class for every Bloomberg-styled Streamlit page
- [[src.pages_lib.currency_strength]] — Currency Strength Meter — rank the 9 base/quote currencies strong to w
- [[src.services.bias_service]] — Streamlit-facing wrapper over the canonical bias engine
- [[src.services.exposure]] — Currency-leg exposure guard — catch the stack `CORR_GROUPS` can't see
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.mt5_link]] — Live MetaTrader 5 terminal link — read the real book, not a saved file
- [[src.services.open_positions]] — Durable store for the positions you are actually holding
- [[src.services.parallel_fetch]] — Thread-pool fan-out for I/O-bound per-item fetches (yfinance/Postgres)
- [[src.services.signal_store]] — Shared trade-signal persistence — save any page's signals to ``trade_s
- [[src.ui.components]] — Object-oriented Bloomberg-terminal UI primitives
- [[src.ui.theme]] — Bloomberg-terminal palette + global CSS injection

## Imported by (2)

- [[src.pages_lib.fib_entry]]
- [[src.services.background_scanner]]

## Public surface

`alert_price_bucket`, `fmt_price`, `trade_levels`, `money_breakdown`, `risk_deviates`, `trade_rationale_template`, `polished_rationale`, `SetupRankerPage`

Back to [[Architecture]].
