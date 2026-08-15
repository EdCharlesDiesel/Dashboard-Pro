---
layer: services
lines: 310
generated: true
---

# `src/services/background_scanner.py`

Unattended ingest + score daemon — the app's data streams in day & night

Source: [[Src/services/background_scanner.py]] · [open on disk](../src/services/background_scanner.py)

## Imports (6)

- [[src.core.bias]] — Canonical directional bias — the single definition of BULLISH / BEARIS
- [[src.core.observability]] — Central observability: structured logging + event recording
- [[src.pages_lib.fib_entry]] — 15M Fibonacci Entry — Bloomberg-terminal version
- [[src.pages_lib.setup_ranker]] — Setup Ranker page — Bloomberg-terminal version
- [[src.services.market_data]] — Canonical market-data spine — the one OHLC feed every page reads from
- [[src.services.precomputed]] — Precomputed score board — the worker's output the UI reads instead of

## Imported by (1)

- [[src.pages_lib.navigation]]

## Public surface

`ensure_started`, `scan_once`, `run_forever`

Back to [[Architecture]].
