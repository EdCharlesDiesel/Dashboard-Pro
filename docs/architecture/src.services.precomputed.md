---
layer: services
lines: 223
generated: true
---

# `src/services/precomputed.py`

Precomputed score board — the worker's output the UI reads instead of

Source: [[Src/services/precomputed.py]] · [open on disk](../src/services/precomputed.py)

## Imports (2)

- [[src.core.bias]] — Canonical directional bias — the single definition of BULLISH / BEARIS
- [[src.db.market_cache]] — Streamlit read-through cache for market data, backed by Postgres

## Imported by (3)

- [[src.pages_lib.market_overview_lib]]
- [[src.services.background_scanner]]
- [[src.services.bias_service]]

## Public surface

`serialize_house_view`, `deserialize_house_view`, `build_board`, `board_age_seconds`, `board_is_fresh`, `house_view_from_board`, `setup_from_board`, `board_pairs`, `store_board`, `read_board`

Back to [[Architecture]].
