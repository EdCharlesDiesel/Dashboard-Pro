---
layer: services
lines: 431
generated: true
---

# `src/services/mt5_link.py`

Live MetaTrader 5 terminal link — read the real book, not a saved file

Source: [[Src/services/mt5_link.py]] · [open on disk](../src/services/mt5_link.py)

## Imports (3)

- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.broker_symbols]] — Broker symbol → registry instrument mapping
- [[src.services.open_positions]] — Durable store for the positions you are actually holding

## Imported by (2)

- [[src.pages_lib.setup_ranker]]
- [[src.services.mt5_mcp]]

## Public surface

`AccountSnapshot`, `available`, `probe`, `positions_to_rows`, `read_terminal`, `sync`, `margin_warning`, `detect_server_utc_offset`, `closed_deals`

Back to [[Architecture]].
