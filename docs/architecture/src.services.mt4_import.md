---
layer: services
lines: 330
generated: true
---

# `src/services/mt4_import.py`

Parse an MT4 'Save as Report' / 'Detailed Report' HTML statement into rows

Source: [[Src/services/mt4_import.py]] · [open on disk](../src/services/mt4_import.py)

## Imports (3)

- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.broker_symbols]] — Broker symbol → registry instrument mapping
- [[src.services.open_positions]] — Durable store for the positions you are actually holding

## Imported by (1)

- [[src.services.mt5_trade_import]]

## Public surface

`session_for_hour`, `parse_mt4_balance`, `parse_mt4_html`, `parse_mt4_open_positions`, `to_open_position_rows`, `to_journal_rows`

Back to [[Architecture]].
