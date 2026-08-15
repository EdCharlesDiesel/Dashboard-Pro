---
layer: services
lines: 722
generated: true
---

# `src/services/mt5_mcp.py`

MCP server exposing the live MetaTrader 5 terminal to an AI client

Source: [[Src/services/mt5_mcp.py]] · [open on disk](../src/services/mt5_mcp.py)

## Imports (3)

- [[src.instruments.registry]] — Instrument registry — single source of truth
- [[src.services.broker_symbols]] — Broker symbol → registry instrument mapping
- [[src.services.mt5_link]] — Live MetaTrader 5 terminal link — read the real book, not a saved file

## Imported by (0)

_Nothing imports this. Either an entry point or dead code._

## Public surface

`iso`, `stamp`, `round_volume`, `pick_filling`, `instrument_for`, `MT5Error`, `mt5_status`, `account_info`, `list_symbols`, `symbol_info`, `get_quote`, `get_candles`, `get_candles_range`, `get_positions`, `get_pending_orders`, `get_history`, `calc_margin`, `open_position`, `close_position`, `modify_position`, `place_pending_order`, `cancel_pending_order`

Back to [[Architecture]].
