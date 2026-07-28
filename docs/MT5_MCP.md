# The MT5 MCP server

`src/services/mt5_mcp.py` exposes the live MetaTrader 5 terminal to an AI client
(Claude Code) as a set of tools. It is the trading-capable counterpart to
`src/services/mt5_link.py`, which stays read-only for the Streamlit app.

## Why two modules

| | `mt5_link.py` | `mt5_mcp.py` |
| --- | --- | --- |
| Consumer | The Streamlit app (exposure guard, Trade Journal sync) | An AI client, over MCP |
| Can place orders | **No — by design** | Yes, behind four gates |
| Connection | `initialize()`/`shutdown()` per read | Held open for the process lifetime |
| Imported by pages | Yes | **Never** |

The connection styles are incompatible in one process — either module's
`shutdown()` would kill the other's session — so the MCP server runs as its own
process. That is also the honest boundary for the capability: a page has no
legitimate reason to be able to send an order, so it keeps no route to one.

Both resolve broker symbols through `broker_symbols.normalize_symbol`, so every
tool result carries the app's registry `pair` name (`EURUSDm` → `EUR/USD`)
alongside the raw broker symbol.

## Running it

Registered with Claude Code at user scope, so it is available in every project:

```bash
claude mcp get mt5
```

It launches itself; you never start it by hand. The module inserts the repo root
into `sys.path`, so it works from any working directory. To run it manually for
debugging:

```bash
C:/x/Dashboard-Pro/.venv/Scripts/python.exe C:/x/Dashboard-Pro/src/services/mt5_mcp.py
```

The MT5 terminal must be **running and logged in** on the same machine. There is
no API key — the server attaches to whichever account the terminal holds.

## Tools

**Read** — `mt5_status` (start here when debugging), `account_info` (includes the
shared `margin_warning`), `list_symbols`, `symbol_info`, `get_quote`,
`get_candles`, `get_candles_range`, `get_positions`, `get_pending_orders`,
`get_history`, `calc_margin`.

**Trade** — `open_position`, `close_position`, `modify_position`,
`place_pending_order`, `cancel_pending_order`.

Candles come from the **broker's** feed, not yfinance, so they can differ
slightly from the dashboard's charts. That is correct: this is the execution
venue's own view of price.

## The four gates

An order reaches the broker only if all four hold:

1. `MT5_ALLOW_TRADING=1` in the server config.
2. `confirm=true` passed explicitly on the tool call. Without it the tool raises
   and nothing is sent.
3. Volume within `MT5_MAX_VOLUME`.
4. Algo Trading enabled in the terminal (Tools > Options > Expert Advisors >
   "Allow algorithmic trading", plus the toolbar button). MT5 blocks
   programmatic orders otherwise.

Every request is then dry-run through `order_check`; a request the broker would
reject raises instead of being sent.

## Configuration

Set via `claude mcp add -e KEY=value`. Current registration: trading enabled,
0.5-lot cap.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MT5_ALLOW_TRADING` | off | Master switch for all trade tools. |
| `MT5_MAX_VOLUME` | `1.0` | Per-order lot cap. |
| `MT5_ALLOWED_SYMBOLS` | any | Comma-separated whitelist, e.g. `EURUSD,XAUUSD`. |
| `MT5_MAGIC` | `770001` | Magic number stamped on this server's orders. |
| `MT5_TERMINAL_PATH` | auto | Path to `terminal64.exe` if several terminals are installed. |
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | — | Only to log the terminal in programmatically. |

To change a setting, re-register:

```bash
claude mcp remove mt5 -s user
```

```bash
claude mcp add mt5 -s user -e MT5_ALLOW_TRADING=1 -e MT5_MAX_VOLUME=0.1 -- "C:/x/Dashboard-Pro/.venv/Scripts/python.exe" "C:/x/Dashboard-Pro/src/services/mt5_mcp.py"
```

Dropping `-e MT5_ALLOW_TRADING=1` makes it read-only.

## Tests

`tests/test_mt5_mcp.py` covers the pure helpers (volume stepping, filling-mode
choice, symbol mapping, timestamps) and every guardrail, plus a guard asserting
`mt5_link` never grows an order function. It skips wholesale off Windows.

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_mt5_mcp.py --no-cov
```

The module itself is in the coverage `omit` list — it is live-terminal I/O with a
Windows-only dependency, same category as `evening_sentry` / `news_fetcher`.
