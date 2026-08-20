# 2 — MT5 Tooling

**Source of truth: `docs/MT5_MCP.md`.** Read it for the tool list, the module
split and the configuration table. Nothing here repeats it.

What this file adds is the **pre-flight checklist** — what to confirm before
proposing an order, and the read/write split that decides which module to reach
for.

## Read or write — pick the right module

| Need | Module | Note |
|---|---|---|
| Balance, positions, quotes, candles, history | `src/services/mt5_link.py` | **Exposes no order call, by design.** Cannot place anything |
| Place, modify, close, pending orders | `src/services/mt5_trade.py` | The only writer. Behind all four gates |

An agent that wants to *know* something uses the read path, always. If a read
seems to need the trade module, the read is being done wrong.

## The four gates — confirm before proposing an order

```
1. MT5_ALLOW_TRADING=1 on the server      -> else read-only. Say so; do not retry
2. confirm=true passed explicitly         -> never defaulted, never inferred from "yes"
3. volume <= MT5_MAX_VOLUME               -> currently 0.5
4. Algo Trading enabled in the terminal   -> else MT5 blocks it silently
```

Then `order_check` dry-runs the request; one the broker would reject raises
instead of being sent.

**`confirm=true` comes from the owner, in words, for that specific order.** It
is not something an agent sets on the owner's behalf, and a general "go ahead"
earlier in a session is not it. `tests/test_mt5_mcp.py:93`
(`test_unconfirmed_call_is_refused`) exists to keep that true.

## When the terminal is unreachable

`IPC timeout (-10005)` means the terminal is hung, not closed. Reconnecting
does not fix it — `mt5_link` already re-initialises every cycle. Check
`logs/mt5_watchdog.log`; the watchdog bounces the terminal after 5 consecutive
failures. Do not loop retries in the meantime.
