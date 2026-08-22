# Trade Journal — realised growth in money, and the open book's outcomes

> **Reconstructed 2026-08-22** from the session transcript. Approved and
> executed on 2026-08-20 but never written to `docs/plans/` at the time — see
> `2026-08-20-plans-in-docs.md`. **This version is already committed** as
> `938d3e6 … V1.10.22`, so until now the shipped code had no plan behind it.

**Goal:** Make the Trade Journal show what the account has actually made in money, and what the open book is worth if every take-profit hits or every stop hits — fed by MT5 automatically instead of by hand.

**Architecture:** Four small changes and one new tab. The heavy lifting already exists: `mt5_trade_import.import_closed_trades()` reads and stores closed deals with net money; `position_risk.usd_pnl()` values a move from entry to any price. The work is wiring, not building. The one genuinely new piece is a pure `book_projection()` in `position_risk`, which is in coverage scope and gets tests first.

**Tech Stack:** Python 3.14, Streamlit, Plotly, Postgres, pytest.

**Spec:** The owner's request, 2026-08-20: realised growth from closed trades in monetary value, include the open positions, and show what happens if all the take-profits get hit.

## Global Constraints

- **Never commit.** `VERSION` reads 1.10.21, so this plan takes **1.10.22**.
- **TDD**: `src/services/position_risk.py` and `src/db/trade_repository.py` are inside `--cov=src` and must carry real tests. `pages/` is omitted from coverage — page code is verified by running it.
- **Never weaken the four MT5 trade gates.** Nothing here places, modifies or closes anything.
- Reuse, do not re-derive: `usd_pnl`, `usd_rates`, `contract_units`, `import_closed_trades`, `positions_from_store` all exist.

---

## Context

The page is disconnected in three specific places, each verified against the live database:

1. **The root cause — a one-word mismatch that hides every MT5 trade.** `src/db/trade_repository.py:48` reads `EXECUTED_SOURCES = ("checklist", "mt4_import", "mt5_sync")`, but `src/services/mt5_trade_import.py:38` writes `SOURCE = "mt5_import"`. **Nothing in the codebase writes `"mt5_sync"`.** The journal's "trades you took" filter silently drops all **31 real MT5 trades carrying $795.01 of realised profit**. The same tuple governs the Martingale page.
2. **The money column is stored but never selected.** `trade_setups.profit` exists and `trips_to_journal_rows` writes it as `profit + commission + swap`; `load_journal_trades` never asks for it.
3. **Nothing on the page imports closed deals.** `import_closed_trades` is called only from a CLI run by hand. Data stops at 2026-08-14.

**Two facts that shape what can honestly be drawn:**

- **Money exists only for the MT5 era.** `mt5_import`: 31 closed rows, 31 with `profit`. `mt4_import`: 42 closed rows, **0 with `profit` and 0 with `risk_amount`**, so `r_multiple × risk_amount` cannot reconstruct money either.
- **The container cannot reach MetaTrader.** The app runs on Linux where the `MetaTrader5` package does not work, so the page must read stored data.

**Why the broker's own P/L solves the currency problem.** `usd_pnl` needs a quote→USD rate, and `todays_trades.usd_rates()` derives rates from the book itself — which fails on a lone EUR/AUD position with no USD leg, leaving the projection ~1.4x out and mislabelled as dollars. Every MT5 position already carries its floating P/L in account currency. Storing that plus the current price recovers the exact factor the broker used:

```
at_target = profit_now * (target - entry) / (price_now - entry)
```

P&L is linear in price, so this is exact, needs no contract sizes and no quote lookup.

---

## Task 1: Unhide the MT5 trades

Modify `src/db/trade_repository.py:48`. Failing test first: `SOURCE in EXECUTED_SOURCES`. Replace `"mt5_sync"` with `"mt5_import"` — replace, not append, since no row uses `mt5_sync`. Add a second guard asserting every name in the tuple is written as a string literal somewhere.

## Task 2: Carry the broker's P/L into the store

Modify `mt5_link.positions_to_rows` and `open_positions._FIELDS` to add `profit` and `price_current`. `_clean` rebuilds rows as `{k: row.get(k) for k in _FIELDS}`, so a field missing from `_FIELDS` is **silently dropped**.

## Task 3: `book_projection()`

New pure function in `position_risk`, returning `{floating, at_target, at_stop, converted, unconverted}`. Precedence per position: broker ratio → `usd_pnl` with rates → name the pair as unconverted. Six failing tests first, using real rows from the live book.

## Task 4: Import closed deals on the 5-minute loop

`deploy/mt5_sync.py::sync_once` calls `import_closed_trades(days=7)` in its own `try/except`. It must never change the return code — a stale journal is an inconvenience, a dead sync loop is a wrong position size.

## Task 5: The page

Add `profit` to the SELECT; add a **💰 Growth** tab beside the existing equity tab (a new tab, because the R curve works and covers full history). Four headline cells, a balance curve with all-TP/all-SL lines, the open book per position, a staleness banner, and a manual pull button. Never print a currency symbol on an unconverted number.

---

## Verification

1. Unit tests across the five affected files.
2. **The hidden trades are visible again:** the executed filter returns 74 rows / $795.01, versus 43 rows / NULL before.
3. The sync imports closed deals and the journal passes Aug 14.
4. The projection is arithmetically right, checked by hand against the terminal.
5. Full suite; coverage ≥ 80%.
6. The page renders — pages are outside coverage, so this is the only proof.
7. Deploy 1.10.22. Show the diff. **Never commit.**

## What actually happened

All verified. Before/after on the live DB: **43 rows, no money → 74 rows, $795.01**; automating the import added 12 more trades, reaching **$1,843.07**. `book_projection` matched contract arithmetic to **$0.00** on all three USD-quoted positions, and floating matched broker equity−balance exactly.

Two defects were caught during verification, both mine. A test stub took no arguments while the caller passed `days=7`; the production `except Exception` swallowed the `TypeError`, so it read as "never ran". And the deployed page first showed `Floating +0.00` with six ZAR pairs flagged unconverted — **the background sync loop had started before the edit and Python does not hot-reload**, so it kept writing old-format rows and overwrote the good data. Restarting the loop fixed it. Had verification stopped at the host, this would have been reported as working over a page showing zeros.
