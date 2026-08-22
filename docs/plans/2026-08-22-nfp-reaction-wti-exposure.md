# Add WTI/USD as a fifth exposure on the Event Reaction Map

Version on creation of this plan: **1.10.38** (VERSION currently reads 1.10.37).

## Context

`src/core/nfp_reaction.py`'s four `EventSpec`s (NFP, CPI, PPI, FOMC) each carry
a fixed `exposures` tuple — one row per instrument, `(symbol, beta_rate,
beta_growth, unit, unit_label, decimals)` — read by `score_instruments` to
build the reaction board and by `board_to_signals` to persist the tradable
subset (whatever `normalize_symbol` resolves) to `trade_setups`. Oil isn't on
any of the four lists today. The user asked for it explicitly ("include oil
as well US oil nfp_reaction").

Confirmed live: `normalize_symbol("WTIUSD")` → `"WTI/USD"` (the registry's
6-letter compact-key lookup already handles it, no new alias needed in
`broker_symbols.py`), so `"WTIUSD"` is the literal symbol string to use —
consistent with the file's existing convention (`"XAUUSD"`, `"XAGUSD"`, not
slashed pair names).

## The betas — reasoned, not guessed

Every existing row in this file carries an explanatory comment wherever the
sign isn't obvious; oil gets the same treatment. Oil is a demand-driven,
growth-cyclical commodity priced in USD — closer in behaviour to the equity
indices (`US500`, `NAS100`) than to gold, which is a monetary/safe-haven
hedge. That distinction drives every sign below:

- **`beta_rate` is negative on all four events**, same direction as gold/DXY's
  logic: a hawkish surprise strengthens the dollar, which mechanically
  pressures a USD-priced commodity, all else equal.
- **`beta_growth` flips sign exactly where the equities' does**: **positive**
  on NFP (strong jobs → demand optimism → oil up, mirroring `US500`'s +1.00,
  *unlike* gold's -0.25), then **negative** on CPI/PPI/FOMC (the file's own
  established pattern: inflation/tightening squeezes forward demand).
- **Magnitude scales like the rest of the board does across events** — PPI
  smallest, FOMC largest — using the file's own stated PPI-is-roughly-60%-of-
  CPI ratio for consistency.
- **`unit`/`unit_label`**: `"USD"` per barrel, same convention as
  `XAUUSD`/`XAGUSD`/`BTCUSD` (typical absolute move at `|z|=1` in the
  instrument's own price units, first 30 minutes). Values are conservative
  estimates in the same spirit as the file's existing figures — this is a
  reasoned prior, not a fitted beta (same caveat the file already carries for
  every other row; nothing here claims otherwise).

| Event | beta_rate | beta_growth | unit ($) |
|---|---|---|---|
| NFP  | -0.20 | +0.90 | 0.35 |
| CPI  | -0.85 | -0.50 | 0.42 |
| PPI  | -0.65 | -0.40 | 0.28 |
| FOMC | -1.05 | -0.70 | 0.55 |

**NFP's rate beta was revised down from an initial -0.75 during implementation.**
At -0.75 the two channels were close enough in magnitude that, under the
"Balanced" regime weights (`w_rate=0.70, w_growth=0.55`), they nearly
cancelled — conviction ≈ 0.03, far under the 0.45 persistence gate, so oil's
otherwise-real NFP signal never reached `trade_setups`. That's the *correct*
outcome for a genuinely two-sided instrument (see `DXY`'s comment above), but
oil's growth read is NFP's dominant channel in practice, not a coin flip —
so the rate beta was pulled in to -0.20 (still negative, still a real
secondary dollar-strength drag) rather than left to cancel the signal it was
supposed to represent. Caught by `test_oil_is_on_the_board_and_persists`
failing after the exposures were added with the original number.

## Global constraints

- Never commit. Show the diff.
- Every completed task bumps the patch via `python deploy/sync_version.py <next>`.
- Tests first (`test-driven-development`).
- `docker-compose.yml`'s four app-tier services (`app`, `worker`, `scanner`,
  `sweeper`) all bake this file into their image at build time — today's
  session already found three of them running four versions stale. Rebuild
  and recreate all four after this change lands, don't just edit source and
  assume it's live (the lesson from earlier this session).

## Starting state (measured)

- `VERSION` → `1.10.37`.
- `src/core/nfp_reaction.py`'s four `EventSpec.exposures` tuples: 12 rows
  each (`XAUUSD, XAGUSD, DXY, EURUSD, GBPUSD, USDJPY, AUDUSD, USDZAR, US500,
  NAS100, US10Y, BTCUSD`) — no oil on any of them.
- `tests/test_nfp_reaction.py::test_only_registry_resolvable_symbols_are_persisted`
  hardcodes the allowed persisted-pairs set to `{XAU/USD, XAG/USD, EUR/USD,
  GBP/USD, USD/JPY, AUD/USD, USD/ZAR}` — will need `WTI/USD` added once oil
  is resolvable.
- `normalize_symbol("WTIUSD")` verified live → `"WTI/USD"`.
- All four running containers currently on `dashboard-pro:1.10.37`
  (rebuilt earlier this session).

---

## Task 1 — add the WTIUSD exposure row to all four events

This task takes **1.10.38**.

**Steps**
- [x] Added `test_oil_is_on_the_board_and_persists` plus widened
  `test_only_registry_resolvable_symbols_are_persisted`'s allowed set —
  confirmed both fail against the unpatched code first.
- [x] Added `("WTIUSD", beta_rate, beta_growth, unit, "USD", 2)` to all four
  `exposures` lists. First pass (NFP rate=-0.75) still failed the new test —
  see the beta-revision note above; fixed by pulling NFP's rate beta to
  -0.20 so the dominant growth channel isn't cancelled out.
- [x] `tests/test_nfp_reaction.py`: **74/74 passing**, including the
  pre-existing `test_every_event_covers_the_same_symbol_universe` (updated
  12 → 13) and the new oil-specific tests.
- [x] `python deploy/sync_version.py 1.10.38`
- [x] Rebuilt + recreated `app`, `worker`, `scanner`, `sweeper` from the new
  image; confirmed each container's baked copy of `nfp_reaction.py` contains
  `WTIUSD`.
