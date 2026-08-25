# Currency Strength Index page — per-currency indexes + correlation views

Version on creation of this plan: **1.10.45** (VERSION currently reads 1.10.44).

## Context

User request, following an explicit investigate-first pass (reported back
before any code was written): a new page showing per-currency strength as a
**time series** (not just today's snapshot), plus correlation views, with one
specific headline feature — a synced stacked XAU/USD-over-DXY chart (like a
TradingView comparison, not overlaid on one axis) with the actual rolling
correlation coefficient displayed prominently, since that's the one thing the
user explicitly called "especially" wanted.

## What already exists (from the investigation)

- **`src/pages_lib/currency_strength.py`**`_currency_returns()` already
  computes per-currency strength correctly (average of every registry FX
  pair's % return, sign-flipped for the quote leg) — but only as a **snapshot**
  at "now," not a stored time series. Covers exactly the 9 registry FX
  currencies: USD, EUR, GBP, AUD, NZD, JPY, CHF, CAD, ZAR.
- **`src/services/index_analysis.py`** — pure, already-unit-tested,
  general-purpose two-series correlation math (`rolling_correlation`,
  `correlation_summary`, `daily_returns`). Takes arbitrary Close-price Series;
  has no idea what a currency index is, which is exactly why it's reusable
  as-is for Gold vs DXY or Gold vs a synthetic USD index.
- **`src/pages_lib/correlations.py`** — has the UI patterns to copy
  (`_render_heatmap`, `_corr_color`, a rolling-vs-focus-instrument line chart)
  but no synthetic currency-index series; only real registry price series.
  Also already defines `_ALL_INSTRUMENTS["DXY"] = "DX-Y.NYB"` as the ticker
  this app already uses for DXY everywhere it's needed ad hoc (it isn't a
  registry instrument).

## Architecture

**New pure module — `src/services/currency_index.py`** (no Streamlit, no I/O,
unit-tested): generalizes `_currency_returns`'s snapshot method into a daily
time series.

- `daily_currency_returns(closes: pd.DataFrame) -> pd.DataFrame` — for every
  registry FX pair present in `closes`'s columns, `pct_change()` its Close,
  attribute that day's return to the base currency and the sign-flipped
  return to the quote currency, then average **cross-sectionally per day**
  across whichever pairs have data that day (`skipna` — a single pair's
  holiday gap doesn't null out the whole currency for that day, matching how
  `_currency_returns` already tolerates missing pairs).
- `currency_index_series(closes, base=100.0) -> pd.DataFrame` — cumulative
  product of `(1 + daily_currency_returns)`, starting at `base`. Missing days
  fill to a 0% move (flat, not propagated NaN) so the index never breaks.

**New page — framework-style (`BloombergPage`)**, nav code `CIDX`:
`src/pages_lib/currency_strength_index.py` + `pages/currency_strength_index.py`
shim, registered in `src/pages_lib/navigation.py`'s Research Lab section next
to `DXAU`/`IDXC` (same neighborhood — this page touches both).

Reuses `src.pages_lib.currency_strength._fetch_pair_closes()` directly for the
FX-pair universe (no duplicate fetcher), plus a small new cached fetch for
Gold (`GC=F`) + DXY (`DX-Y.NYB`) closes via `cached_closes`, matching the
ticker `correlations.py` already uses for DXY.

**Body layout**:
1. **Index chart** — one line per currency (default all 9, multiselect),
   `currency_index_series()` output, standard `go.Figure` line chart in the
   terminal theme.
2. **Gold vs DXY — headline panel** (always shown, not behind a selector,
   since this is the one thing explicitly called out):
   - Stacked dual-panel chart via `make_subplots(rows=2, shared_xaxes=True)`
     — XAU/USD on top, DXY directly below, same time window. Not overlaid on
     one axis, so the inverse relationship is visually obvious without a
     dual-y-axis scaling illusion.
   - The current rolling correlation coefficient (`index_analysis.
     rolling_correlation`/`correlation_summary`) displayed as a large,
     dedicated `MetricCell` directly beside/below the chart — not buried in a
     matrix.
3. **Correlation heatmap** — 9 currency indexes + Gold + DXY, using
   `correlations.py`'s heatmap visual pattern (copied/adapted locally, not
   imported — it's page-specific rendering, not shared logic) fed by the new
   index series instead of raw pair returns.

Audit-only, like `indices-correlation.py`/`disconnect_mon`: logs to
`tool_usage_log` via the existing `log_tool_usage`/`NotifyCache` idiom
(deduped per instrument-set + window), **not** wired to `persist_signals` —
this is a descriptive correlation view, not a directional pair+bias call.

## Tech stack

Same as every other page: Streamlit, pandas, Plotly (`go`/`make_subplots`),
the existing `BloombergPage` template method, `src.db.market_cache.
cached_closes` for the network/DB-cached fetch.

## Global constraints

- Never commit. Show the diff.
- Every completed task bumps the patch via `python deploy/sync_version.py <next>`.
- Tests first (`test-driven-development`) for the new pure module.
- `.foglamp/introspect.py` + `render.py` rerun after the change (new page =
  new node; also the standing "update dynamically" rule from this session).
- New page → add to `src/pages_lib/navigation.py`'s `NAV_SECTIONS` (the single
  source of truth for the sidebar) and to `.foglamp/scan.json` as a new node.
- Docker rebuild: this is a source change baked into all four app-tier
  images — rebuild + recreate `app`, `worker`, `scanner`, `sweeper` once the
  page is verified locally, per the standing lesson from earlier this
  session.

## Starting state (measured)

- `VERSION` → `1.10.44`.
- No `src/services/currency_index.py`, no `pages/currency_strength_index.py`,
  no `CIDX` nav entry.
- Registry FX currencies confirmed: USD, EUR, GBP, AUD, NZD, JPY, CHF, CAD,
  ZAR (9). DXY ticker confirmed `DX-Y.NYB`, Gold ticker `GC=F`
  (`INSTRUMENTS["XAU/USD"].ticker`).

---

## Task 1 — pure currency-index module + tests

This task takes **1.10.46**.

**Steps**
- [x] Added `tests/test_currency_index.py` — confirmed
  `ModuleNotFoundError: No module named 'src.services.currency_index'` first.
- [x] Implemented `src/services/currency_index.py`. All 3 tests green,
  including the holiday-gap (NaN) and monotonic-strengthening cases.
- [x] `python deploy/sync_version.py 1.10.46`.

## Task 2 — the page itself

This task takes **1.10.47**.

**Steps**
- [x] Implemented `src/pages_lib/currency_strength_index.py`
  (`CurrencyStrengthIndexPage(BloombergPage)`) + `pages/currency_strength_index.py`
  shim, per the body layout above.
- [x] Registered `CIDX` in `src/pages_lib/navigation.py`'s Research Lab
  section, right after `DXAU`.
- [x] Verified via `AppTest.from_file("pages/currency_strength_index.py")`
  against live data — no exceptions, 4 charts rendered. Confirmed the
  headline numbers actually populate: **Gold vs DXY full-period corr −0.35,
  20D rolling −0.41**, 125 observations, 9 currencies loaded.
- [x] `python .foglamp/introspect.py && python .foglamp/render.py` — also
  hand-edited `scan.json`'s aggregate `"pages"` node label from "57" to "58"
  signal & tool pages (individual pages/small pure-math services like this
  one and its `index_analysis.py` precedent aren't separately graphed —
  confirmed by checking neither `currency_strength.py` nor `index_analysis.py`
  have dedicated nodes either).
- [x] Full suite: 1990 passed, only the 2 pre-existing/unrelated GARCH
  `arch`-package failures (local-venv-only, documented limitation).
- [x] `python deploy/sync_version.py 1.10.47`.
- [ ] Rebuild + recreate all four app-tier Docker services; confirm the new
  page loads inside the container via the same AppTest check.
