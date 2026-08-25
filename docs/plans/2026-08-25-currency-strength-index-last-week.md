# Add "Last Week" to the Currency Strength Index page's lookback options

Version on creation of this plan: **1.10.48** (VERSION currently reads 1.10.47).

## Context

User request: add "Last Week" as a lookback option alongside 3/6/12 months on
`pages/currency_strength_index.py`'s "Index Settings" sidebar.

## Root cause of a real gap found while investigating

`_PERIOD_OPTIONS` (label -> yfinance period string) currently drives **only**
the Gold/DXY fetch (`_fetch_gold_dxy_closes(period)`). The currency index
chart and the correlation heatmap both consume `index_df`, which is built
from `currency_strength._fetch_pair_closes()` — a fetcher with **no period
parameter at all**, hardcoded to a full year. So today, changing "Lookback"
only ever re-windows the Gold/DXY panel; the index chart and heatmap silently
keep showing a full year regardless of the selection, even though the panel
tag label displays whatever period is chosen. Adding "Last Week" makes this
immediately obvious and wrong (a 5-day selection still showing 252 days of
index history), so this task fixes the underlying gap rather than adding a
fourth option on top of a selector that doesn't fully work.

## Fix

`_PERIOD_OPTIONS` becomes label -> `(yf_period, trading_days_to_window)`:

```python
_PERIOD_OPTIONS = {
    "Last Week": ("5d", 5),
    "3 Months":  ("3mo", 63),
    "6 Months":  ("6mo", 126),
    "1 Year":    ("1y", 252),
}
```

`"5d"` is the yfinance-native period token closest to "last week" (yfinance
has no `"1wk"` period; `5d` = 5 trading days, matching how the currency-index
fetcher already deals in trading days, not calendar days).

`body()` slices `index_df` to `.tail(trading_days_to_window)` **before**
passing it to both `_index_chart` and `_heatmap`, so every panel — index
chart, Gold/DXY panel (already correct), and heatmap — honors the same
selected window consistently. `_fetch_pair_closes()` itself is untouched (no
new period parameter there — it stays a single cached 1y fetch; the new page
just windows its own view of that data), so no change to the existing
Currency Strength page.

Graceful degradation already exists for sparse data (5 trading days is less
than the rolling-correlation window's minimum of 10): `correlation_summary`'s
`w{window}` comes back `None` and the existing `"—"` / `if not rc.empty`
guards already handle it — no new code needed there, just confirmed by test.

## Global constraints

- Never commit. Local changes only — do not stage/commit/push.
- Version bump via `python deploy/sync_version.py <next>` on completion.
- `.foglamp` scan rerun after the change (standing rule from this session).
- Docker rebuild + recreate all four app-tier services after verification.

## Starting state (measured)

- `VERSION` -> `1.10.47`.
- `_PERIOD_OPTIONS` in `src/pages_lib/currency_strength_index.py` has 3
  entries (3/6/12 months), values are bare yfinance period strings, and only
  `_fetch_gold_dxy_closes` consumes them — `index_df`/heatmap ignore period
  entirely.

---

## Task 1 — "Last Week" option + make Lookback actually window every panel

This task takes **1.10.49**.

**Steps**
- [x] Changed `_PERIOD_OPTIONS` to the label -> (yf_period, window_days)
  shape, added "Last Week" -> `("5d", 5)`.
- [x] `body()`: unpacks `(period, window_days)`, `index_df =
  currency_index_series(pair_closes).tail(window_days)` before it reaches the
  index chart, heatmap, or the `selected`/`Currencies Loaded` metric.
- [x] Verified via `AppTest` across all 4 options — 0 exceptions for any.
  "Last Week" correctly renders 3 charts instead of 4 (rolling-correlation
  chart gracefully omitted: 5 days < the 20D default window), full-period
  Gold/DXY correlation still shows a real number (−0.80 on 3 observations),
  rolling shows "—", panel tag correctly reads "Last Week".
- [x] `python .foglamp/introspect.py && python .foglamp/render.py`.
- [x] `python deploy/sync_version.py 1.10.49`.
- [ ] Rebuild + recreate all four app-tier Docker services; confirm inside
  the container via the same AppTest check.
