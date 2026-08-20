# Wire `pages/nfp_reaction.py` into Dashboard-Pro

Version on creation of this plan: **1.10.21** (VERSION currently reads 1.10.20).

## Context

A new page, `pages/nfp_reaction.py` (679 lines), was added to the working tree
(`git status`: `AM`). It models the NFP release as a standardised surprise
score, pushes that score through a **regime-aware** transmission chain
(jobs → spending → inflation → rates → assets), and ranks instrument exposure
with a conviction score that collapses when the rate channel and the growth
channel disagree. Nothing else in this repo scores a scheduled macro release
this way — `surprise_tab.py` builds a broad Citi-ESI-style index across the
whole calendar; this goes deep on one release. It is worth keeping.

But it was written against a **different tree** and cannot run here as-is:

| Problem | Evidence |
|---|---|
| Never renders | The module only *defines* `render(engine)`. Streamlit runs a page file top-to-bottom, so it registers as a multipage entry that draws a blank page. Peers end with an unconditional `_page()` or an `if __name__ == "__main__":` block. |
| Wrong import root | `from theme import BloombergTheme` — the path here is `src.ui.theme`. The `try` therefore **always** falls through to the hardcoded fallback hexes (`#0B0F14`, `#C8952B`), so the page is off-palette by construction. |
| No page chrome | No `st.set_page_config()`, no `BloombergTheme.apply()`, no `render_sidebar_nav()`. |
| Orphaned | Absent from `NAV_SECTIONS` in `src/pages_lib/navigation.py`. |
| Foreign symbols | Module-level `INSTRUMENTS` shadows the registry name and uses broker symbols (`XAUUSD`, `US500`, `DXY`) that don't join to `src/instruments/registry.py` names (`XAU/USD`). |
| Duplicate persistence | Its own `DDL` creates `nfp_releases` + `nfp_outcomes`, and `calibrate_from_log` re-implements a hit-rate scorecard that `src/services/source_scorecard.py` already does over `trade_setups`. |
| Wrong tree | Docstring says `from tabs import nfp_reaction`. There is no `tabs` package here. |

One earlier claim of mine needs correcting: `render(engine)` is **not** foreign
to this repo. `pages/surprise_tab.py`, `pages/threat_board_tab.py` and
`pages/liquidity_hunt_tab.py` each build a SQLAlchemy engine from
`src.core.secrets.db_config()` / `src.db.connection.current_db_config()` and
call `render(conn)` under a `__main__` guard. So the signature is a known
pattern — it just has no caller yet. It becomes moot anyway (see below).

**Outcome:** the page renders on-theme, appears in the sidebar, logs its read to
`tool_usage_log`, and persists its high-conviction directional calls to
`trade_setups` so the Source Scorecard can resolve them — with **no new tables**.

## Decisions taken (owner-confirmed)

1. **Reuse existing tables.** Drop `DDL`, `ensure_tables`, `save_release`,
   `save_outcome`, `load_outcomes`, `wilson`, `calibrate_from_log` (~150 lines).
   The read goes to `tool_usage_log`; the directional calls go to
   `trade_setups`; `source_scorecard` resolves them against price action.
   **Trade-off, stated plainly:** this loses the bespoke per-symbol OLS beta
   refit. The betas in `INSTRUMENTS` stay priors indefinitely. If the generic
   scorecard turns out not to answer "is my XAUUSD beta right", the calibration
   loop comes back as its own change with its own version bump — not as a
   passenger here.
2. **Persist high-conviction rows** to `trade_setups` under
   `source='nfp_reaction'` (14 chars, fits `VARCHAR(20)`).

Removing the persistence layer also removes the need for a SQLAlchemy engine:
`log_tool_usage` and `persist_signals` resolve their own psycopg2 target. So
`render()` takes **no argument**, and the `__main__`/engine question disappears.

## Global constraints

- Never commit. Show the diff.
- Every completed task bumps the patch: read `VERSION`, add one, write it back
  via `python deploy/sync_version.py <next>`. Global sequence, no reservations.
- Tests first inside each task (`test-driven-development`).
- Behaviour of the surprise/scoring math is preserved **byte-for-byte** when
  moved. Diff against the staged file to prove it.
- No second broker-symbol table — reuse `broker_symbols.normalize_symbol`.

## Starting state (measured)

- `VERSION` → `1.10.20`
- `pages/nfp_reaction.py` → 679 lines, staged-and-modified, imports `sqlalchemy`,
  never executes.
- `signal_sweep.PAGES` → 27 entries; `AUDIT_ONLY` → `{confluence_zone_4h, confluence_checker}`.
- `normalize_symbol` verified live: `XAUUSD→XAU/USD`, `XAGUSD→XAG/USD`,
  `EURUSD→EUR/USD`, `USDZAR→USD/ZAR`; `DXY`/`US500`/`NAS100`/`US10Y`/`BTCUSD` → `None`.
- `ChartKit.bars` has **no** `orientation` argument — vertical only.

---

## Task 1 — Extract the pure core to `src/core/nfp_reaction.py`

The scoring must be unit-testable without a Streamlit runtime, same as
`src/core/fibo_ribbon.py` is the pure twin of its page.

**Steps**
- [ ] Write `tests/test_nfp_reaction.py` first and watch it fail:
  - `compute_surprise` renormalises weights when U3/AHE are omitted (headline-only
    entry still scores); U3 is inverted (a *lower* print is hawkish).
  - `Surprise.label` / `.direction` boundaries at 0.35 / 1.0 / 2.0.
  - `score_instruments` conviction → 0 when the two channels cancel, → 1 when
    they agree in sign; sorted by `abs_score` descending.
  - Regime rotation flips the sign of the gold and equity rows between
    "Rates-led" and "Growth-scare".
  - `next_nfp_date` returns the first Friday, rolls to next month once past,
    and `release_datetime_sast` is DST-aware across a US DST boundary.
  - `board_to_signals` (new, Task 3) maps only registry-resolvable symbols.
- [ ] Move `Surprise`, `SURPRISE_SD`, `SURPRISE_W`, `compute_surprise`, `REGIMES`,
  `score_instruments`, `release_datetime_sast`, `next_nfp_date`, `timing_frame`
  into `src/core/nfp_reaction.py`. **Rename the module-level `INSTRUMENTS`
  constant to `EXPOSURES`** so it cannot be confused with the registry's
  `INSTRUMENTS`.
- [ ] Type hints on everything moved (already mostly present).
- [ ] Prove equivalence: `git show :pages/nfp_reaction.py` diffed against the
  moved bodies — formulas, weights and thresholds unchanged.
- [ ] `python deploy/sync_version.py <VERSION+1>`

## Task 2 — Rewrite `pages/nfp_reaction.py` as a real page

**Steps**
- [ ] Delete the `try: from theme import ...` block entirely; import
  `from src.ui.theme import BloombergTheme as T` and use its tokens
  (`T.BG`, `T.GREEN`, `T.RED`, `T.AMBER`, `T.GREY`) — no local hexes.
- [ ] Delete `DDL`, `ensure_tables`, `save_release`, `save_outcome`,
  `load_outcomes`, `wilson`, `calibrate_from_log`, and the `sqlalchemy` /
  `math` imports.
- [ ] `render()` takes no argument. Add the page entry following
  `pages/event_week_vol_tab.py`'s tail: `st.set_page_config(page_title="NFPR ·
  NFP Reaction Map", page_icon="🧾", layout="wide")` → `T.apply()` → sidebar
  (**page controls first, then `st.divider()`, then `render_sidebar_nav()`** —
  the sidebar contract) → `render()`, called unconditionally at module bottom.
- [ ] Move the release-date picker and the regime selectbox into the sidebar
  (symbol/scope controls first per the contract); the four actual-vs-consensus
  number inputs stay in the body where they read as a data-entry form.
- [ ] Charts: `chain_figure` and `board_figure` stay hand-rolled `go.Figure` —
  ChartKit's primitives are price-panel shaped and `bars` has no horizontal
  orientation, so these are page-specific composites like the volume profile
  and radar. But re-point them at `BloombergTheme` tokens and render with
  `st.plotly_chart(fig, use_container_width=True, config=ChartKit.PLOTLY_CONFIG)`.
- [ ] Replace the "Log the release / Save outcome" block and the out-of-sample
  scorecard with a short caption + `st.page_link` to the Trade Journal's
  Source Scorecard tab.
- [ ] `python deploy/sync_version.py <VERSION+1>`

## Task 3 — Persistence

**Steps**
- [ ] In `src/core/nfp_reaction.py`, add pure
  `board_to_signals(board, release_date, regime, composite, *, min_abs_composite=0.5,
  min_conviction=0.45) -> list[dict]`:
  - `pair = normalize_symbol(row.symbol)`; **skip when `None`** — `DXY`, `US500`,
    `NAS100`, `US10Y`, `BTCUSD` stay on the board for display but never reach
    `trade_setups`. Same rule `disconnect_mon` uses.
  - `bias = "Bullish"/"Bearish"` from `score` sign.
  - `bar_time = release_date` — one stored row per pair+bias **per release**,
    which is exactly the period on which this read is new. Mirrors
    `cot_composite` keying on the CFTC report date.
  - Carry `conviction`, `strength_score` (scaled `abs_score`), and a `thesis`
    naming the regime and the composite z.
  - Return `[]` when `abs(composite) < min_abs_composite` — an in-line print is
    the page declining to forecast, same policy as `forecast_dashboard`.
- [ ] In the page, after scoring:
  - `log_tool_usage("nfp_reaction", {...})` guarded by
    `NotifyCache("nfp_reaction_log").filter_new([key])` where the key is
    `release_date|regime|rounded inputs` — Streamlit reruns on every widget
    touch and would otherwise log an unchanged read on every keystroke.
  - `persist_signals("nfp_reaction", board_to_signals(...))`.
- [ ] **The widget-dependence rule, addressed explicitly.** CLAUDE.md: *what a
  page persists must never depend on a display widget.* Here the inputs **are**
  widgets. The resolution is the `min_abs_composite` gate: at defaults
  actual == consensus, so `composite == 0.0` and **nothing persists**. A
  headless sweep therefore stores zero rows unless a human has typed a real
  release — which is correct, not a bug. **Do not add a `PREPARE` hook** for
  this page: a hook that types in actuals would fabricate a release.
- [ ] `python deploy/sync_version.py <VERSION+1>`

## Task 4 — Register

**Steps**
- [ ] `src/pages_lib/navigation.py` → add to the **🔬 RESEARCH LAB** section,
  next to `SURP`: `NavEntry("NFPR", "NFP Reaction Map", "🧾", "pages/nfp_reaction.py")`.
  (Research Lab, not Morning Brief: it is a what-if/prep tool that only carries
  live information for ~60 minutes once a month.)
- [ ] `src/services/signal_sweep.py` → add `("nfp_reaction", "pages/nfp_reaction.py")`
  to `PAGES`. Required — `tests/test_signal_sweep.py::test_every_persisting_page_is_registered`
  fails otherwise. **Not** in `AUDIT_ONLY` (it does call `persist_signals`).
  It is pure math with no yfinance call, so it adds ~1s to a pass.
- [ ] `.foglamp/introspect.py` then `.foglamp/render.py --check` — add the page
  node and its edges to `scan.json` (edges to `tool_usage_log` and
  `trade_setups`, labelled with the business rule, e.g. `"only |z| >= 0.5"`).
- [ ] `docs/System_Guide.md` + README: add the page under Research Lab, then
  `python docs/generate_guide_pdf.py`.
- [ ] `python deploy/sync_version.py <VERSION+1>` and `python deploy/verify_deploy.py`.

---

## Verification

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_nfp_reaction.py tests/test_signal_sweep.py -v
```

```bash
PYTHONIOENCODING=utf-8 python -m pytest
```

```bash
python -m py_compile pages/nfp_reaction.py src/core/nfp_reaction.py src/pages_lib/navigation.py src/services/signal_sweep.py
```

Page smoke under AppTest (must report `exceptions: []`):

```bash
PYTHONIOENCODING=utf-8 python -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file('pages/nfp_reaction.py', default_timeout=120); at.run(); print('exceptions:', [str(e.value) for e in at.exception])"
```

Sweep it in isolation — expect **`saved 0`**, which proves the in-line gate:

```bash
python -m src.services.signal_sweep --only nfp_reaction --no-sync
```

Then in the browser (`streamlit run app.py` → NFPR): type a real hawkish
release (e.g. NFP 275 vs 150, AHE 0.5 vs 0.3), confirm the board flips, the
chain recolours, low-conviction symbols are called out, and one row per
resolvable pair appears in the Trade Journal filtered to `source='nfp_reaction'`.
Re-run with the same inputs and confirm **no** duplicate row (dedupe on
pair+bias+release-date).

Architecture map:

```bash
python .foglamp/introspect.py && python .foglamp/render.py --check
```

## Watch item (not a blocker)

`pages/surprise_tab.py` already z-scores actual-vs-forecast across the calendar.
The two overlap conceptually but not in output — one is a breadth index, the
other a single-release exposure map. If a third surprise scorer ever appears,
that is the moment to consolidate, not now.
