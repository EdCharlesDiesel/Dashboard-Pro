# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A professional forex/metals day-trading dashboard built on Streamlit, one
multipage app:

- **`app.py`** — the entry point. A thin shim that runs `DailyTradingPage`
  (the 18-point checklist). This is the master: every page follows its
  terminal look (`BloombergPage` + `BloombergTheme`).
- **`pages/`** — the 22 workflow pages, including `market-overview.py` (the
  former standalone "Macro Dashboard Pro", now a tabbed page sharing
  `src/core/`).
- **`archive/`** — dead experiments; never touch.

## Commands

```bash
# Run the app (app.py is the entry; pages/ auto-register as multipage)
streamlit run app.py

# Syntax-check everything you changed (no linter is configured)
python -m py_compile app.py pages/*.py $(find src -name '*.py')
```

Unit tests (pytest) cover the pure logic in `src/` — registry, risk/session/
correlation services, indicator math, the setup/trend scorers, **and the
Postgres repository + DB cache layer** (with a mocked psycopg2 connection). They
never touch yfinance, a live DB, or a Streamlit runtime (fixtures build synthetic
OHLC frames; `TradeRepository(cfg, connect_factory=...)` injects a fake conn):

```bash
# Dev deps (pytest, pytest-cov) live in requirements-dev.txt; config is in pyproject.toml.
pip install -r requirements-dev.txt
PYTHONIOENCODING=utf-8 python -m pytest               # ~5s, no network; prints coverage
PYTHONIOENCODING=utf-8 python -m pytest --runslow --no-cov tests/test_pages_smoke.py  # page smoke tests (live yfinance, ~2min)
```

Tests live in `tests/` (mirrors `src/` module-by-module). `pyproject.toml`'s
`pythonpath = ["."]` makes `import src...` resolve from the repo root, exactly
as it does under `streamlit run`. Coverage runs by default and is **scoped to
the pure-logic core + DB layer** (`[tool.coverage.run] omit` excludes pages, UI,
network fetchers, email/observability) with a **`--cov-fail-under=80` floor**;
coverage currently sits ~92%. **Page smoke tests** (`tests/test_pages_smoke.py`)
run each page under AppTest and assert no uncaught exception; they hit live
yfinance so they're marked `@pytest.mark.slow` and skipped unless `--runslow` is
passed (kept out of the coverage gate). To smoke-test a single page ad hoc:

```bash
PYTHONIOENCODING=utf-8 python - <<'EOF'
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("pages/setup-ranker.py", default_timeout=240)
at.run()
print("exceptions:", [str(e.value) for e in at.exception])
EOF
```

**Known AppTest artifact:** `st.page_link` raises `KeyError: 'url_pathname'` under AppTest because the single-file harness has no multipage registry. The shared nav renderer swallows this, so pages run — but never "fix" page_link based on AppTest behavior; it works fine under `streamlit run`.

- Set `PYTHONIOENCODING=utf-8` when piping output on Windows — pages print ◇/emoji glyphs that crash cp1252.
- Changes to `.streamlit/config.toml` need a server restart; code/CSS changes only need a browser reload.
- AppTest runs hit live yfinance (~5–60 s per page). "No pairs scored" usually means real market conditions, not a bug.

## Architecture

### The multipage system

Pages come in two generations that must look identical:

1. **Framework pages** (Checklist, Setup Ranker, Correlations, DXY vs Gold) subclass `BloombergPage` in `src/pages_lib/base.py` — a template method (`configure()` / `sidebar()` / `body()`) that handles page config, theme injection, sidebar nav, and the status bar. The files in `pages/` and the root entrypoint are 5-line shims; implementations live in `src/pages_lib/`. **New pages should use this framework.**
2. **Legacy pages** (the other 18 in `pages/`) are standalone scripts with inline CSS. They are visually harmonized by calling `BloombergTheme.apply()` immediately after `st.set_page_config()` and by a global flatten rule in the theme (no rounded corners anywhere). Their inline hexes were bulk-remapped to the terminal palette — don't reintroduce GitHub-dark colors (`#0d1117`, `#388bfd`, …) or `plotly_white`.

### Single sources of truth (never duplicate these)

- **`src/instruments/registry.py`** — `INSTRUMENTS` (21 forex pairs + Gold/Silver/Platinum with ticker, pip value, pip size), `CORR_GROUPS` (correlated-exposure groups), `TYPICAL_SPREADS`, `TREND_TIMEFRAMES`. The registry is dict-compatible (`INSTRUMENTS[name]["ticker"]`) for legacy callers, but `INSTRUMENTS.get(name)` returns an `Instrument` dataclass — use attribute access on it, never `.get()`.
- **`src/pages_lib/navigation.py`** — `NAV_SECTIONS` defines the sidebar nav for all 21 pages, grouped as the 8-step trading workflow (scan → filter → weekly bias → daily confirm → 4H zone → 15M trigger → risk/execute → review). To add/reorder pages, edit only this file. Legacy pages call `render_sidebar_nav()`; framework pages get it from the base class.
- **`src/ui/theme.py`** — `BloombergTheme` color tokens + injected CSS. Terminal green-on-black. ⚠️ The accent token is named `AMBER` but holds terminal green `#00ff41` (renamed look, kept name to avoid touching every call site). `.streamlit/config.toml` `[theme]` must be kept in sync with these tokens so native widgets (dropdown popovers, sliders) match the custom CSS.
- **`src/core/signals.py`** — `score_setup()` is the 10-point multi-timeframe scorer used by Setup Ranker and `app.py`. Indicator math also lives in `src/indicators/` (`TechnicalIndicators`, `TrendSignalEvaluator` — the 6-condition trend scorer).

### Sidebar contract (applies to every page)

Brand/page header → **page controls (symbol picker first)** → divider → navigation. The base class enforces this for framework pages; legacy pages follow it by convention.

### Data layer

- All yfinance fetchers are wrapped in `@st.cache_data(ttl=300)` (600 for correlations). **Never remove the cache decorator** — Streamlit reruns the whole script on every widget interaction, and uncached fetchers hammer Yahoo into rate-limiting.
- yfinance quirks handled everywhere: MultiIndex column flattening, spot FX (`=X` tickers) reports zero volume (AMD scanner substitutes a true-range activity proxy), 1-hour bars limited to ~730 days.
- Postgres persistence is in `src/db/trade_repository.py` (`trade_setups` table; schema auto-migrates via `ADD COLUMN IF NOT EXISTS`). Connections use `with closing(self._connect()) as conn, conn, cursor` — closing handles close, the middle `conn` handles commit/rollback. DB credentials come from the sidebar/session state, not config files. `TradeRepository` is Streamlit-free; its `connect_factory` arg lets callers inject a pooled or fake connection. **The app auto-connects on startup** — `src/db/connection.py` `auto_connect()` runs once per session (from `SessionStateBootstrap.init()` and the top of `trade-journal.py`), resolving the target from `secrets.toml [database]` (via session `db_*` defaults), initialising both schemas, and flipping the `db_ok`/`journal_db_ok` gates. No manual "Connect" click is needed; the sidebar button is now a "Reconnect" for when credentials are edited (`auto_connect(force=True)`). It never raises — a down/unconfigured DB just leaves the gates False with a message. `save_setup(row, source=…)` tags the row's origin (`source` column, default `'checklist'`); the no-source call path is kept byte-identical.
- **Pages auto-save their signals (system-wide).** Signal/scanner pages persist their reads to `trade_setups` through one shared service, **`src/services/signal_store.py`** → `persist_signals(source, signals)`. It owns every cross-cutting concern so a page is a ~3-line call: per-source dedupe (a `NotifyCache` ledger keyed pair+bias+rounded-entry, surviving reruns/restarts), DB-target resolution via `market_cache._resolve_cfg()`, `source=<page>` tagging, pooled write straight to Postgres, and **graceful no-op** when the DB is down (without consuming the dedupe ledger, so a later configured run still saves). Signals are loose dicts (`pair`+`bias` required; `entry`/`stop_loss`/`take_profit_*`/`strength_score`/`conviction`/`thesis` optional) mapped by the pure, unit-tested `signals.signal_to_setup_row()` (ticker/pip_size from the registry; price targets → pips; sizing fields NULL; full context in `checks_detail` JSON). **Wired pages** (`source` tag): `market_overview`, `setup_ranker` (Grade-A), `weekly_swing`, `trend_signals`, `daily_trend`, `daily_macd`, `amd_scanner`, `confluence_checker`/`confluence_zone_4h` (non-directional → `bias='Neutral'`), `twenty_day_breakout` (open setups only), `vwap_ema_gold` (latest-bar only), `fib_entry`, `market_structure`, `weekly_ema`, `predictive` (price-vs-SMA), `forecast_dashboard` (only when the Monte-Carlo model shows an EDGE), `dxy_gold` (XAU/USD BUY/SELL read), `currency_strength` (strongest-vs-weakest pair, only when the leg spread clears a minimum threshold), `smart_money` (accumulation/distribution; free-text ticker), `risk_reversal` (contrarian read; **never on synthetic demo data**), `seasonality` (current-month bias), `cot_composite` (`pages/cot_composite_trade_signal.py`'s composite BUY/SELL/STRONG_* read over percentile/z-score extremes + divergence + OI context, mapped currency→registry pair with sign-flip for USD/XXX quote-leg currencies (JPY/CHF/CAD); skips NEUTRAL, DXY (no tradable registry pair), and low-confidence reads unless a collapse-watch fires — **source tag is `cot_composite`, not the module's full name, because `trade_setups.source` is VARCHAR(20)**), and `disconnect_mon` (`pages/disconnect_monitor_tab.py`'s driver/asset residual-z "disconnect" read — **only** for the one configured pair with a tradable registry mapping, `Real yield vs Gold` → XAU/USD via `ASSET_TO_REGISTRY_PAIR`; fires only when `|z| >= threshold` AND the regime check is `ALIVE` (expected correlation sign holds), fading the rich/cheap side — the other three configured pairs (DXY, DBC, Nasdaq) have no registry mapping and stay audit-only). **Not wired to `trade_setups`** (no directional pair+bias signal, but see the audit trail below): `trade-journal` (writes MT4 imports directly via `TradeRepository`, not through the signal store), `reports`, `system-logs`, `atr-volatility`, `event_impact`, backtest/simulation pages (`double_zeros`, `backtest-workflow`, `cot_trade_signal_walk_forward_backtest_harness` — no-lookahead walk-forward validation of `cot_composite_trade_signal`'s edge, reusing its exact scoring code path; reports win rate/avg return by signal state and horizon, plus a naive equity curve — a research tool, not itself a signal source), and the weak/ambiguous reads left off by choice (`weekly-rsi`, `macro-bias` [scores currencies not pairs], `regime`, `forex_fundamentals`, `trading_lab`). Writes go straight to Postgres, so a saved signal is reflected in the journal/stats on the next read with no cache to invalidate. Note `signal_to_setup_row` caps `verdict` at the `trade_setups.verdict` VARCHAR(20) width, and any new `source` tag must independently respect that same VARCHAR(20) column.
- **Non-signal pages still log to Postgres — a separate audit trail, not `trade_setups`.** `tool_usage_log` (schema + `log_tool_usage(tool, payload)` in `src/db/trade_repository.py`, one JSONB `payload` column) exists precisely for pages that produce a real read but not a directional pair+bias trade signal, via the thin wrapper **`src/services/tool_log.py`**. Every call site dedupes on the read's own "shape" (rounded inputs, or the report/week date for weekly CFTC data) through a page-specific `NotifyCache` namespace — **not** the signal store's dedupe — since Streamlit reruns the whole script on every widget touch and would otherwise log an unchanged read on every rerun. Wired: the calculators/views `rr_calculator`, `account_risk`, `correlations`, `news_filter`, `stop_structure` (an interactive tool's inputs+outputs, logged once per distinct calculation); the COT positioning-context pages `cot_tab`, `cot_signals`, `cot_open_interest`, `gold_cot`/`wti_cot` (`src/pages_lib/commodity_cot_lib.py`, shared by `pages/gold_cot_tab.py` / `pages/oil_cot_tab.py`) — all explicitly "crowdedness read, not a trade trigger" per their own captions, logged per instrument+week; and the research tools `disconnect_mon` (logged for **all four** configured pairs, not just the one that also escalates to `trade_setups` — see above), `event_week_vol`, `overnight_drift` (ES futures isn't a registry pair, so this stays audit-only even though it's genuinely directional-flavored). **Not logged at all**: `trade-journal`/`reports`/`system-logs` (pure views over already-persisted data) and the backtest/simulation pages (research tools whose whole output *is* the report, not a fire-and-forget read worth journaling).
- **`NotifyCache` dedupe ledgers are themselves durable in Postgres**, not just local JSON. `src/services/alert_service.py`'s `NotifyCache` mirrors its key set to `app_state` under `notify_cache_{namespace}` (same dual-write pattern as `account_state.py`/`score_history.py`: local JSON is the fast path + offline fallback, `load()` unions local ∪ DB so a key written from either side is never lost, `reset()` clears both). This closes the one gap in the "everything survives a restart" story — previously a deleted `*_notify_cache.json` could cause a duplicate save with zero DB backup.
- **DB pooling lives in `src/db/cache.py`** (keeps the repository Streamlit-free). `pooled_repository(cfg)` borrows from a `@st.cache_resource` connection pool (one per DB target); the `cached_*` read fns and `set_state`/`cached_get_state` are **direct pooled passthroughs straight to Postgres — there is no read cache** (Redis was removed). `clear_read_caches()` is retained as a **no-op** so the write→invalidate call sites (signal store, journal/stats pages) don't need to change; with no cache there's nothing to invalidate. Schema init (`init_schema`) stays on a plain (lazy) `TradeRepository` so bad credentials surface as a message instead of crashing on eager pool creation.
- Secrets live in `.streamlit/secrets.toml` (gitignored): `[api]` FRED key, `[database]`, `[gmail]` (AMD scanner email alerts). `.streamlit/config.toml` is theme-only and is tracked — never put keys in it.

## Domain conventions (forex)

- **Prices**: format with 5 decimals when `abs(price) < 100` (FX), else 2–3 (JPY crosses, metals, indices). Two-decimal EUR/USD output is a bug.
- **Risk model**: SL = 1.5 × ATR14 (in pips), TP1 = 2R, TP2 = 3R. Lot size = risk_amount / sl_pips / pip_value. R-multiple = pips gained / SL pips.
- **The 18-point checklist**: checks 11–16 are the "critical path"; verdict is GO only when ≥16/18 checked AND all critical checks pass. A daily-loss limit (2 losses/day, queried from Postgres) blocks new trade entry.
- **Sessions (UTC)**: London Kill Zone 07–09, NY Kill Zone 12–14, London Close 15–17, Tokyo 00–03; everything else is Dead Zone. Prime windows are the only "green" entries.
- **Metals use forex symbols, not icons**: Gold = XAU/USD, Silver = XAG/USD, Platinum = XPT/USD (Yahoo tickers stay GC=F / SI=F / PL=F). Older Postgres rows may carry the previous display names ("Gold").
- **AMD scanner** is fixed to a daily-trading preset: 1H bars × 1 month (`PERIOD`/`INTERVAL`/`BARS_PER_DAY` constants), detection on the full month, chart windowed to the current trading week. Volume-profile colors follow candle green/red by buy/sell dominance.
- **Correlation stacking**: before adding exposure, `CorrelationService.check_exposure()` warns when an open trade in the same `CORR_GROUPS` group shares the direction (e.g. long EUR/USD + long GBP/USD = doubled USD risk).

## Code conventions

- Python 3.14 venv, but **avoid 3.12+-only syntax** (e.g. nested same-quote f-strings) — portability was a deliberate decision here.
- UI components (`src/ui/components.py`) render HTML strings via `st.markdown(unsafe_allow_html=True)`; colors must come from `BloombergTheme` tokens, not inline hexes.
- Refactors of legacy logic preserve behavior byte-for-byte (indicator formulas, SQL, thresholds) — when moving code, diff against `git show HEAD:` to prove equivalence.
- `st.cache_data`-decorated functions are cleared selectively (`fn.clear()`) for one-off refresh buttons, or globally (`st.cache_data.clear()`) for "rescan all" buttons.
