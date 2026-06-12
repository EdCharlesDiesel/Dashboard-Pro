# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A professional forex/metals day-trading dashboard built on Streamlit. Two apps share one repo:

- **`daily-trading-checklist.py` + `pages/`** — the main multipage "terminal" system (22 pages). This is where active development happens.
- **`app.py`** — an older single-page tab dashboard ("Macro Dashboard Pro") kept running in parallel. It shares `src/core/` with the multipage app.
- **`archive/`** — dead experiments; never touch.

## Commands

```bash
# Run the main app (registers pages/ as a multipage app)
streamlit run daily-trading-checklist.py

# Run the legacy tab dashboard
streamlit run app.py

# Syntax-check everything you changed (no linter is configured)
python -m py_compile daily-trading-checklist.py pages/*.py $(find src -name '*.py')
```

There is no test suite. Smoke-test a page headlessly with Streamlit's AppTest:

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
- Postgres persistence is in `src/db/trade_repository.py` (`trade_setups` table; schema auto-migrates via `ADD COLUMN IF NOT EXISTS`). Connections use `with closing(self._connect()) as conn, conn, cursor` — closing handles close, the middle `conn` handles commit/rollback. DB credentials come from the sidebar/session state, not config files.
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
