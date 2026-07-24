# Dashboard-Pro

A professional **forex & metals day-trading terminal** built on Streamlit. It
walks you top-down through a single trading decision — scan the market,
synthesize a composite second opinion, filter the day, establish bias,
confirm, find the zone, wait for the trigger, size the risk, execute, and
review — across **30 daily-workflow pages** plus a **15-page weekly/research
lab** (COT validation, pairs-research, backtests, forecasting) — 45 active pages total,
all sharing one analysis engine and one Bloomberg-style terminal theme.
(Trimmed from 52 for a leaner daily forex workflow — the retired pages live in
`archive/pages/`, not deleted; see [Archived pages](#archived-pages) below.)

**New here?** Open **[📖 System Guide & Playbook](pages/system_guide.py)** in
the app (or `docs/System_Guide.md` / the downloadable PDF) for a full,
page-by-page deep dive — what each page's numbers mean and how to use them —
plus the complete methodology behind the new **🔮 Instrument Predictor**
(its Section 9). This README stays intentionally concise; the guide is the
long-form reference.

It trades the 21-pair forex universe **plus metals** — Gold (XAU/USD), Silver
(XAG/USD) and Platinum (XPT/USD) — from one instrument registry, so every page
scans the identical universe with the identical tickers.

> The master page is the **18-point Daily Checklist** (`app.py`). Every other
> page feeds a check on it. A trade is a **GO** only when ≥16/18 checks pass and
> all critical checks (11–16) are clear.

---

## The house view — one direction, everywhere (2026-07 sync architecture)

Pages used to each define "trend" their own way, on their own private data
windows — so the same pair could read BUY on one page and SELL on the next.
That's gone. Four layers keep every page telling one story:

1. **One data spine** (`src/services/market_data.py`) — every page fetches
   OHLC through canonical windows (weekly = 2y daily resampled, daily = 300d,
   4H = 90d hourly resampled, one shared TTL), Postgres read-through cached.
2. **One bias engine** (`src/core/bias.py`) — the canonical BULLISH / BEARISH
   / NEUTRAL per timeframe (EMA20 vs EMA50 + price position, unanimous votes)
   and a Weekly-weighted composite **house view** per instrument.
3. **Visible everywhere** — every directional page shows a house-view strip
   for its focus pair and flags in red when its own lens opposes it; Market
   Overview's ideas and the Daily Cockpit's aligned ideas carry inline
   conflict flags. **[MTF Matrix](pages/mtf-matrix.py) is the consensus
   board**: the whole universe's house view in one grid, with a
   **walk-forward Validation tab** (no-lookahead, unit-test-proven) to check
   what the house view was actually worth historically.
4. **One parameter home** — indicator defaults (EMA 20/50/200, MACD 12/26/9,
   swing strength) live on `AppConfig`; page sliders default from them and
   label any deviation a "custom lens".

**Read the validation honestly:** the first 5-year walk-forward (EUR/USD,
XAU/USD, WTI/USD) showed the house view has **no standalone entry edge** —
hit rates sat below 50% and trend-alignment was actively harmful on oil. Use
it as what it is: a *consistency and regime filter* that stops pages (and
you) from contradicting each other — the edge must come from the setup layer
(location, trigger, risk) below. The **Source Scorecard** tab in the
[Trade Journal](pages/trade-journal.py) closes the loop: it resolves every
persisted signal against what price did next and ranks each signal source by
realized expectancy — over time it tells you which pages have earned trust.

---

## Project structure

```
Dashboard-Pro/
├── app.py                  # Entry point — the 18-point Daily Checklist (master page)
├── pages/                  # 45 active workflow pages (auto-register as Streamlit multipage)
├── src/
│   ├── core/               # analyzer, signals, config — the shared analysis engine
│   ├── indicators/         # EMA/RSI/MACD/ADX/ATR + the 6-condition trend scorer
│   ├── instruments/        # registry.py — the single source of truth for instruments
│   ├── services/           # ATR, correlation, session, risk, forecast, MT4 import …
│   ├── pages_lib/          # BloombergPage framework + navigation (the workflow order)
│   ├── db/                 # Postgres journal (trade_repository) + cache/pool layer
│   └── ui/                 # terminal theme & HTML components
├── tests/                  # pytest suite (logic + DB layer, 80% coverage floor; page smoke tests)
└── archive/                # legacy experiments + 21 retired pages (archive/pages/) — not imported, do not touch
```

The sidebar navigation is defined in **one file**,
[`src/pages_lib/navigation.py`](src/pages_lib/navigation.py) — it is the single
source of truth for the workflow order below. To reorder or add pages, edit only
that file (and keep this README in sync).

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest/pytest-cov for the test suite
```

**2. Add your secrets** *(all optional — pages degrade gracefully without them)*

Secrets live in `.streamlit/secrets.toml` (gitignored — never put keys in the
tracked `config.toml`):
```toml
[api]
FRED_API_KEY = "your_key"          # live macro data (Macro Bias / Fundamentals)

[database]                          # Postgres journal (Trade Journal / Checklist save)
host = "localhost"
port = 5432
dbname = "dashboardprov1"
user = "postgres"
password = "your_password"

[gmail]                             # optional email alerts from the AMD scanner
user = "you@gmail.com"
app_password = "your_app_password"
```
The DB connection is also editable live from the **Checklist sidebar → Connect &
Init DB** (schema auto-creates).

**3. Run the dashboard**
```bash
streamlit run app.py
```

**4. (optional) Run the tests**
```bash
python -m pytest                                   # fast logic + DB tests, with coverage
python -m pytest --runslow --no-cov tests/test_pages_smoke.py   # page smoke tests (live data)
```

---

## How the system works — the daily path vs. reference & research

44 pages is too many stops for any daily routine to survive. The pages are
grouped by **when in the day you'd actually touch them**, not the order they
were built — separating the daily path (walked in order, execute-or-pass)
from reference and research (visited on demand, never mid-session). This
table mirrors [`src/pages_lib/navigation.py`](src/pages_lib/navigation.py)
exactly — that file is the single source of truth; if the two ever disagree,
the code wins. **For what each page's output actually means and how to act
on it, see the [System Guide](pages/system_guide.py)** — this table is
deliberately just a map.

Re-architected 2026-07 around a 17:00–20:00 SAST session: Morning Brief (5
min, sets bias, no decisions) → Pre-Session (build the shortlist, ~20 min,
frozen once the session opens) → Session (execution only — nothing here
generates a new opinion) → Weekend (weekly-bias tools + the full COT suite)
→ Research Lab (everything else, visited when building or validating, never
during a session) → Reference. MTF Matrix leads the Morning Brief since
becoming the house-view consensus board (2026-07) — it's the canonical
direction, not another scanner opinion. Of the remaining scanners that
produce directional opinions (Setup Ranker, AMD Scanner, Trend Signals,
20-Day Breakout), only Setup Ranker (primary) and Trend Signals (confirmer)
hold a daily-path slot — the others stay in Research Lab until the Trade
Journal's Source Scorecard shows which one actually earns it. Daily Trend / Daily
MACD / Market Structure do more analysis than the Ranker's checklist booleans
(histogram momentum, EMA100/200 stack + slope, CHoCH/BOS) so they're kept as
full pages, just relocated out of the 7-touch day. Stop Structure, R:R
Calculator, and Account Risk were merged into one **Risk Suite** page — they
were one decision ("where's the stop, what's the size") spread across three
separate re-entries of the same instrument/balance/risk inputs.

### 🌅 Morning Brief — 5 min with coffee, sets bias, no decisions
| Page | What it's for |
|------|---------------|
| [🧭 MTF Matrix (House View)](pages/mtf-matrix.py) | **First stop.** The consensus board — the canonical Weekly/Daily/4H house view for the whole universe, plus the walk-forward Validation tab. Every other page anchors to this read. |
| [🛫 Daily Cockpit](pages/daily_cockpit_tab.py) | Regime → rate bias → events → fresh setups, one screen. Covers the full registry. |
| [📊 Market Overview](pages/market-overview.py) | Headline KPIs + price table across FX, metals, indices. |
| [📰 News Filter](pages/news-filter.py) | Today's landmines, in SAST — wait or skip. |

### 📋 Pre-Session — 16:30 SAST, build the shortlist (~20 min)
| Page | What it's for |
|------|---------------|
| [🎰 Setup Ranker](pages/setup-ranker.py) | **Primary scan.** Direction-only scoring (7-8 criteria; ATR/Spread/4H-Zone are a separate quality gate, not points) — 80%+ is Grade A. Grade-A email alerts are **on by default** and also run **unattended**: a background worker re-scans every 5 min inside the server, so alerts arrive even with no browser open. |
| [📡 Trend Signals](pages/trend-signals.py) | **Confirmer.** Confirms/denies the Ranker's top picks — 50/200 EMA, RSI, MACD, with ADX as a hard trend gate (not one point of six). |
| [💪 Currency Strength](pages/currency-strength.py) | Kill conflicted exposure — ranks the 9 registry currencies strong → weak. |
| [🔗 Correlations](pages/correlations.py) | Kill duplicated exposure — stacked-risk check before adding a correlated pair. |
| [🔀 Confluence Check](pages/confluence-checker.py) | The final gate: what survives? At least 2 of 3 confluences (Fib + Pivot + EMA) must line up. |

### ⚡ Session — 17:00–20:00, execution only
The shortlist is frozen at session open. Nothing in this section is
analytical — if you can re-open the Ranker mid-session you will, and that's
how a flat day becomes a revenge trade.
| Page | What it's for |
|------|---------------|
| [⚡ 15M Fib Entry](pages/15m-fib-entry.py) | Retrace into the 0.382–0.618 golden zone + a confirming candle. Email alerts on by default; auto-rescans every 5 min while open. |
| [🛡️ Risk Suite](pages/risk-suite.py) | Stop Structure + R:R Calculator + Account Risk, one page, three tabs — where's the stop, is R:R ≥2:1, what's the size. |
| [📓 Trade Journal](pages/trade-journal.py) | Log at entry, not after. Also the 19:00 post-session review stop — equity curve, win rate vs 66% target, and the **Source Scorecard** tab ranking every signal source by realized expectancy. |

### 📅 Weekend — weekly bias & the full COT suite
Having these in the daily path just adds scroll — check once a week, not
every session.
| Page | What it's for |
|------|---------------|
| [📉 Weekly EMA](pages/weekly-ema.py) | Weekly 20/50 EMA alignment = the macro trend. |
| [🔄 Weekly Swing](pages/weekly-swing.py) | Weekly pivot swing setups + daily confirmation. |
| [📔 Swing Playbook](pages/swing_playbook_tab.py) | Weekly pre-flight: price, COT crowdedness, gold disconnect, supply/demand zones, vol-scaled sizing, written thesis journal. |
| [🏛️ COT Positioning](pages/cot_tab.py) | Weekly CFTC institutional/speculative positioning — a crowdedness read, not a trade trigger on its own. |
| [🧭 COT Signals](pages/cot_signals.py) | Extreme-positioning + price/positioning divergence, plus a composite squeeze-watch flag. |
| [🧮 COT Open Interest](pages/cot_open_interest.py) | Position-change + open-interest-change context — fresh conviction or an unwind? |
| [🥇 Gold COT](pages/gold_cot_tab.py) | The same positioning read, gold-specific. |
| [🛢️ Oil COT](pages/oil_cot_tab.py) | The same positioning read, WTI-specific. |
| [🏦 Bonds → Gold → DXY](pages/bonds_gold_dxy_app.py) | Treasury-yield ⇄ gold ⇄ dollar seesaw — cross-asset context, educational. |
| [🧩 COT Composite Signal](pages/cot_composite_trade_signal.py) | Combines the three COT pages into one scored STRONG_BUY…STRONG_SELL read plus a collapse-watch flag. Rules-based heuristic — backtest it before trusting it. |
| [🧪 COT Composite Backtest](pages/cot_trade_signal_walk_forward_backtest_harness.py) | No-lookahead walk-forward validation of the composite signal. |

### 🔬 Research Lab — on demand, never mid-session
Visited when building or validating, not part of the routine.
| Page | What it's for |
|------|---------------|
| [📊 AMD Scanner](pages/amd-scanner.py) | Accumulation / Manipulation / Distribution scanner (1H × 1 month). |
| [🧲 Leading Indicators](pages/leading-indicators.py) | DeMarker exhaustion, volume-delta flow proxy, Stoch/Williams %R/CCI context, prior-bar pivots, and the **RSI divergence + pivot confluence** read tuned for hard-trending metals. Audit-only context lens, not a signal source. |
| [🚀 20-Day Breakout](pages/twenty_day_breakout_tab.py) | Donchian-style 20-day breakout candidates. |
| [📦 CME FX Futures](pages/cme-futures-volume.py) | Real, exchange-reported volume for FX via CME currency futures — OBV/CMF that actually mean something. |
| [💱 Predictive Analytics](pages/predictive.py) | Statistical/ML directional read. |
| [🟡 VWAP-EMA Gold](pages/vwap-ema-gold.py) | A dedicated VWAP+EMA strategy view for Gold. |
| [🔮 Instrument Predictor](pages/instrument-predictor.py) | Composite second opinion — Setup Score + Trend Signal + Currency Strength + COT Composite, one weighted read. Heuristic, not a validated model. |
| [💵 DXY vs Gold](pages/dxy-gold.py) | Dollar vs Gold inverse — cross-asset confirmation. |
| [🇺🇸 Indices Correlation](pages/indices-correlation.py) | S&P 500 vs Dow Jones — candles, rolling return-correlation, and the S&P/Dow relative-strength ratio. Cross-asset risk-on/risk-off context (a decoupling or ratio breakout is the tell). Dual chart engine (TradingView pilot / Terminal). Audit-only. |
| [📈 Daily Trend](pages/daily-trend.py) | EMA20/50/100/200 stack + slope + crossover-recency + gap history — more than the Ranker's single boolean. |
| [📊 Daily MACD](pages/daily-macd.py) | Full histogram-momentum analysis (rising/falling counts, acceleration, 6-way verdict) — more than the Ranker's MACD>signal check. |
| [🏗️ Market Structure](pages/market-structure.py) | HH/HL/LH/LL + CHoCH/BOS across 4 timeframes — more than the Ranker's single Daily read. |
| [🎯 4H Confluence Zone](pages/4H-confluence-zone.py) | Fib + Pivot + EMA20 overlap detail behind the Pre-Session confluence check. |
| [📅 Busy-Week Anatomy](pages/event_week_vol_tab.py) | Historical study of price behavior during high-event-density weeks. |
| [🔌 Disconnect Monitor](pages/disconnect_monitor_tab.py) | Tests "this divergence should close" theses via a rolling-regression residual z-score. |
| [🌙 Overnight Drift](pages/overnight_drift_tab.py) | Studies the overnight-session return pattern for index futures. |
| [⏱️ Optimal Holding Period](pages/holding_period_tab.py) | Backtests how many days a breakout-style entry should be held, per pair. |
| [🧮 Quant Models Lab](pages/quant_models_tab.py) | Engle-Granger cointegration, Kalman dynamic hedge ratio, OU mean-reversion, GARCH(1,1) vol regime, UIP/carry, GBM null test. |
| [🔭 Forecast](pages/forecast_tab.py) | GARCH(1,1) volatility cone + a transparent driver score + a narrative, journaled and self-scored against realized outcomes. |
| [😲 Surprise Awareness](pages/surprise_tab.py) | Economic surprise index, an event-proximity gate, and the gold~oil regime-inversion correlation. |

### 📖 Reference
| Page | What it's for |
|------|---------------|
| [📋 Daily Checklist (18-point)](app.py) | The original pre-trade gate; saves the trade and the journal. Superseded as the daily entry point by the Cockpit → Ranker → Confluence → Execute path above, kept for the detailed 18-point audit trail. |
| [📖 System Guide & Playbook](pages/system_guide.py) | The full deep-dive: every page, what its numbers mean, the Instrument Predictor's methodology. In-app or as a downloadable PDF. |
| [🧱 ABR Toolkit](pages/abr_toolkit_tab.py) | Structure (BoS/CHoCH) + order blocks + auto trendlines + MTF EMA bias + quality-graded trade plan across gold, silver, majors, WTI, BTC. |

---

## Archived pages

24 pages were retired from the daily workflow (moved to `archive/pages/`, not
deleted — full git history intact, code still imports/compiles) to keep the
system focused on a lean, forex-daily-execution path. Three reasons:

- **Not forex-usable**: Smart Money and Volume Profile both need real
  exchange volume; spot FX (`=X` tickers) reports zero on yfinance, so
  volume-based indicators on them are meaningless. [CME FX
  Futures](pages/cme-futures-volume.py) replaces the forex-relevant half of
  Smart Money — CME currency futures (`6E=F`, `6B=F`, `6J=F`, …) carry real,
  exchange-reported volume, so OBV/CMF on them are actually meaningful.
- **Redundant or ambiguous**: several pages overlapped a kept page (Event
  Impact vs News Filter, Weekly RSI vs Weekly EMA/Swing, Trading
  Ideas/MTF Matrix/Technical Chart/Pivots & Fibonacci vs the pages that
  already embed the same numbers), or were already flagged in this codebase
  as a weak/ambiguous signal not wired to auto-save (Macro Bias, Forex
  Fundamentals, Market Regime), or are periodic/research tools rather than
  part of the daily loop (Backtest Lab, Trading Lab, Forecast Lab,
  Seasonality, Reports, System Logs).
- **Superseded by a merge**: Stop Structure, R:R Calculator, and Account Risk
  were one decision spread across three pages — merged into
  [Risk Suite](pages/risk-suite.py).

Archived: Trading Ideas, MTF Matrix (early version), Technical Chart, Pivots
& Fibonacci, Volume Profile, FRED Macro Grid, Smart Money, Forecast Lab,
Macro Bias, Forex Fundamentals, Market Regime, Risk Reversals, Event Impact,
Seasonality, ATR Volatility, Weekly RSI, Double Zeros, Backtest Lab, Trading
Lab, Reports, System Logs, Stop Structure, R:R Calculator, Account Risk. To
bring one back into the daily nav, move its file back to `pages/` and add its
`NavEntry` back to `src/pages_lib/navigation.py`.

---

## Top-down trading framework

The core principle is **confluence** — never trade a single signal. Each step
narrows the universe until only high-probability setups remain.

### Step 1 — Macro backdrop · [Daily Cockpit](pages/daily_cockpit_tab.py)
Before touching a chart, understand *why* a pair should move. The Cockpit's
rate-bias table (per-pair, full registry) and risk-regime read replace the
old dedicated Macro Bias / Forex Fundamentals pages (archived — see
[Archived pages](#archived-pages)) in one fused screen.

| Factor | What to look for |
|--------|-----------------|
| **Interest-rate differential** | The primary driver. Capital flows to the higher-rate currency. |
| **Inflation trajectory** | A central bank hiking to fight inflation = currency strength. |
| **GDP growth divergence** | Faster growth on one side = structural tailwind. |
| **Risk regime** | Risk-off → havens (USD/JPY/CHF/Gold) bid *regardless* of fundamentals. |

**Output:** *"Fundamental bias for [pair] is [Long/Short/Neutral] because [reason]."*
Only look for setups that align with it.

**Weekend add-on — positioning context:** [COT Positioning](pages/cot_tab.py) /
[COT Signals](pages/cot_signals.py) / [COT Open Interest](pages/cot_open_interest.py) /
[COT Composite Signal](pages/cot_composite_trade_signal.py) (all in the Weekend
section) show what leveraged funds are actually positioned for, and
whether that positioning is crowded, diverging from price, or actively unwinding. It's
a crowdedness/contrarian read on top of the fundamental bias above, not a replacement
for it — and it only refreshes weekly.

**A fast second opinion (Research Lab):** [Instrument Predictor](pages/instrument-predictor.py)
combines several of the tools above (plus Setup Score and Trend Signal) into one
composite read before you commit to the deeper weekly → daily → 4H → 15M pass below.
See the [System Guide](pages/system_guide.py) for the full methodology.

### Step 2 — Weekly trend · [Weekly EMA](pages/weekly-ema.py) · [Weekly Swing](pages/weekly-swing.py)
- **EMA20 vs EMA50** — EMA20 above EMA50 with price above both = weekly uptrend.
- **Price vs weekly pivot** — sustained above PP = buyers in control.
- **RSI** — above 50 confirms bullish momentum; above 70, be cautious adding longs.

> **Rule:** If the weekly trend is up, only look for longs on lower timeframes. Never fight the weekly.

### Step 3 — Daily confirmation (Research Lab) · [Daily Trend](pages/daily-trend.py) · [Daily MACD](pages/daily-macd.py) · [Market Structure](pages/market-structure.py)
- **EMA alignment** — daily EMA20>50 should match the weekly bias. A bearish daily cross under a bullish weekly = a *pullback* (a buying opportunity), not a reversal.
- **RSI 40–60 on a pullback** gives room to run; >70 = extended, wait for a reset.
- **MACD histogram** turning from negative toward zero signals exhausting sell-side momentum.
- **ATR** sets your minimum stop distance — anything tighter than 1× ATR is noise.

These three do more analysis than the Setup Ranker's checklist booleans, but
aren't part of the frozen daily path — the Ranker + Confluence Check already
cover this step for the 7-touch day; visit these when validating a candidate
in more depth.

### Step 4 — 4H zone · [Confluence Check](pages/confluence-checker.py) (detail: [4H Confluence Zone](pages/4H-confluence-zone.py), Research Lab)
The **confluence zone** = Fibonacci level (38.2 / 50 / 61.8%) + Pivot S/R + EMA20,
all overlapping at one price. When two or three align there, that is where you execute.

### Step 5 — Entry trigger · [15M Fib Entry](pages/15m-fib-entry.py)
Wait for at least **two of three** within 1–2 candles: Stochastic cross below 25
(longs), 15M RSI reset off 40, lower Bollinger-band touch curling back in.
**Do not enter** if price is still slicing through the zone, the daily shows
⚠️ Conflicting, or the spread is abnormally wide ahead of news.

### Step 6 — Risk definition · [Risk Suite](pages/risk-suite.py) (Stop Structure + R:R Calculator + Account Risk tabs)
- **Stop** below the structure that justifies the trade; SL = 1.5 × ATR14.
- **Targets** TP1 = 2R, TP2 = 3R.
- **Size** = (Account × Risk%) ÷ (SL pips × pip value). Risk 1–2% per trade.
- Only take trades with **≥2:1** R:R to TP1.

> A 40% win rate at 2.5:1 is profitable. A 60% win rate at 1:1 barely breaks even after costs.

### Step 7 — Trade management
1. Stop to **breakeven** once price moves 1R in your favour.
2. Take **50% at TP1**, let the rest run to TP2.
3. Exit without hesitation if a 4H closes the wrong side of EMA20, daily RSI crosses back through 50 against you, or a weekly closes through the PP.

---

## Pre-trade checklist (the 18-point gate, condensed)

```
[ ]  1. Macro bias confirmed (rates, GDP, inflation favour direction)
[ ]  2. Weekly EMA aligned with trade direction
[ ]  3. Weekly RSI has room (not already overbought/oversold)
[ ]  4. Daily trend intact (EMA20 > EMA50 for longs)
[ ]  5. Daily MACD momentum turning in direction
[ ]  6. Price at a 4H confluence zone (Fib + Pivot + EMA overlap)
[ ]  7. 15M entry signal fired (Stoch crossover + RSI reset)
[ ] 11–16. CRITICAL PATH — session, structure, trigger, spread, correlation, risk
[ ]    Stop is below structure, minimum 1× ATR distance
[ ]    R:R is at least 2:1 to TP1
```

Verdict is **🟢 GO only when ≥16/18 are checked AND all critical checks (11–16)
pass.** If steps 1–4 conflict with the direction, skip the trade. If 5–7 haven't
fired, wait.

---

## Daily routine — SAST (UTC+2)

You wake at the London Kill Zone open — the highest-probability window of the day.

| Session | UTC | SAST | Priority |
|---------|-----|------|----------|
| London Kill Zone | 07:00–09:00 | 09:00–11:00 | 🟢 Prime |
| NY Kill Zone | 12:00–14:00 | 14:00–16:00 | 🟢 Prime |
| London Close | 15:00–17:00 | 17:00–19:00 | 🟡 Secondary |
| Tokyo | 00:00–03:00 | 02:00–05:00 | 🔴 Avoid |

The routine is keyed to a 17:00–20:00 SAST session (London Close window).
Seven touches, in order — analysis is fully done before the session opens;
the shortlist is frozen at 17:00.

**🌅 Morning Brief — 5 min with coffee, sets bias, no decisions.**
[MTF Matrix](pages/mtf-matrix.py) first (the house-view consensus board —
which pairs are aligned, which are contested) → [Daily
Cockpit](pages/daily_cockpit_tab.py) (regime → rate bias → events → setups,
one screen) → [Market Overview](pages/market-overview.py) →
[News Filter](pages/news-filter.py) (today's landmines, in SAST).

**📋 Pre-Session — 16:30 SAST, build the shortlist (~20 min).**
[Setup Ranker](pages/setup-ranker.py) (primary scan — direction-only score,
Grade A ≥80%) → [Trend Signals](pages/trend-signals.py) (confirm/deny the
Ranker's top picks; ADX is a hard trend gate) →
[Currency Strength](pages/currency-strength.py) + [Correlations](pages/correlations.py)
(kill conflicted/duplicated exposure) → [Confluence Check](pages/confluence-checker.py)
(the final gate — what survives?). **Output:** a frozen shortlist, direction set.

**📅 Weekend (not daily) — positioning & weekly bias.**
[Weekly EMA](pages/weekly-ema.py) / [Weekly Swing](pages/weekly-swing.py) /
[Swing Playbook](pages/swing_playbook_tab.py) →
[Bonds → Gold → DXY](pages/bonds_gold_dxy_app.py) → [COT Positioning](pages/cot_tab.py) /
[COT Signals](pages/cot_signals.py) / [COT Open Interest](pages/cot_open_interest.py) →
[COT Composite Signal](pages/cot_composite_trade_signal.py) for any pair you're
already watching. CFTC data only updates weekly, so re-checking this intraday is
wasted effort — fold whatever bias it gives into that week's macro read from
[Daily Cockpit](pages/daily_cockpit_tab.py). Run [COT Composite Backtest](pages/cot_trade_signal_walk_forward_backtest_harness.py)
occasionally to sanity-check the composite signal still has edge on the instrument you trade.

**⚡ 17:00–20:00 — Session, execution only.** Nothing here generates a new
opinion — if you can re-open the Ranker mid-session you will, and that's how
a flat day becomes a revenge trade. Wait for [15M Fib Entry](pages/15m-fib-entry.py)
to fire on a shortlisted pair, then work [Risk Suite](pages/risk-suite.py)
(Stop Structure tab → R:R Calculator tab → Account Risk tab) → daily-loss
check (2 losses → stop) → **log the trade at entry** in
[Trade Journal](pages/trade-journal.py), not after.

**🌙 19:00 — Post-session review.** [Trade Journal](pages/trade-journal.py)
again — equity curve, win rate vs 66% target, tag outcomes (aligned score,
grade, no-trade days). Monthly, open its **Source Scorecard** tab: it ranks
every signal source by realized expectancy — demote what's negative, trust
what's positive. The 18-point [Daily Checklist](app.py) remains
available as a detailed audit trail if you want the full 18-point gate on a
specific trade.

---

## Discipline rules baked into the system

- **2 losses today** → the engine blocks new entries (daily loss limit, from Postgres).
- **Correlated pairs open** → it warns you (stacked risk, via `CorrelationService`).
- **Not in a kill zone** → the session check shows red.
- **Setup Ranker direction score below 60%** → skip it (Grade A ≥80%, B ≥60% —
  quality-gate criteria like ATR/spread/4H-zone no longer inflate the score;
  a wide spread caps the grade at C regardless of direction score).
- **Same currency on both sides of the shortlist** → the Ranker's net-exposure
  strip flags it as a conflict.
- **High-vol GARCH regime** → the Instrument Predictor caps its own confidence
  at Medium even when its four components otherwise agree unanimously.
- **A page's lens opposes the house view** → the shared strip flags it in red
  (and Market Overview / Daily Cockpit flag conflicted ideas inline) — the
  system won't let one page's opinion masquerade as the canonical read.

One focused trade in the London window, reviewed that evening — done consistently,
that is the path to a 66% win rate.

---

## Appendix — Forex Fundamentals methodology

> **Forex Fundamentals is archived** (see [Archived pages](#archived-pages)) —
> kept in `archive/pages/forex_fundamentals_tab.py`, not deleted. This section
> documents its two engines for reference / in case it's restored.

Documents the two trickier engines on
[Forex Fundamentals](archive/pages/forex_fundamentals_tab.py): the **Priced-In
Analyzer** and the **Risk Sentiment** gauge.

### Part 1 · Priced-In Analyzer

A currency reacts to the **surprise**, not the event. To trade fundamentals you
must know what the market *already expects* so you can judge whether that
expectation is too high or too low. Two engines:

| Engine | Covers | Source | Output |
|---|---|---|---|
| Curve-implied | All majors | FRED (no key) | Direction + rough size of priced policy change |
| Fed funds futures | US only | yfinance `ZQ=F` | Probability of a move at the next meeting |

**Engine 1 — Curve-implied.** A short (~3-month) rate reflects the current policy
stance; a longer rate reflects the average expected policy rate. Their spread is
roughly the net policy change priced:
```
priced_bps   = (long_rate − short_rate) × 100
moves_priced = priced_bps / 25          # in 25bp units
```
Positive = hikes priced; negative = cuts priced; flat = on hold. *Caveat:* the
long rate carries a **term premium**, so this overstates the pure expectation —
reliable for direction and rough magnitude, not a clean probability.

**Engine 2 — Fed funds futures (US).** A 30-day Fed funds future settles on the
average daily effective rate over its month. With one meeting mid-month:
```
avg_implied = 100 − futures_price
rate_after  = (avg_implied × N − current_rate × days_before) / days_after
prob        = (rate_after − current_rate) / hike_size
```
This is the simplified **CME FedWatch** methodology. *Assumptions:* one meeting
per month, effective rate = target before the meeting, settlement = simple
average. Near 0%/100% treat as approximate.

*Worked example:* current 5.00%, meeting day 15 of 30, future 94.8667 →
`avg_implied 5.1333`, `rate_after 5.25`, `prob 1.00` → a 25bp hike is **fully
priced**, so the decision is a non-event and the move comes from the guidance.

**Use it:** read what's priced → form your own view (cross-check Economic
Surprise) → on the day, *the signal is the gap* between outcome and priced-in.

### Part 2 · Risk Sentiment & capital flows

Beyond rates and data, the market has a **mood**: risk-on or risk-off. In
risk-off, capital stampedes into havens (USD, JPY, CHF, Gold) *regardless of
fundamentals* — that flow can override every other tab. Read the regime **first**.

| | Risk-ON (greed) | Risk-OFF (fear) |
|---|---|---|
| Equities / VIX | Up / Low | Down / Spiking |
| Bought | AUD, NZD, CAD, EM, copper | USD, JPY, CHF, gold, Treasuries |
| Sold | JPY, CHF (funders) | AUD, NZD, EM |

JPY and CHF are havens because their low rates make them **funding currencies**;
risk-off unwinds the carry trade and they surge. **AUD/JPY** captures both sides
and is the classic risk barometer.

The gauge turns five signals into z-scores, oriented so **positive = risk-on**:

| Signal | Source | kind | orient |
|---|---|---|---|
| S&P 500 | `^GSPC` | ~1m momentum | +1 |
| VIX | `^VIX` | level | −1 |
| AUD/JPY | `AUDJPY=X` | ~1m momentum | +1 |
| Copper/Gold | `HG=F` / `GC=F` | ~1m momentum | +1 |
| HY credit spread | FRED `BAMLH0A0HYM2` | level | −1 |

```
composite = mean(component z-scores)     # + = risk-on
composite > +0.4 → Risk-On ;  < −0.4 → Risk-Off ;  else Neutral
```

**Use it:** check the regime before anything else. Risk-off → lean with havens
(short AUD/JPY, short NZD/JPY). Risk-on → lean with yield (long AUD/JPY). Neutral
→ the rate-differential / surprise / priced-in signals carry the day. The trap it
avoids: being long a fundamentally strong but risk-sensitive currency right as a
risk-off wave hits.

---

## Notes

- `archive/` holds earlier iterations and is **not** imported by the dashboard.
- Macro fallback data is static; add a FRED key for live values.
- The Trading Ideas view auto-refreshes every 5 minutes via `@st.fragment`.
- Tests live in `tests/`; the DB journal is unit-tested with a mocked connection,
  and `tests/test_pages_smoke.py` runs each page headlessly (gated behind
  `--runslow` because it hits live yfinance). See `CLAUDE.md` for details.

---

## Appendix — `src/data_backbone/` (optional, experimental, not part of the daily app)

A separate, standalone Postgres-backed data layer that **no active page imports
today** — `app.py` and every page in `pages/` read live via `src/db/` and
yfinance/FRED directly (see "Data layer" in `CLAUDE.md`). This appendix
documents it for whoever picks it up next; skip it for day-to-day trading.

> The commands below reference `docker-compose.yml` / `Dockerfile` /
> `.env.example` files that **don't exist in this repo yet** — write them
> first, or run the "without docker" path.

A durable data layer for a Streamlit dashboard: tabs read from a permanent
database and only hit yfinance/FRED when the stored data is missing or stale.

```
Streamlit tab ─▶ data_access.get_ohlcv() ─▶ Postgres ─▶ yfinance/FRED
                                              (durable)   (source of truth)

worker.py (APScheduler) ─▶ fetch on schedule ─▶ upsert Postgres
```

### Files (package `src/data_backbone/`)
- `config.py` — env-driven settings + the worker's watchlists.
- `db.py` — SQLAlchemy tables and `INSERT ... ON CONFLICT DO UPDATE` upserts.
- `data_access.py` — `get_ohlcv()` / `get_fred()`: the db→api read path.
- `worker.py` — scheduled refresh service.
- `app_demo.py` — minimal demo dashboard (the main app stays `app.py` at root).
- `docker-compose.yml` / `Dockerfile` — postgres, worker, app (not yet added — see note above).

### Run it
```bash
cp .env.example .env          # adjust if you like — you'll need to create this file
docker compose up --build     # starts postgres, worker, app
# app on http://localhost:8501 ; worker warms the store then refreshes daily
```

Run locally without docker (needs a local Postgres):
```bash
pip install -r requirements.txt
python -m src.data_backbone.worker          # one process: warms + schedules refresh
streamlit run src/data_backbone/app_demo.py # another process: the demo dashboard
```

### Wire an existing page to it (one line each)
Swap a page's loader for the backbone, e.g.:
```python
from src.data_backbone import data_access as da

@_cache(ttl=3600)
def load_ohlcv(ticker, period="5y"):
    return da.get_ohlcv(ticker, period)   # was: yf.download(...)
```
And for FRED-based reads, replace the direct FRED fetch with `da.get_fred(sid)`.

Keep `@st.cache_data` on the page's own loader too — it's a per-process layer in
front of the database. Order of speed: `st.cache_data` (in-process) → Postgres
(durable) → API.

### Move the trade journal into this backbone's Postgres
The live app's journal already persists to Postgres via `src/db/trade_repository.py`
(see "Data layer" in `CLAUDE.md`) — this is a separate, alternate path if you
consolidate onto `data_backbone` instead:
```python
from src.data_backbone import db
db.init_db()
db.migrate_journal_csv("trade_journal.csv")
```
Then point the journal page's load/save at `db.read_trades()` / `db.save_trade()`.

### Notes
- Add watched tickers/series in `src/data_backbone/config.py` (`WATCH_TICKERS`, `WATCH_FRED`).
- The worker refreshes weekdays at 22:00 UTC (after the US close); change the
  cron in `src/data_backbone/worker.py`.
- Stale thresholds: prices refetch if the latest stored bar is older than
  `STALE_DAYS`; FRED uses a 30-day window. Tune in `config.py` / `data_access.py`.
- Real upserts mean re-running is always safe — no duplicate rows.

### Seed deep history (one-shot)
Backfill the deepest history yfinance gives and see your real coverage:
```bash
python -m src.data_backbone.seed_history                # whole watchlist, period="max" -> Postgres
python -m src.data_backbone.seed_history EURUSD=X --no-save  # just report how far back it goes
python -m src.data_backbone.seed_history --fred         # also backfill FRED series
```
It prints first/last date, bar count, years, and whether each instrument has
real volume (spot FX = none). Daily FX typically reaches ~2003; yfinance intraday
is capped (1m ~7 days, ≤1h ~60 days, hourly ~730 days), so seed daily for deep
backtests and use Dukascopy/HistData if you need years of intraday.