# Dashboard-Pro

A professional **forex & metals day-trading terminal** built on Streamlit. It
walks you top-down through a single trading decision — scan the market, filter
the day, establish bias, confirm, find the zone, wait for the trigger, size the
risk, execute, and review — across **31 active pages** (plus a signal-lab pair
for validating the COT composite read) that all share one analysis engine and
one Bloomberg-style terminal theme. (Trimmed from 52 for a leaner daily forex
workflow — the retired pages live in `archive/pages/`, not deleted; see
[Archived pages](#archived-pages) below.)

It trades the 21-pair forex universe **plus metals** — Gold (XAU/USD), Silver
(XAG/USD) and Platinum (XPT/USD) — from one instrument registry, so every page
scans the identical universe with the identical tickers.

> The master page is the **18-point Daily Checklist** (`app.py`). Every other
> page feeds a check on it. A trade is a **GO** only when ≥16/18 checks pass and
> all critical checks (11–16) are clear.

---

## Project structure

```
Dashboard-Pro/
├── app.py                  # Entry point — the 18-point Daily Checklist (master page)
├── pages/                  # 31 active workflow pages (auto-register as Streamlit multipage)
├── src/
│   ├── core/               # analyzer, signals, config — the shared analysis engine
│   ├── indicators/         # EMA/RSI/MACD/ADX/ATR + the 6-condition trend scorer
│   ├── instruments/        # registry.py — the single source of truth for instruments
│   ├── services/           # ATR, correlation, session, risk, forecast, MT4 import …
│   ├── pages_lib/          # BloombergPage framework + navigation (the workflow order)
│   ├── db/                 # Postgres journal (trade_repository) + cache/pool layer
│   └── ui/                 # terminal theme & HTML components
├── tests/                  # pytest suite (logic + DB layer ~92% cov; page smoke tests)
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

## How the system works — the 8-step workflow (+ a weekly signal lab)

The pages are grouped and numbered the way you actually trade a session. Work
them top to bottom; most days you touch a handful and take one clean trade.
The one exception is section 9 (Signal lab) — it's a weekly check, not part of
the daily top-to-bottom pass. This table mirrors
[`src/pages_lib/navigation.py`](src/pages_lib/navigation.py) exactly — that
file is the single source of truth; if the two ever disagree, the code wins.

### 0 · Start here
| # | Page | What it's for |
|---|------|---------------|
| 00 | [📋 Daily Checklist](app.py) | **The cockpit.** The 18-point pre-trade gate; saves the trade and the journal. Everything else feeds this. |
| 01 | [🛫 Daily Cockpit](pages/daily_cockpit_tab.py) | Pre-market routine on one screen — risk regime → rate bias → events → fresh setups, with the parts that agree flagged. Covers the full 21-pair + metals registry. |
| 02 | [📊 Market Overview](pages/market-overview.py) | Morning snapshot — headline KPIs + price table across FX, metals, indices. |

### 1 · Scan — build the shortlist
| # | Page | What it's for |
|---|------|---------------|
| 03 | [🎰 Setup Ranker](pages/setup-ranker.py) | The 10-point multi-timeframe scorer. Auto-refreshes every 5 min; run **Both** directions; pairs scoring 7+/10 are today's candidates, 9-10 fire an email alert. |
| 04 | [📊 AMD Scanner](pages/amd-scanner.py) | Accumulation / Manipulation / Distribution scanner (1H × 1 month). |
| 05 | [📡 Trend Signals](pages/trend-signals.py) | The 6-condition trend-following scan (50/200 EMA, RSI, MACD, ADX). |
| 06 | [🚀 20-Day Breakout](pages/twenty_day_breakout_tab.py) | Donchian-style 20-day breakout candidates. |
| 07 | [📦 CME FX Futures](pages/cme-futures-volume.py) | Real, exchange-reported volume for FX via CME currency futures (6E=F, 6B=F, …) — OBV/CMF that actually mean something, unlike on zero-volume spot pairs. |
| 08 | [💱 Predictive Analytics](pages/predictive.py) | Statistical/ML directional read. |
| 10 | [🟡 VWAP-EMA Gold](pages/vwap-ema-gold.py) | A dedicated VWAP+EMA strategy view for Gold. |

### 2 · Filter the day — macro & risk backdrop
| # | Page | What it's for |
|---|------|---------------|
| 13 | [💵 DXY vs Gold](pages/dxy-gold.py) | Dollar vs Gold inverse — the cross-asset confirmation. |
| 14 | [💪 Currency Strength](pages/currency-strength.py) | Ranks the 9 registry currencies strong → weak from average pair returns; surfaces the strongest-vs-weakest pair as the highest-probability trend trade. |
| 15 | [🏦 Bonds → Gold → DXY](pages/bonds_gold_dxy_app.py) | Treasury-yield ⇄ gold ⇄ dollar seesaw, explained then tested against live data — cross-asset context, educational. |
| 16 | [📰 News Filter](pages/news-filter.py) | Red-folder events in the next hours — wait or skip. |
| 17 | [🏛️ COT Positioning](pages/cot_tab.py) | Weekly CFTC institutional/speculative positioning for majors, gold, DXY. Updates weekly (Friday, for the week ending the prior Tuesday) — a crowdedness read, not a trade trigger on its own. |
| 18 | [🧭 COT Signals](pages/cot_signals.py) | Extreme-positioning + price/positioning divergence on top of 17, plus a composite squeeze-watch flag. |
| 19 | [🔗 Correlations](pages/correlations.py) | Stacked-exposure check before adding correlated risk. |
| 20 | [🧮 COT Open Interest](pages/cot_open_interest.py) | Position-change + open-interest-change context — is a move fresh conviction or an unwind? |

> **Cadence note:** 17/18/20 (and 35/36 below) run off weekly CFTC data — check
> them once a week (Monday morning is a good habit), not every session.

### 3 · Weekly bias
| # | Page | What it's for |
|---|------|---------------|
| 21 | [📉 Weekly EMA](pages/weekly-ema.py) | Weekly 20/50 EMA alignment = the macro trend. |
| 23 | [🔄 Weekly Swing](pages/weekly-swing.py) | Weekly pivot swing setups + daily confirmation. |

### 4 · Daily confirm
| # | Page | What it's for |
|---|------|---------------|
| 24 | [📈 Daily Trend](pages/daily-trend.py) | Daily EMA20>50 confirms the weekly is intact. |
| 25 | [📊 Daily MACD](pages/daily-macd.py) | Daily MACD momentum — building or fading. |
| 26 | [🏗️ Market Structure](pages/market-structure.py) | HH/HL or LH/LL — is structure intact? |

### 5 · 4H zone
| # | Page | What it's for |
|---|------|---------------|
| 27 | [🎯 4H Confluence Zone](pages/4H-confluence-zone.py) | Fib + Pivot + EMA20 overlap = the execution zone. |
| 28 | [🔀 2/3 Confluence Check](pages/confluence-checker.py) | Quick gate: do at least 2 of 3 confluences line up? |

### 6 · 15M trigger
| # | Page | What it's for |
|---|------|---------------|
| 29 | [⚡ 15M Fib Entry](pages/15m-fib-entry.py) | Retrace into the 0.382–0.618 golden zone + a confirming candle. Optional email alerts. |

### 7 · Risk & execute
| # | Page | What it's for |
|---|------|---------------|
| 31 | [🛡️ Stop Structure](pages/stop-structure.py) | Is the stop behind a real structure level (≥1× ATR)? |
| 32 | [⚖️ R:R Calculator](pages/rr-calculator.py) | Is R:R ≥ 2:1 to TP1? (the engine filters at 1.5:1). |
| 33 | [💵 Account Risk](pages/account-risk.py) | Position size = (Account × Risk%) ÷ (SL pips × pip value). |

### 8 · Review
| # | Page | What it's for |
|---|------|---------------|
| 34 | [📓 Trade Journal](pages/trade-journal.py) | Equity curve, win rate vs 66% target, MT4 statement import. |

### 9 · Signal lab (not a sequential step)
A combined signal layered on top of 17/18/20, plus its own validation tool.
Numbered after the 8-step workflow rather than squeezed into "Filter the day"
so the sidebar's numbers stay strictly increasing top to bottom — check it
weekly alongside the COT pages above, not every session.

| # | Page | What it's for |
|---|------|---------------|
| 35 | [🧩 COT Composite Signal](pages/cot_composite_trade_signal.py) | Combines 17/18/20 into one scored STRONG_BUY…STRONG_SELL read plus a collapse-watch flag; auto-saves actionable, non-neutral, higher-confidence reads to the journal. **Rules-based heuristic, not a validated edge** — backtest it (36) per instrument before trusting it. |
| 36 | [🧪 COT Composite Backtest](pages/cot_trade_signal_walk_forward_backtest_harness.py) | No-lookahead walk-forward validation of 35 — win rate / avg return by signal state and horizon, plus a naive equity curve. A research tool; it saves nothing to the journal. |

---

## Archived pages

21 pages were retired from the daily workflow (moved to `archive/pages/`, not
deleted — full git history intact, code still imports/compiles) to keep the
system focused on a lean, forex-daily-execution path. Two reasons:

- **Not forex-usable**: Smart Money and Volume Profile both need real
  exchange volume; spot FX (`=X` tickers) reports zero on yfinance, so
  volume-based indicators on them are meaningless. [07 · CME FX
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

Archived: Trading Ideas, MTF Matrix, Technical Chart, Pivots & Fibonacci,
Volume Profile, FRED Macro Grid, Smart Money, Forecast Lab, Macro Bias, Forex
Fundamentals, Market Regime, Risk Reversals, Event Impact, Seasonality, ATR
Volatility, Weekly RSI, Double Zeros, Backtest Lab, Trading Lab, Reports,
System Logs. To bring one back into the daily nav, move its file back to
`pages/` and add its `NavEntry` back to `src/pages_lib/navigation.py`.

---

## Top-down trading framework

The core principle is **confluence** — never trade a single signal. Each step
narrows the universe until only high-probability setups remain.

### Step 1 — Macro backdrop · [01 Daily Cockpit](pages/daily_cockpit_tab.py)
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

**Weekly add-on — positioning context:** [17 COT Positioning](pages/cot_tab.py) /
[18 COT Signals](pages/cot_signals.py) / [20 COT Open Interest](pages/cot_open_interest.py) /
[35 COT Composite Signal](pages/cot_composite_trade_signal.py) show what leveraged
funds are actually positioned for, and whether that positioning is crowded, diverging
from price, or actively unwinding. It's a crowdedness/contrarian read on top of the
fundamental bias above, not a replacement for it — and it only refreshes weekly.

### Step 2 — Weekly trend · [21 Weekly EMA](pages/weekly-ema.py) · [23 Weekly Swing](pages/weekly-swing.py)
- **EMA20 vs EMA50** — EMA20 above EMA50 with price above both = weekly uptrend.
- **Price vs weekly pivot** — sustained above PP = buyers in control.
- **RSI** — above 50 confirms bullish momentum; above 70, be cautious adding longs.

> **Rule:** If the weekly trend is up, only look for longs on lower timeframes. Never fight the weekly.

### Step 3 — Daily confirmation · [24 Daily Trend](pages/daily-trend.py) · [25 Daily MACD](pages/daily-macd.py) · [26 Market Structure](pages/market-structure.py)
- **EMA alignment** — daily EMA20>50 should match the weekly bias. A bearish daily cross under a bullish weekly = a *pullback* (a buying opportunity), not a reversal.
- **RSI 40–60 on a pullback** gives room to run; >70 = extended, wait for a reset.
- **MACD histogram** turning from negative toward zero signals exhausting sell-side momentum.
- **ATR** sets your minimum stop distance — anything tighter than 1× ATR is noise.

### Step 4 — 4H zone · [27 4H Confluence Zone](pages/4H-confluence-zone.py) · [28 Confluence Check](pages/confluence-checker.py)
The **confluence zone** = Fibonacci level (38.2 / 50 / 61.8%) + Pivot S/R + EMA20,
all overlapping at one price. When two or three align there, that is where you execute.

### Step 5 — Entry trigger · [29 15M Fib Entry](pages/15m-fib-entry.py)
Wait for at least **two of three** within 1–2 candles: Stochastic cross below 25
(longs), 15M RSI reset off 40, lower Bollinger-band touch curling back in.
**Do not enter** if price is still slicing through the zone, the daily shows
⚠️ Conflicting, or the spread is abnormally wide ahead of news.

### Step 6 — Risk definition · [31 Stop Structure](pages/stop-structure.py) · [32 R:R Calculator](pages/rr-calculator.py) · [33 Account Risk](pages/account-risk.py)
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

**☕ 08:30–09:00 — Pre-market prep.** [01 Daily Cockpit](pages/daily_cockpit_tab.py)
(regime + rate bias + fresh setups, one screen) → [03 Setup Ranker](pages/setup-ranker.py)
(Both modes; 7+/10 = candidates) → [16 News Filter](pages/news-filter.py) →
[19 Correlations](pages/correlations.py) → [21 Weekly EMA](pages/weekly-ema.py) /
[23 Weekly Swing](pages/weekly-swing.py) → [24 Daily Trend](pages/daily-trend.py) /
[25 Daily MACD](pages/daily-macd.py).
**Output:** 1–2 pairs confirmed by score + macro, with direction set.

**📅 Monday morning (weekly, not daily) — positioning & cross-asset context.**
[15 Bonds → Gold → DXY](pages/bonds_gold_dxy_app.py) → [17 COT Positioning](pages/cot_tab.py) /
[18 COT Signals](pages/cot_signals.py) / [20 COT Open Interest](pages/cot_open_interest.py) →
[35 COT Composite Signal](pages/cot_composite_trade_signal.py) for any pair you're
already watching. CFTC data only updates weekly, so re-checking this intraday is
wasted effort — fold whatever bias it gives into that week's macro read from
[01 Daily Cockpit](pages/daily_cockpit_tab.py). Run [36 COT Composite Backtest](pages/cot_trade_signal_walk_forward_backtest_harness.py)
occasionally (not daily) to sanity-check 35 still has edge on the instrument you trade.

**🔎 09:00–09:15 — Confirm the zone.**
[27 4H Confluence Zone](pages/4H-confluence-zone.py) →
[28 Confluence Check](pages/confluence-checker.py).

**⚡ 09:15–11:00 — Watch for the entry (London KZ).** Open
[00 Daily Checklist](app.py) on your pair; tick checks from your pre-market work;
watch the MTF alignment strip and the correlation-exposure warning. Wait for
[29 15M Fib Entry](pages/15m-fib-entry.py) to fire. **Take it only at 16/18+ with
checks 11–16 ticked and the chip showing 🟢 GO.**

**🛡️ Before you click.** [31 Stop Structure](pages/stop-structure.py) →
[32 R:R Calculator](pages/rr-calculator.py) → daily-loss check (2 losses → stop).
Then **save** the trade in [00 Daily Checklist → Save Trade Setup](app.py).

**🌇 14:00–16:00 — NY KZ (second chance).** If London gave no GO, re-run the zone
pages for the NY open — price often re-tests the London level.

**📓 19:00 — End-of-day review.** [34 Trade Journal](pages/trade-journal.py)
(equity up? win rate toward 66%?) → [26 Market Structure](pages/market-structure.py)
→ close open trades in [00 Daily Checklist → Close Trade](app.py).

---

## Discipline rules baked into the system

- **2 losses today** → the engine blocks new entries (daily loss limit, from Postgres).
- **Correlated pairs open** → it warns you (stacked risk, via `CorrelationService`).
- **Not in a kill zone** → the session check shows red.
- **Setup ranks below 7/10** → skip it.

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