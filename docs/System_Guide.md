# Dashboard Pro — System Guide & Trading Playbook

*A complete walkthrough of every page, what its numbers mean, and how to turn
them into a single, disciplined trade decision.*

---

## 1. What this system is

Dashboard Pro is a **forex & metals day-trading terminal** built on Streamlit.
It is not a black box that spits out "buy" or "sell" — it is a set of 45
purpose-built lenses on the same 22-instrument universe (21 forex pairs +
Gold/Silver/Platinum), organized so that working through them in order
narrows a huge, noisy market down to one or two trades you can actually
defend.

**The core principle is confluence: never trade a single signal.** Every
section below exists to either *confirm* or *veto* the ones before it. A
trade only survives to execution if the weekly trend, the daily trend, the
4-hour zone, the 15-minute trigger, the macro backdrop, and your risk
parameters all agree at the same time.

**The master page is the 18-point Daily Checklist** (`app.py` — what loads
when you run `streamlit run app.py`). Every other page in this guide feeds a
line item on that checklist. A trade is a **GO** only when at least 16 of 18
checks pass *and* every check in the critical path (11–16: session, structure,
trigger, spread, correlation, risk) is clear.

### Who this guide is for

If you are new to the system: read this document top to bottom once, then
keep it open in a second tab while you trade — it is written to match the
sidebar order exactly, section by section. If you already know the system:
jump to Section 7 (the daily routine) or Section 8 (the Instrument Predictor) for what's new.

---

## 2. How the sidebar is organized

44 pages is too many stops for any daily routine to survive. The sidebar
(`src/pages_lib/navigation.py`) is grouped by **when in the day you'd
actually touch a page**, not the order it was built — separating the daily
path (walked in order, ≤8 touches) from reference and research (visited on
demand, never mid-session):

| Section | Cadence | Purpose |
|---------|---------|---------|
| 🌅 Morning Brief | Every session, 5 min | Sets bias, no decisions — regime, snapshot, today's events |
| 📋 Pre-Session | Every session, ~20 min, 16:30 SAST | Build the shortlist: primary scan → confirmer → kill conflicted/duplicated exposure → final gate |
| ⚡ Session | Every session, 17:00–20:00 SAST | Execution only. Nothing here generates a new opinion — the shortlist is frozen at session open |
| 📅 Weekend | Weekly | Weekly-bias tools + the full COT suite |
| 🔬 Research Lab | On demand | The other scanners, deeper single-timeframe analysis, stat-arb research, backtests — visited when building or validating, never during a session |
| 📖 Reference | As needed | System Guide, ABR Toolkit, the 18-point Daily Checklist |

Of the five scanners that produce directional opinions (Setup Ranker, AMD
Scanner, Trend Signals, 20-Day Breakout, MTF Matrix), only **Setup Ranker**
(primary) and **Trend Signals** (confirmer) hold a Pre-Session slot — the
other three live in Research Lab until there's enough journaled history to
know which one actually earns a slot. **Daily Trend / Daily MACD / Market
Structure** do more analysis than the Setup Ranker's checklist booleans
(histogram momentum, EMA100/200 stack + slope, CHoCH/BOS) so they're kept as
full pages in Research Lab, not folded into the Ranker. **Stop Structure, R:R
Calculator, and Account Risk** were one decision spread across three pages —
merged into one **Risk Suite** page (three tabs) in the Session section.

---

## 3. Section 0 — Cockpit

| Page | What it does | What the output means |
|---|---|---|
| **📋 Daily Checklist** (`app.py`) | The 18-point pre-trade gate. Every check you tick here is informed by a page further down this guide. Also where you **save** a setup and **close** a trade. | A 🟢 GO chip means ≥16/18 checked and the critical path (11–16) is clear. Anything less is 🟡 WAIT or 🔴 NO — don't override it. A daily-loss gate (2 losses) blocks new entries regardless of score. |
| **📖 System Guide & Playbook** | This document, in-app, plus a PDF download. | Read once, keep as reference. |
| **🛫 Daily Cockpit** | Pre-market fusion screen: risk regime (risk-on/off), per-pair rate bias, upcoming high-impact events, and any fresh setups already scored — across the full registry, one screen. | If the regime is risk-off, expect havens (USD/JPY/CHF/Gold) bid regardless of what any single pair's chart shows. Use this to set your directional prior *before* you look at any chart. |
| **📊 Market Overview** | Headline KPIs and a price table across FX, metals, and indices. | A fast "what moved overnight" read — not a signal, a snapshot. |

---

## 4. Section 1 — Scan (build the shortlist)

The goal here is not to trade — it's to end up with 1–3 pairs worth a closer
look. Run these, don't act on any of them alone.

| Page | What it computes | Reading the output |
|---|---|---|
| **🎰 Setup Ranker** | The multi-timeframe checklist (`src/core/signals.score_setup`) run for every registry pair, both LONG and SHORT: 10 technical criteria (weekly EMA alignment, weekly RSI room, weekly/daily/4H structure, daily EMA/MACD, ATR expansion, 4H confluence-zone proximity, spread/ATR ratio), plus an 11th **Currency Strength** criterion for FX pairs (20-day base-vs-quote return differential; commodities have no single driving currency and stay on the 10-point scale — the "Min score" filter and email threshold are both applied as a percentage of whichever scale a pair is on, not a raw count). | Auto-refreshes every 5 min. **70%+ = today's candidates; 90%+ fires an email alert.** The grade (A–D) is the score bucketed as a percentage — grade A is ≥80%. |
| **📊 AMD Scanner** | Accumulation / Manipulation / Distribution phases on 1H bars over the current trading week, with a true-range activity proxy substituting for spot FX's meaningless zero-volume data. | Tells you whether a pair is still ranging (Accumulation), about to fake out (Manipulation), or already trending (Distribution) — context for *when* in the AMD cycle a breakout is more or less likely to hold. |
| **📡 Trend Signals** | The 6-condition trend scorer (`TrendSignalEvaluator`): price vs 200 EMA, 50/200 EMA cross, price vs 50 EMA, RSI band, MACD vs signal, ADX > 25. | ≥4/6 conditions = BUY/SELL; ≥5/6 = STRONG BUY/SELL. This is the same evaluator the new Instrument Predictor (Section 8) uses as one of its four votes. |
| **🧭 MTF Matrix** | A grid of the trend read across multiple timeframes at once for every pair. | Look for a pair where several timeframes agree — that's alignment, the opposite of "conflicting timeframes," which is a checklist fail. |
| **🚀 20-Day Breakout** | Donchian-style: is price making a new 20-day high/low? | A breakout candidate list. Cross-reference with AMD (is this leaving Accumulation?) before trusting it. |
| **📦 CME FX Futures** | Real, exchange-reported volume via CME currency futures (`6E=F`, `6B=F`, …) — OBV/CMF that actually mean something, unlike on zero-volume spot pairs. | Volume confirming a move (OBV rising with price) is a stronger read than price alone; volume divergence is a warning. |
| **💱 Predictive Analytics** | A standalone statistical/ML directional read (price vs SMA-style) per pair. | A single model's opinion — treat it as one more vote, not the final word. (Don't confuse this with the Instrument Predictor in Section 8, which combines *several* of this system's own tools rather than being its own model.) |
| **🟡 VWAP-EMA Gold** | A dedicated VWAP + EMA strategy view, specific to Gold. | Gold-specific execution context if Gold is your candidate. |
| **🧱 ABR Toolkit** | Swing detection → BoS/CHoCH (with ATR strength) → order blocks (mitigation + age) → auto trendlines + break detection → MTF EMA bias → a 0–100 quality score → a full trade plan (entry/SL/TP1-3, USD lot sizing) for Gold, Silver, the majors, WTI, and BTC. Tries a running Deriv/MT5 terminal first for exact broker candles (avoids the futures/spot basis offset yfinance carries on GC=F/SI=F), falling back to yfinance when MT5 isn't available. | A structure-first second read on a Scan candidate — order blocks and trendline breaks are additional confluence, not a replacement for the top-down pass below. |

---

## 5. Section 2 — Synthesize (NEW)

| Page | What it does |
|---|---|
| **🔮 Instrument Predictor** | Combines **Setup Score** (Scan), **Trend Signal** (Scan), **Currency Strength** differential, and **COT Composite** (Filter/Research, majors + gold only) into one weighted composite: a STRONG_SELL…STRONG_BUY label, a confidence level, and an agreement percentage. |

This is new — see **Section 9** below for the full methodology, worked example, and
how to read its confidence/agreement numbers. In one line: run it on your
Scan-stage shortlist as a fast second opinion before sinking 20 minutes into
the macro/weekly/daily/4H/15M pass on a candidate that doesn't hold up once
you look at it from four angles instead of one.

---

## 6. Section 3 — Filter the day (daily macro & risk backdrop)

These are the tools that can **veto** a Scan-stage candidate today, even if
its technical score is high. Checked every session, unlike Section 10.

| Page | What it shows | How to use it |
|---|---|---|
| **💵 DXY vs Gold** | The classic dollar/gold inverse relationship, tested against live data. | If DXY is breaking out while Gold isn't breaking down (or vice versa), the "classic" relationship is temporarily broken — a warning to reduce confidence in Gold setups until it re-aligns. |
| **💪 Currency Strength** | Ranks all 9 registry currencies strong→weak from their average % return (per window) across every pair they appear in; surfaces the strongest-vs-weakest pair as the cleanest trend combination available. | The **spread** between strongest and weakest is the signal — a wide spread means a clean trend combo exists; a narrow spread means no currency is dominating today. |
| **📰 News Filter** | Red-folder (high-impact) events in the next hours, entry-window conflict check, session danger windows. | If a high-impact event falls within your buffer window (default ±60 min) of your planned entry, **wait or skip** — this is one of the 18 checklist points. |
| **🔗 Correlations** | Warns when an open trade shares directional exposure with a correlated pair (e.g. long EUR/USD + long GBP/USD = doubled USD risk), via `CorrelationService.check_exposure()` and the `CORR_GROUPS` registry table. | Don't stack correlated risk without knowing it — this literally computes your true net exposure, not just position count. |

---

## 7. Sections 4–9 — the top-down technical pass

This is the classic weekly→daily→4H→15M cascade. Once Scan + Synthesize +
Filter have given you a candidate and a directional bias, this is where you
confirm it survives every timeframe on the way down to your actual entry.

### 4 · Weekly bias
| Page | Rule |
|---|---|
| **📉 Weekly EMA** | EMA20 above EMA50 with price above both = weekly uptrend (and the mirror for downtrends). **If the weekly trend is up, only look for longs on lower timeframes. Never fight the weekly.** |
| **🔄 Weekly Swing** | Weekly pivot swing setups plus daily confirmation. |
| **📔 Swing Playbook** | Weekly pre-flight checklist per instrument: last price, COT positioning percentile/z-score (majors + gold), gold's real-yield disconnect z-score, nearest swing-pivot supply/demand levels, GARCH/EWMA-scaled position sizing, and a hand-typed weekly thesis + invalidation. Red row = no thesis written yet; yellow row = an event gate is blocking entry. Audit-only — logged to `tool_usage_log`, not a `trade_setups` signal, since the bias is your own written call, not a computed read. |

### 5 · Daily confirm
| Page | Rule |
|---|---|
| **📈 Daily Trend** | Daily EMA20>50 should match the weekly bias. A bearish daily cross *under* a bullish weekly is a pullback (buying opportunity), not a reversal. |
| **📊 Daily MACD** | Histogram turning from negative toward zero = exhausting sell-side momentum — an early heads-up, not a trigger by itself. |
| **🏗️ Market Structure** | HH/HL (bullish) or LH/LL (bearish) — is price structure actually intact, independent of any indicator? |

### 6 · 4H zone
| Page | Rule |
|---|---|
| **🎯 4H Confluence Zone** | The execution zone = Fibonacci retracement (38.2/50/61.8%) + Pivot S/R + EMA20, all overlapping at one price. |
| **🔀 2/3 Confluence Check** | Quick gate: do at least 2 of those 3 confluences line up right now? If not, you're not at the zone yet — wait. |

### 7 · 15M trigger
| Page | Rule |
|---|---|
| **⚡ 15M Fib Entry** | Wait for **at least two of three**, within 1–2 candles: Stochastic cross below 25 (longs) / above 75 (shorts), 15M RSI reset off 40/60, a Bollinger-band touch curling back in. **Do not enter** if price is still slicing through the zone, the daily shows a conflicting flag, or the spread is abnormally wide ahead of news. |

### 8 · Risk & execute — 🛡️ Risk Suite (one page, three tabs)
Stop Structure, R:R Calculator, and Account Risk were one decision spread
across three separate pages that each re-asked for instrument/balance/risk —
merged into a single **Risk Suite** page sharing those inputs once, with each
tool's original math preserved exactly.

| Tab | Formula |
|---|---|
| **🛡️ Stop Structure** | Stop must sit behind a real structure level, minimum 1× ATR14 distance. SL = 1.5 × ATR14 in pips. |
| **⚖️ R:R Calculator** | TP1 = 2R, TP2 = 3R. **Only take trades with ≥2:1 R:R to TP1** (a 40% win rate at 2.5:1 is profitable; a 60% win rate at 1:1 barely breaks even after costs). |
| **💵 Account Risk** | Position size = (Account × Risk%) ÷ (SL pips × pip value). Risk 1–2% per trade. |

### 9 · Review
| Page | What it tracks |
|---|---|
| **📓 Trade Journal** | Equity curve, win rate vs a 66% target, MT4 statement import. Also where you close a trade (updates the Checklist's daily-loss counter). Every open signal is checked against live price for a crossed stop level — 🔴 INVALIDATED is a visibility badge only; it never changes Outcome/Close/Open, which still require an actual close. |

---

## 8. Section 10 — Weekly & Research Lab

Everything here runs on weekly-cadence data (CFTC reports update Fridays,
covering positions as of the prior Tuesday) or is an occasional research
tool. Check this section **weekly** (Monday morning is a good habit) or
whenever you're doing deeper research — not every session.

| Page | What it is |
|---|---|
| **🏛️ COT Positioning** / **🧭 COT Signals** / **🧮 COT Open Interest** / **🥇 Gold COT** / **🛢️ Oil COT** | Weekly CFTC institutional/speculative positioning: raw net position, extreme-percentile/z-score reads, price/positioning divergence, and open-interest context (is a move fresh conviction or an unwind?). **A crowdedness read, not a trade trigger on its own.** |
| **🏦 Bonds → Gold → DXY** | The Treasury-yield ⇄ gold ⇄ dollar seesaw, explained then tested against live data. Educational cross-asset context. |
| **🧩 COT Composite Signal** | Combines the three COT pages above into one scored STRONG_BUY…STRONG_SELL read plus a "collapse watch" flag (a violent single-week unwind from a crowded position). **Rules-based heuristic, not a validated edge** — backtest it (next page) per instrument before trusting it. This is also one of the four votes inside the Instrument Predictor (Section 9). |
| **🧪 COT Composite Backtest** | No-lookahead walk-forward validation of the page above — win rate / avg return by signal state and horizon, plus a naive equity curve. Saves nothing; it's a research tool. |
| **📅 Busy-Week Anatomy** | Historical study of price behavior during high-event-density weeks. |
| **🔌 Disconnect Monitor** | Tests "this divergence should close" theses (e.g. real yields vs gold) via a rolling-regression residual z-score, then an event study of what actually happened after past disconnects at this threshold. |
| **🌙 Overnight Drift** | Studies the overnight-session return pattern for index futures. |
| **⏱️ Optimal Holding Period** | Backtests how many days a breakout-style entry should be held before mean-reverting/decaying, per pair. |
| **🧮 Quant Models Lab** | Six statistical models for pairs research: Engle-Granger cointegration test, Kalman-filter dynamic hedge ratio, Ornstein-Uhlenbeck mean-reversion fit (half-life + z-bands), GARCH(1,1) volatility regime + vol-targeted position sizing, UIP/carry deviation, and a GBM Monte-Carlo null test for judging whether *any* backtest number beats pure luck. See Section 9.4 for how this feeds the Instrument Predictor's vol-regime confidence modifier. |
| **🔭 Forecast** | A GARCH(1,1)-t volatility cone (1 week / 1 month / 1 quarter) around a random-walk center — deliberately no point forecast, since beating a random walk at these horizons is hard (Meese-Rogoff). Alongside it, a transparent driver score (trend, 200d MA, momentum, 20-day breakout, and the ABR Toolkit's structure bias when available) and a plain-language narrative (template by default, polished by the Claude API when `ANTHROPIC_API_KEY` is set). Every forecast is journaled and later scored against what actually happened — hit rate inside the 68%/95% bands, and whether the directional call was right. | The self-scoring history is the point: it tells you honestly, over time, whether this page's driver score has any real edge for a given instrument, rather than asking you to trust it on faith. |
| **😲 Surprise Awareness** | Three tools sharing one economic-calendar feed: (1) a **surprise index** — z-scored actual-vs-forecast per data release (Citi-ESI style), covering every registry currency, not just USD; (2) an **event-proximity gate** per instrument — suppresses/flags signals when a high-impact release for its currencies is imminent or just happened; (3) a **gold~oil rolling correlation** to catch when "escalation = buy gold" has inverted (it did, Feb 2026). Telegram-alerts on qualifying surprises if `[telegram]` is configured in secrets.toml. | Check the gate for your candidate instrument before triggering an entry — a red-folder release inside the window means stand aside, the geopolitical/data premium unwinds in hours, not weeks. |

---

## 9. Deep dive — how to actually form a prediction for an instrument

This is the "how do I turn all of this into one answer" chapter.

### 9.1 The manual version (what the top-down pass already does)

Walking sections 0–8 top to bottom **is** the prediction process: each stage
either confirms or vetoes the ones before it, and what survives to the 15M
trigger is, by construction, the system's best read on that instrument right
now. The **direction** comes from whichever side (long/short) keeps passing
every gate; the **confidence** is qualitative — how many of the 18 checklist
points are ticked, and whether the critical path (11–16) is clean.

### 9.2 The composite version — 🔮 Instrument Predictor

The new Instrument Predictor (`src/services/prediction_service.py` +
`src/pages_lib/instrument_predictor.py`) automates a *narrower* version of
the same idea: instead of the full 18-point manual pass, it pulls four
already-existing scorers for one instrument and combines them into a single
number.

**The four components:**

| Component | Weight | Source | Normalized how |
|---|---|---|---|
| Setup Score | 30% | `score_setup()` — the same scorer Setup Ranker uses, run for both LONG and SHORT, best direction kept. Deliberately called *without* a Currency Strength differential here (unlike Setup Ranker/Fib Entry) so it stays a pure-technical 10-point read and doesn't double-count against this page's own separate Currency Strength component below | `± (score / max_score)` |
| Trend Signal | 25% | `TrendSignalEvaluator` — the same 6-condition scorer Trend Signals uses | STRONG_BUY=+1.0, BUY=+0.5, NEUTRAL=0, SELL=-0.5, STRONG_SELL=-1.0 |
| Currency Strength | 25% | The same 20-day base/quote strength differential Currency Strength ranks by (forex pairs only — commodities have no single driving currency, so this component is simply unavailable for XAU/XAG/XPT and its weight is redistributed) | `clip(diff_% / 2.0, -1, 1)` |
| COT Composite | 20% | The same `generate_trade_signal()` logic COT Composite Signal uses, run on the relevant currency's percentile + z-score (majors + gold only — no CFTC series exists for a pair with no COT-tracked currency, so it's unavailable elsewhere) | `± (score / 3.0)`, sign-flipped for USD/XXX quote-leg currencies |

**Why a missing component doesn't just silently pull the score toward zero:**
each component carries a weight of `0` when it can't be computed (e.g. COT
for a cross with no CFTC series), and `aggregate()` renormalizes across only
the *available* weights. Two components agreeing strongly produce the same
composite whether or not a third, unavailable one is "missing" — tested
explicitly in `tests/test_prediction_service.py`.

**Reading the output:**

- **Composite score** (−1 to +1) → label: ≥0.5 STRONG_BUY, ≥0.15 BUY,
  ≤−0.5 STRONG_SELL, ≤−0.15 SELL, else NEUTRAL.
- **Agreement** — the fraction of components (that could be computed) that
  point the same way the composite does. 100% agreement with a strong
  composite is the strongest possible read this tool can give you.
- **Confidence** (Low/Medium/High) — derived from agreement *and* the
  composite's magnitude together, not either alone. NEUTRAL is always Low
  confidence, by definition.
- **Vol regime** (from a GARCH(1,1) fit on the instrument's own daily
  returns, via Quant Models Lab's engine) is a **modifier, not a vote** — a
  "high" vol regime caps confidence at Medium even if the four components
  otherwise agree unanimously, because directional reads are inherently less
  reliable in a choppy, high-volatility regime. This is the one place Quant
  Models Lab's machinery feeds directly into another page.

**Worked example** (an actual run captured while building this guide):
EUR/USD came back **composite −0.51, label STRONG_SELL, confidence High,
agreement 100%** — Setup Score had scored a SHORT setup 6/10 (grade B),
and the other three components all leaned bearish too. Because the label
wasn't NEUTRAL, the page persisted this as a `bias=Short` row to
`trade_setups` (source `instrument_predictor`) exactly the way Setup Ranker
or Currency Strength would, so it shows up in the Trade Journal alongside
every other page's signals.

**The caveat, stated plainly (same one COT Composite carries):** this is a
**heuristic combination of the dashboard's own existing tools, not a new,
independently validated model.** A "High confidence, STRONG_BUY" read means
four of this system's own scorers agree with each other right now — it does
not mean those four scorers are collectively *correct* more often than any
one of them alone. Nothing here has been walk-forward backtested the way
COT Composite Signal has been (via COT Composite Backtest). Treat it as a
fast second opinion that either strengthens your conviction on a Scan-stage
candidate or flags that you should look closer before spending 20 minutes on
the full top-down pass — not as a replacement for that pass, and never as
the sole reason to enter a trade.

### 9.3 Putting Section 9.1 and Section 9.2 together

In practice: run Setup Ranker → take your 7+/10 candidates → run Instrument
Predictor on each → **keep only the candidates where the Predictor's
direction agrees with the Setup Ranker's own direction** (if they disagree,
that's exactly the "second opinion caught something" case the Predictor is
for — dig into *why* before proceeding) → then continue the full weekly →
daily → 4H → 15M pass from Section 7 on whatever survives, exactly as before.

### 9.4 What Quant Models Lab adds for a specific pair thesis

If your instrument doesn't fit the standard playbook — e.g. you suspect gold
and real yields have de-coupled and want to know if that gap actually closes
historically, or you want to check whether EUR/USD's recent breakout return
beats what pure random-walk luck would produce — that's what Quant Models
Lab and Disconnect Monitor are for, not the Instrument Predictor. They answer
narrower statistical questions (is this pair actually cointegrated with its
hypothesized driver? does the volatility regime justify a wider stop?) that
feed *into* your read on an instrument without being a single directional
signal themselves.

---

## 10. Domain conventions (the vocabulary this whole system shares)

- **Prices**: 5 decimals when `|price| < 100` (FX pairs), 2–3 decimals for
  JPY crosses, metals, and indices.
- **Risk model**: SL = 1.5 × ATR14 (pips). TP1 = 2R, TP2 = 3R. Lot size =
  risk_amount ÷ sl_pips ÷ pip_value. R-multiple = pips gained ÷ SL pips.
- **Sessions (UTC)**: London Kill Zone 07–09 🟢, NY Kill Zone 12–14 🟢,
  London Close 15–17 🟡, Tokyo 00–03 (everything else) 🔴 Dead Zone. Only the
  green windows are prime entry windows.
- **Metals use forex symbols**: Gold = XAU/USD, Silver = XAG/USD, Platinum
  = XPT/USD, WTI crude = WTI/USD (Yahoo tickers stay `GC=F`/`SI=F`/`PL=F`/`CL=F`).
- **Correlation stacking**: before adding exposure, check whether an open
  trade in the same `CORR_GROUPS` shares direction with a new candidate
  (e.g. long EUR/USD + long GBP/USD = doubled USD risk).
- **The daily-loss gate**: 2 losses in a day blocks new entries, enforced
  from the Postgres journal, regardless of how good the next setup looks.

---

## 11. Discipline rules baked into the system

- **2 losses today** → new entries are blocked (from the journal, not a
  suggestion).
- **Correlated pairs open** → you get an explicit warning before stacking risk.
- **Not in a kill zone** → the session check shows red; the checklist can't
  reach GO.
- **Setup ranks below 7/10** → skip it; don't argue with the scanner.
- **High-vol GARCH regime** → the Instrument Predictor caps its own
  confidence rather than letting you assume a High read that the market's
  actual volatility doesn't support.

One focused trade in the London window, reviewed that evening — done
consistently, that is the path this system is built around.

---

## 12. Appendix — where things live in code

| Concept | Lives in |
|---|---|
| Instrument universe, pip sizes, correlation groups | `src/instruments/registry.py` |
| Sidebar order (this guide's Section 2 table) | `src/pages_lib/navigation.py` |
| Setup Score (10 technical + optional Currency Strength) | `src/core/signals.py::score_setup` |
| 6-condition Trend Signal | `src/indicators/trend_signal.py::TrendSignalEvaluator` |
| Currency Strength ranking | `src/pages_lib/currency_strength.py` |
| COT Composite scoring | `pages/cot_composite_trade_signal.py::generate_trade_signal` |
| Six quant models (OU/GARCH/Kalman/Engle-Granger/UIP/GBM) | `src/core/quant_models.py` |
| Instrument Predictor aggregator (pure logic, unit-tested) | `src/services/prediction_service.py` |
| Instrument Predictor page | `src/pages_lib/instrument_predictor.py` |
| Trade persistence (`trade_setups` table) | `src/services/signal_store.py` → `src/db/trade_repository.py` |
| Non-signal audit trail (`tool_usage_log` table) | `src/services/tool_log.py` |

This guide mirrors `src/pages_lib/navigation.py` exactly — if the two ever
disagree, the code wins. Regenerate the PDF from this file after any
navigation change (see `docs/generate_guide_pdf.py`).
