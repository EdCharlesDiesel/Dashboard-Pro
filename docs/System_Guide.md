# Dashboard Pro — System Guide & Trading Playbook

*A complete walkthrough of every page, what its numbers mean, and how to turn
them into a single, disciplined trade decision.*

---

## 1. What this system is

Dashboard Pro is a **forex & metals swing-trading terminal** built on Streamlit.

> **Swing, not day trading — and the code says so.** Measured 2026-08-15:
> every persisted signal carries a 5- or 20-day horizon and none is shorter
> than a day; the canonical spine holds only daily (`300d`) and weekly (`2y`)
> windows, and `hourly_ohlc` has no callers at all; stops are 1.5x the
> *daily* ATR; and the dedupe key is `pair + bias + FX-session-day`, which
> makes a second same-day entry structurally impossible. The 15-minute pages
> time the entry, they do not pick the trade - `fib_entry` reads 15m bars and
> still stamps a 5-day horizon. The 5-minute sweep is a refresh cadence, not
> a signal frequency.
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

**MTF Matrix leads the Morning Brief** (moved from Research Lab, 2026-07): it
is now the **house-view consensus board**, not another scanner opinion — see
the subsection below. Of the remaining scanners that produce directional
opinions (Setup Ranker, AMD Scanner, Trend Signals, 20-Day Breakout), only
**Setup Ranker** (primary) and **Trend Signals** (confirmer) hold a
Pre-Session slot — the others live in Research Lab until the Trade Journal's
Source Scorecard shows which one actually earns a slot. **Daily Trend / Daily
MACD / Market Structure** do more analysis than the Setup Ranker's checklist
booleans (histogram momentum, EMA100/200 stack + slope, CHoCH/BOS) so they're
kept as full pages in Research Lab, not folded into the Ranker. **Stop
Structure, R:R Calculator, and Account Risk** were one decision spread across
three pages — merged into one **Risk Suite** page (three tabs) in the Session
section.

### The house view — one canonical direction (2026-07)

Pages used to each define "trend" their own way on their own private data
windows, so the same pair could read BUY on one page and SELL on the next.
The sync architecture removes that:

- **One data spine** (`src/services/market_data.py`) — every page fetches
  OHLC through the same canonical windows (weekly = 2y of daily bars
  resampled, daily = 300d, 4H = 90d of hourly resampled).
- **One bias engine** (`src/core/bias.py`) — per timeframe, three unanimous
  votes (EMA20 vs EMA50, price vs each) give BULLISH / BEARISH / NEUTRAL; a
  Weekly-weighted composite (Weekly 3, Daily 2, 4H 1) is the instrument's
  **house view**.
- **The strip** — every directional page shows the house view for its focus
  pair and flags in red when its own lens opposes it. A page's read is a
  *lens*; the house view is the reference. Market Overview's trading ideas
  and the Daily Cockpit's aligned ideas carry the same conflict flag inline.
- **One parameter home** — indicator defaults (EMA 20/50/200, MACD 12/26/9,
  swing strength) live on `AppConfig`; sliders default from them and label
  any deviation a "custom lens".

**What the house view is NOT:** an entry signal. The walk-forward Validation
tab on MTF Matrix (no-lookahead, unit-test-proven equivalent to the live
engine) showed on its first 5-year run (EUR/USD, XAU/USD, WTI/USD) that hit
rates sat below 50% and trend-alignment was actively harmful on oil. Use it
as the consistency/regime filter it is — it keeps the pages (and you) from
contradicting each other and keeps you out of counter-trend trades in
trending instruments; the edge must come from the setup layer (location,
trigger, risk management).

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
| **🎰 Setup Ranker** | The multi-timeframe checklist (`src/core/signals.score_setup`) run for every registry pair, both LONG and SHORT: 10 technical criteria (weekly EMA alignment, weekly RSI room, weekly/daily/4H structure, daily EMA/MACD, ATR expansion, 4H confluence-zone proximity, spread/ATR ratio), plus an 11th **Currency Strength** criterion for FX pairs (20-day base-vs-quote return differential; commodities have no single driving currency and stay on the 10-point scale — the "Min score" filter and email threshold are both applied as a percentage of whichever scale a pair is on, not a raw count). | Auto-refreshes every 5 min. **70%+ = today's candidates.** The grade (A–D) is the score bucketed as a percentage — grade A is ≥80%. A **background worker** re-runs this exact scan every 5 minutes inside the server itself, journaling new Grade-A setups **even when no browser is open** — the app only needs to be running, not watched. **Grade A no longer emails on its own**: it's a week-long opinion that fired too often to be worth interrupting you. Emails now require **triple confluence** (see below). Criteria include a **Daily 200 SMA** regime filter — longs want price above it, shorts below — scored, never a veto, because every new trend begins by crossing the 200. Note it is the *daily* 200; a 200-period MA on an intraday chart spans ~8 days and will often disagree. |
| **📊 AMD Scanner** | Accumulation / Manipulation / Distribution phases on 1H bars over the current trading week, with a true-range activity proxy substituting for spot FX's meaningless zero-volume data. | Tells you whether a pair is still ranging (Accumulation), about to fake out (Manipulation), or already trending (Distribution) — context for *when* in the AMD cycle a breakout is more or less likely to hold. |
| **📡 Trend Signals** | The 6-condition trend scorer (`TrendSignalEvaluator`): price vs 200 EMA, 50/200 EMA cross, price vs 50 EMA, RSI band, MACD vs signal, ADX > 25. | ≥4/6 conditions = BUY/SELL; ≥5/6 = STRONG BUY/SELL. This is the same evaluator the new Instrument Predictor (Section 8) uses as one of its four votes. |
| **🧭 MTF Matrix** (now first in Morning Brief) | The **house-view consensus board**: the canonical Weekly/Daily/4H direction (`src/core/bias.py`) for every registry pair — composite score, per-timeframe arrows, ALIGNED flag — plus a **🧪 Validation** tab that walk-forward-backtests the house view with no lookahead. The legacy candle-sentiment grid survives in an expander as a secondary lens. | Start the day here. ALIGNED rows are where every timeframe agrees; contested rows are the market telling you there's no trend trade. Remember the validation finding: this is a consistency/regime filter, not an entry signal. |
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
| **🗓️ Week Ahead** | The Sunday pre-flight, and the honest answer to "what will next week do?". Per instrument: the **expected range** (±1σ over 5 trading days from a GARCH(1,1) term-structure forecast — roughly 2 weeks in 3 finish inside it), the **vol regime**, **COT crowding**, nearest **supply/demand**, and the count of **high-impact events** hitting that instrument's currencies. A cone chart shows the band widening with √time from the last close. The page deliberately forecasts **volatility, not direction** — the house-view walk-forward proved direction has no standalone edge — so a move beyond the band is unusual (fade territory) and a move inside it is noise (not a breakout). Finish by writing a **bias + invalidation** per instrument; it saves to the same `swing_theses` table the Swing Playbook reads. Research context: never writes `trade_setups`. |
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
| **⚡ 15M Fib Entry** | Wait for **at least two of three**, within 1–2 candles: Stochastic cross below 25 (longs) / above 75 (shorts), 15M RSI reset off 40/60, a Bollinger-band touch curling back in. **Do not enter** if price is still slicing through the zone, the daily shows a conflicting flag, or the spread is abnormally wide ahead of news. Email alerts for fresh golden-zone entries are **on by default**, and the page re-scans itself every 5 minutes while open — leave it up during your session and the fresh triggers come to you. |

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
| **📓 Trade Journal** | Equity curve, win rate vs a 66% target, MT4 statement import. Also where you close a trade (updates the Checklist's daily-loss counter). Every open signal is checked against live price for a crossed stop level — 🔴 INVALIDATED is a visibility badge only; it never changes Outcome/Close/Open, which still require an actual close. The **🏆 Source Scorecard** tab resolves every persisted signal against what price did afterwards (recorded R for closed trades; conservative TP/SL bar replay, stop-only horizon marks, or 10-bar directional hit/miss for open signals) and ranks every signal source by realized expectancy. Alongside expectancy it reports **profit factor** (gross winning R / gross losing R; 1.00 is break-even, blank when a source has no losses yet) and a **Sortino ratio** (mean R over *downside* deviation only, so a source is not penalised for its big wins the way a Sharpe ratio would penalise it). Those two exist because expectancy is a mean and a mean hides how it was earned: two sources both averaging +0.25R are not the same source if one grinds and the other is carried by a single outlier. The Sortino is **per signal and not annualised** — these R-multiples are irregularly spaced and each source is marked on its own horizon, so there is no fixed period to annualise against. Check it monthly: demote sources with negative expectancy, trust the positive ones — but only once a source has ~20+ resolved signals, a floor that matters more for profit factor than for win rate since one extra loss can halve it. |

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
| **🧲 Leading Indicators** | Oscillators and flow proxies that can turn *before* price, deliberately orthogonal to the trend/structure stack: DeMarker (buying/selling exhaustion, >0.70 / <0.30), a volume-delta flow proxy (candle-direction-signed volume; true-range weight on zero-volume spot FX), Stochastic / Williams %R / CCI context, prior-bar floor-trader pivots, and the combined **RSI divergence + pivot confluence** read — the one that suits hard-trending XAU/XAG, where naked overbought/oversold gets run over. The house-view strip flags when its watch direction opposes the canonical read. Audit-only (`tool_usage_log`, tool `leading_ind`) — context, never a standalone trigger. |
| **🇺🇸 Indices Correlation** | S&P 500 (^GSPC) vs Dow Jones (^DJI): candles for each, the rolling correlation of their daily returns (short + long windows), and the S&P/Dow relative-strength ratio. Correlation is normally very high (~0.95); the read is what happens when it *falls* — a decoupling means a rotation or a single-index shock — or when the ratio breaks trend (which benchmark is leading). Cross-asset risk-on/risk-off backdrop, not a trade trigger. Dual chart engine (TradingView pilot / Terminal Plotly). Audit-only (`tool_usage_log`, tool `indices_corr`). |
| **📅 Busy-Week Anatomy** | Historical study of price behavior during high-event-density weeks. |
| **🔌 Disconnect Monitor** | Tests "this divergence should close" theses (e.g. real yields vs gold) via a rolling-regression residual z-score, then an event study of what actually happened after past disconnects at this threshold. |
| **🌙 Overnight Drift** | Studies the overnight-session return pattern for index futures. |
| **⏱️ Optimal Holding Period** | Backtests how many days a breakout-style entry should be held before mean-reverting/decaying, per pair. |
| **🎓 Stochastic Calculus** | The maths sitting under the risk model, applied to live prices. **Volatility**: five estimators side by side — close-to-close throws away everything that happened inside the bar, while the range estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang) don't and are several times more efficient on the same sample; prefer Yang-Zhang, the only one handling both overnight gaps and intraday drift. A Hurst exponent then tests whether variance really scales linearly in time — the assumption you make *every time* you scale a daily vol by √21. **Drag & leverage**: Itô's lemma turns an arithmetic drift μ into a log-drift ν = μ − σ²/2, and that gap is what volatility costs you before any spread — not a fee, arithmetic. The Kelly curve g(f) = fμ − f²σ²/2 shows where drag beats edge (past f*) and where growth returns to zero (2f*). **Target vs stop**: the scale function gives P(target fills before stop) — and with zero drift it collapses to gambler's ruin l/(u+l), meaning **no choice of target and stop creates edge**; that has to come from entry timing. **Options**: Black-Scholes price, Greeks and implied vol, where the real-world drift μ is conspicuously absent — two traders who disagree on direction still owe the same price, which is what makes an options market a market in σ. Audit-only. |
| **🎲 Martingale Diagnostics** | The question every trader eventually has to face: **is my equity curve edge, or luck?** Doob's decomposition splits your closed-trade R series into a *compensator* (the part predictable from prior trades — edge) and a *martingale* remainder (luck). "Edge share" tells you what fraction of net R the drift explains; a share near zero means the result rests on the unpredictable part. **Azuma** and **Freedman** then bound how far the luck component can plausibly wander — Azuma assumes every trade could be a maximal loser, Freedman adapts to your realised variance and is far tighter, so it's the one to act on. Also answers two questions that ruin people: **McDiarmid** shows how much a single trade can move your Sharpe (if the band straddles zero, that Sharpe is indistinguishable from no skill), and the **optional-stopping bootstrap** shows that a tight target with a wide stop wins ~80% of the time on a *driftless* series and still nets exactly zero — which is why win rate alone says nothing. Uses **executed trades only**, in R-multiples, so it agrees with the Journal's equity curve. ⚠️ The "is there an edge" verdict runs three tests at 5% each *without* multiple-comparison correction, so an edgeless series is flagged in ~12–14% of samples by chance — read a single rejection as a hint, never a finding. |
| **🧮 Quant Models Lab** | Six statistical models for pairs research: Engle-Granger cointegration test, Kalman-filter dynamic hedge ratio, Ornstein-Uhlenbeck mean-reversion fit (half-life + z-bands), GARCH(1,1) volatility regime + vol-targeted position sizing, UIP/carry deviation, and a GBM Monte-Carlo null test for judging whether *any* backtest number beats pure luck. See Section 9.4 for how this feeds the Instrument Predictor's vol-regime confidence modifier. |
| **🔭 Forecast** | A GARCH(1,1)-t volatility cone (1 week / 1 month / 1 quarter) around a random-walk center — deliberately no point forecast, since beating a random walk at these horizons is hard (Meese-Rogoff). Alongside it, a transparent driver score (trend, 200d MA, momentum, 20-day breakout, and the ABR Toolkit's structure bias when available) and a plain-language narrative (template by default, polished by the Claude API when `ANTHROPIC_API_KEY` is set). Every forecast is journaled and later scored against what actually happened — hit rate inside the 68%/95% bands, and whether the directional call was right. | The self-scoring history is the point: it tells you honestly, over time, whether this page's driver score has any real edge for a given instrument, rather than asking you to trust it on faith. |
| **😲 Surprise Awareness** | Three tools sharing one economic-calendar feed: (1) a **surprise index** — z-scored actual-vs-forecast per data release (Citi-ESI style), covering every registry currency, not just USD; (2) an **event-proximity gate** per instrument — suppresses/flags signals when a high-impact release for its currencies is imminent or just happened; (3) a **gold~oil rolling correlation** to catch when "escalation = buy gold" has inverted (it did, Feb 2026). Telegram-alerts on qualifying surprises if `[telegram]` is configured in secrets.toml. | Check the gate for your candidate instrument before triggering an entry — a red-folder release inside the window means stand aside, the geopolitical/data premium unwinds in hours, not weeks. |
| **🧾 Event Reaction Map** | Four scheduled US releases — **NFP, CPI, PPI and FOMC** — scored rather than narrated. Enter what printed against what was expected; each component is z-scored against its own historical surprise SD and weighted into one composite (positive = hawkish). Leaving a consensus box empty *drops* that component and renormalises, which is not the same as typing the actual into it — the first says "I don't know", the second says "it came in on forecast". The composite is then pushed through **that event's own** transmission chain rather than the naive one-way chain that circulates on social media. Three things differ per event and they are the substance of the page. **(1) "Hawkish" does not mean the same thing for growth.** A hot NFP is hawkish *and* growth-positive; a hot CPI is hawkish and growth-*negative* — a real-income squeeze plus tighter policy — so every risk asset's growth beta flips sign between them. Strong payrolls buy equities in a growth-scare regime; hot CPI sells them. **(2) The chain differs.** CPI never passes through jobs, and its `real income` node moves *against* the print. PPI runs input costs → margins (against) → consumer prices. FOMC runs policy → front end → real yields → liquidity (against). **(3) FOMC is not an 08:30 event.** It lands 14:00 New York ≈ 20:00 SAST — you are awake for it — and its session map carries a **Presser** phase at +30 rather than a US cash open, because 09:30 New York is hours *before* the decision. Its surprise is not a statistic minus a forecast but decision-vs-priced (bp), the dot-plot median shift (bp), and a tone dial. **Conviction** is `|a+b| / (|a|+|b|)` over the rate and growth channel contributions — it collapses toward 0 when they fight, which is exactly when the first fifteen minutes are a coin flip. | Set the regime from what the market has been *rewarding* in the last few prints, not from what should be true. Skip anything flagged low-conviction. **The betas are priors, not findings**, and the CPI/PPI/FOMC tables have less standing than NFP's — none is fitted to your tape. What makes them falsifiable is that each event persists under its **own** source tag (`nfp_reaction` / `cpi_reaction` / `ppi_reaction` / `fomc_reaction`), so the Trade Journal's Source Scorecard grades each separately — CPI betas being good says nothing about NFP's. Expect a useful read after roughly a dozen releases per event, which is a year for FOMC. DXY, US500, NAS100, US10Y and BTCUSD are read on the board but never scored — no tradable registry pair. Release dates come from `src/core/event_calendar.py`; the CPI/PPI/FOMC lists are hand-maintained seeds and the picker says so when one runs out rather than inventing a date. |

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
