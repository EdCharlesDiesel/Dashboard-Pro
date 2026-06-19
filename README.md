# Dashboard-Pro

A modular, professional Forex Macro Dashboard combining technical analysis, macro fundamentals, and entry signal generation.

---

## Project Structure

```
Dashboard-Pro/
├── app.py                         # Entry point — runs the Daily Trading checklist (master page)
├── pages/                         # 22 workflow pages (incl. market-overview.py, the macro dashboard)
├── src/
│   ├── core/                      # analyzer, data_provider, signals, config (shared engine)
│   ├── pages_lib/                 # BloombergPage framework + navigation
│   ├── indicators/ instruments/   # indicator math + instrument registry (single source of truth)
│   ├── services/ ui/ db/          # forecast/atr/etc · terminal theme & components · Postgres
└── archive/                       # Legacy and experimental versions — do not touch
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your FRED API key** *(optional but recommended for live macro data)*

Add to `.streamlit/secrets.toml`:
```toml
FRED_API_KEY = "your_api_key_here"
```
Or export as an environment variable:
```bash
export FRED_API_KEY=your_api_key_here
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

---

## Dashboard Tabs

The tabs are ordered to mirror your daily top-down analysis workflow — open them left to right.

| # | Tab | Purpose |
|---|-----|---------|
| 0 | 📊 Market Overview | Morning scan — what moved overnight |
| 1 | 🌍 Macro Fundamentals | Rates, GDP, inflation per currency |
| 2 | 🏛 Macro Dashboard | Deep FRED macro data grid |
| 3 | 📅 Weekly Swing | Weekly pivot setups and directional filter |
| 4 | 🧭 Multi-Timeframe Matrix | Sentiment alignment across all timeframes |
| 5 | 📈 Technical Chart | Chart analysis per pair (EMA, RSI, MACD) |
| 6 | 🛒 Pivots & Fibonacci | Session key levels and S/R zones |
| 7 | ⚡ Signal Pro | QuantConnect-style signal confirmation |
| 8 | 🎯 Trading Ideas | Auto-generated setups, refreshed every 5 min |
| 9 | ⏱️ 15-Min Entry | Execution timing — Stoch, RSI, BB triggers |
| 10 | 🔥 Trend Following | Scanner for strong trending conditions |
| 11 | 🔊 Volume Profile | Volume distribution (POC, HVN, LVN, VA) |
| 12 | 🛡️ Supertrend Strategy | Dual Supertrend + QQE + DEMA strategy |
| 13 | 🧪 Backtest Lab | Historical strategy testing |

---

## Top-Down Trading Framework

The core principle is **confluence** — never trade a single signal. Each step below narrows the universe of possible trades until only high-probability setups remain.

---

### Step 1 — Macro Backdrop
*Tabs: Macro Fundamentals · Macro Dashboard*

Before touching a chart, understand *why* a pair should move.

| Factor | What to look for |
|--------|-----------------|
| **Interest rate differential** | The primary driver. Capital flows toward the higher-rate currency. |
| **Inflation trajectory** | A central bank actively hiking to fight inflation = currency strength. |
| **GDP growth divergence** | Compare both sides of the pair. Faster growth = structural tailwind. |
| **Yield curve** | Inverted (2Y > 10Y) signals recession risk — medium-term currency negative. |

**Output:** One sentence — *"Fundamental bias for [pair] is [Long/Short/Neutral] because [reason]."* This is your directional filter. Only look for setups that align with it.

---

### Step 2 — Weekly Trend
*Tab: Weekly Swing*

The weekly chart defines the macro trend and where institutional money is positioned.

- **EMA 20 vs EMA 50** — EMA20 above EMA50 with price above both = weekly uptrend.
- **Price vs Weekly Pivot** — sustained above PP = buyers in control; below PP = sellers.
- **RSI** — above 50 confirms bullish momentum. Above 70 = overbought, be cautious adding longs.
- **R1/R2/S1/S2** — these become your weekly profit targets and invalidation levels.

> **Key rule:** If the weekly trend is up, you only look for long setups on lower timeframes. Never fight the weekly trend.

---

### Step 3 — Daily Confirmation
*Tab: Technical Chart*

The daily chart confirms the weekly is intact and tells you whether you're entering well or chasing.

- **EMA alignment** — EMA20 > EMA50 should match the weekly bias. If daily has crossed bearish while weekly is still bullish, you are in a *pullback*, not a reversal — pullbacks are buying opportunities.
- **Daily Pivot** — bullish structure requires price above the daily PP.
- **RSI** — between 40–60 on a pullback gives room to run. RSI above 70 = momentum is extended, wait for a reset.
- **MACD histogram** — turning from negative toward zero on a pullback signals exhausting sell-side momentum.
- **ATR** — sets your minimum stop distance. If ATR is 80 pips and your stop is 10 pips, noise will stop you out.

**Output:** Are you in an *impulse* leg (trending) or a *corrective* leg (pullback)? Enter at the end of the corrective leg.

---

### Step 4 — 4H Structure & Entry Zone
*Tab: Pivots & Fibonacci*

The 4H chart defines the exact zone where you want to execute.

- **4H Pivot Points** — R1/S1 = immediate targets; R2/S2 = extended targets.
- **Fibonacci retracement** — the 38.2%, 50% and 61.8% levels mark where institutional orders cluster. A pullback to one of these levels in a trending market is a high-probability setup.
- **Price structure** — confirm higher highs and higher lows (uptrend) or lower highs and lower lows (downtrend) are intact.
- **EMA 20 as dynamic support** — in a healthy uptrend, price bounces off the 4H EMA20.
- **4H MACD crossover** — MACD crossing above signal while price is at a Fibonacci level and above PP is a strong trigger.

**The confluence zone** = Fibonacci level + Pivot S/R + EMA20 overlap. When two or three of these align at the same price, that is where you execute.

---

### Step 5 — Entry Trigger
*Tab: 15-Min Entry*

You have a direction, a zone, and a bias. Now you need a precise trigger.

- **Stochastic crossover below 25** (for longs) — K crosses above D in oversold territory.
- **RSI reset** — 15M RSI dips below 40 then turns up, confirming the pullback is exhausting.
- **Lower Bollinger Band touch** — price hits the lower band and begins curling back inside.

Wait for at least two of these three to fire within the same candle or within 1–2 candles of each other. One signal alone is not enough.

**Do not enter if:**
- Price is still falling through the 4H confluence zone without finding support
- Weekly Swing tab shows **⚠️ Conflicting** on the daily confirmation
- The spread is abnormally wide (news event imminent)

---

### Step 6 — Risk Definition
*Before order placement — not after*

**Stop Loss**
Place the stop *below the structure that justifies the trade*. For a long at the 50% Fibonacci, the stop goes below the 61.8% level or below the swing low that created the move. Minimum distance = 1× ATR — anything tighter will be hit by normal noise.

**Position Sizing**
```
Position size = (Account × Risk %) ÷ (Stop distance in pips × Pip value)
```
Risk 1–2% of account per trade. Never size based on how confident you feel.

**Risk:Reward**
Only take trades where TP1 gives at minimum **2:1 R:R** (dashboard filters at 1.5:1). Target 2.5:1 or better.

> A strategy with a 40% win rate and 2.5:1 R:R is profitable over 100 trades. A 60% win rate with 1:1 R:R barely breaks even after costs.

---

### Step 7 — Trade Management

**While in profit:**
1. Move stop to **breakeven** once price moves in your favour by the same distance as your initial stop.
2. Take **50% off at TP1** (R1 pivot), let the remainder run to TP2.
3. Once TP2 is in range, move the remaining stop to just below TP1.

**Exit without hesitation if:**
- A 4H candle closes on the wrong side of the EMA20
- Daily RSI crosses back through 50 against your position
- A weekly candle closes below the PP while you are long

---

## Pre-Trade Checklist

Run through this in order before every entry.

```
[ ] 1. Macro bias confirmed (rates, GDP, inflation favour direction)
[ ] 2. Weekly EMA aligned with trade direction
[ ] 3. Weekly RSI has room (not already overbought/oversold)
[ ] 4. Daily trend intact (EMA20 > EMA50 for longs)
[ ] 5. Daily MACD momentum turning in direction
[ ] 6. Price at a 4H confluence zone (Fib + Pivot + EMA overlap)
[ ] 7. 15M entry signal fired (Stoch crossover + RSI reset)
[ ] 8. Weekly Swing tab daily confirmation shows ✅ Aligned
[ ] 9. Stop is below structure, minimum 1× ATR distance
[ ] 10. R:R is at least 2:1 to TP1
```

If any of steps 1–4 conflict with the trade direction, skip it entirely. If steps 5–7 have not fired, wait — the setup is not ready.

---

## Daily Trading Routine — SAST (UTC+2)

You wake up exactly at the London Kill Zone open — your highest-probability window of the day.

### Kill Zone Schedule

| Session | UTC | SAST | Priority |
|---------|-----|------|----------|
| London Kill Zone | 07:00–09:00 | 09:00–11:00 | 🟢 Prime |
| NY Kill Zone | 12:00–14:00 | 14:00–16:00 | 🟢 Prime |
| London Close | 15:00–17:00 | 17:00–19:00 | 🟡 Secondary |
| Tokyo | 00:00–03:00 | 02:00–05:00 | 🔴 Avoid |

You have two prime windows — morning and afternoon. Most days one clean trade is enough.

---

### ☕ 08:30–09:00 SAST — Pre-Market Prep (30 min)

Start with the Setup Ranker to build your candidate shortlist, then validate direction top-down.

| Page | What you're deciding |
|------|----------------------|
| 01. Setup Ranker | Run in **Both** direction mode — which pairs score 7+/10? These are your candidates for today. |
| 02. Macro Bias | For each candidate, confirm the fundamental direction — rates, GDP, inflation back the bias? |
| 03. News Filter | Any red-folder news in the next 2 hours? If yes, wait or skip. |
| 04. Correlations | Is DXY, Gold, S&P moving in a way that confirms your bias? |
| 06. Weekly EMA | Which pairs have clean weekly trend alignment? |
| 07. Weekly RSI | Is RSI extended (avoid) or has room (favour)? |
| 09. Daily Trend | Daily EMA tells you the directional bias for today. |
| 10. Daily MACD | MACD histogram — is momentum building or fading? |

**Output:** 1–2 pairs confirmed by both technical score and macro fundamentals, with direction set.

---

### 🔎 09:00–09:15 SAST — Confirm the Entry Zone

| Page | What you're deciding |
|------|----------------------|
| 11. 4H Confluence Zone | Is price at a 4H zone right now? PDH/PDL in the area? |
| 05. ATR Volatility | Is spread/ATR ratio ≤5%? Is volatility ok to trade? |
| 12. Confluence Checker | Do 2 of 3 key confluences line up? |

**Output:** One pair, one direction, confirmed at a key level.

---

### ⚡ 09:15–11:00 SAST — Watch for the Entry (London Kill Zone)

This is the execution window. Open the checklist on your target pair.

**Step 1 — Fill the checklist (00. Checklist)**
- Checks 1–10 you can tick from your pre-market work.
- The MTF Alignment strip tells you instantly if Weekly/Daily/4H all agree.
- Watch the correlation exposure warning — if you already have a correlated position open, skip.

**Step 2 — Wait for the trigger**

| Page | What fires the entry |
|------|----------------------|
| 20. 15M Fib Entry | Price retraces into the 0.382–0.618 Fibonacci golden zone of the 15M impulse leg and a confirming candle closes in the bias direction — only on Setup Ranker signals, with optional email alerts |

> **Rule:** Only take the trade when the checklist hits 16/18+ AND all checks 11–16 are ticked. The signal chip must show 🟢 GO.

---

### 🛡️ Before You Click — 2-Minute Checks

| Page | Quick check |
|------|-------------|
| 15. Stop Structure | Is the stop behind a real structure level? |
| 16. R:R Calculator | Is R:R ≥ 2:1 to TP1? |
| Daily Loss Limit | Already 1 loss today? Trade smaller. Already 2? Close the laptop. |

---

### 📋 Log the Trade

In **00. Checklist → Save Trade Setup to PostgreSQL**. This captures everything for the journal.

---

### 🌇 14:00–16:00 SAST — NY Kill Zone (Second Opportunity)

If the London window gave no clean GO signal:
- Re-run pages 11 → 12 → 13 → 14 for the NY open.
- Same checklist process.
- Price often re-tests the London session level — this is a second entry chance.

---

### 📓 19:00 SAST — End of Day Review (10 min)

| Page | What to review |
|------|----------------|
| 17. Trade Journal | Did equity curve move up? Win rate trending toward 66%? |
| 18. Market Structure | Did the pairs you traded break structure as expected? |
| Close any open trades | 00. Checklist → Close Trade — record outcome, pips, R multiple. |

---

### The 66% Win Rate Formula

The system is designed around this logic:

```
High-probability setup =
  Weekly bias aligned          ✅
  Daily trend confirmed        ✅
  4H zone + PDH/PDL            ✅
  MTF FULLY ALIGNED            ⭐
  London/NY kill zone          ✅
  Rejection candle             ✅
  Entry signal CONFIRMED
  (vol spike)                  ✅
  ≥16/18 checks + critical
  path                         ✅
```

When all of those are present: take the trade.
When even one critical check (11–16) is missing: sit on your hands.

The Trade Journal page tracks your win rate in real time against the 66% target — if you're below it, filter the journal by session and pair to find where you're leaking (wrong session, wrong pair, skipping the checklist).

---

### Discipline Rules Baked Into the System

- **2 losses today** → system blocks new entries (daily loss limit tracker).
- **Correlated pairs open** → system warns you (stacked risk).
- **Not in a kill zone** → Check #10 shows red (session enforcement).
- **Setup ranks below 7/10** → skip it (Setup Ranker filter).

One focused trade in the London window, reviewed that evening — done consistently, that's the path to 66%.

---

## Notes

- The `archive/` folder contains earlier iterations and is not imported by the production dashboard.
- Macro fallback data in `data_provider.py` is static as of 2024-Q1. Add a FRED API key to get live values.
- The Trading Ideas tab auto-refreshes every 5 minutes using Streamlit's `@st.fragment` — no manual refresh needed.
# Forex Fundamentals — Methodology

This documents two of the trickier tabs: the **Priced-In Analyzer** and the
**Risk Sentiment** gauge.

---

# Part 1 · Priced-In Analyzer

## The idea in one line
A currency reacts to the **surprise**, not the event. To trade fundamentals you
must know what the market *already expects* — what's "priced in" — so you can
judge whether that expectation is too high or too low. This tool reads the
expectation straight from the interest-rate market.

Two engines, because there's no single free source that does both well:

| Engine | Covers | Source | Output |
|---|---|---|---|
| Curve-implied | All majors | FRED (no key) | Direction + rough size of priced policy change |
| Fed funds futures | US only | yfinance `ZQ=F` | Probability of a move at the next meeting |

---

## Engine 1 — Curve-implied expectations

A short market rate (~3-month) reflects the **current** policy stance. A longer
market rate reflects the **average expected** policy rate over its horizon. The
spread between them is, roughly, the net policy change the market has priced.

```
priced_bps   = (long_rate − short_rate) × 100
moves_priced = priced_bps / 25          # in 25bp units
```

Read it as direction and magnitude:

- **Positive spread** → hikes priced in (curve upward-sloping)
- **Negative spread** → cuts priced in (curve inverted)
- **Flat** → market expects policy on hold

### The one caveat that matters
The long rate also contains a **term premium** (compensation for holding
duration), so this *overstates* the pure rate-expectation component. It is
reliable for **direction and rough magnitude**, not for a clean probability.
Two ways to tighten it if you want: swap the 10y for a 2y (less term premium),
or use OIS/swap rates (cleanest, but not free). For a precise single-meeting
number, use Engine 2.

---

## Engine 2 — Fed funds futures probability (US)

A 30-day Fed funds future settles on the **average daily effective rate** over
its contract month. If a meeting falls mid-month, that month's average is a
blend of the rate *before* the meeting and the rate *after*:

```
avg_implied = 100 − futures_price
avg_implied = (current_rate × days_before + rate_after × days_after) / N
```

Solve for the post-meeting rate, then express how much of a full move is priced:

```
rate_after = (avg_implied × N − current_rate × days_before) / days_after
bp_priced  = rate_after − current_rate
prob       = bp_priced / hike_size      # fraction of a full move, e.g. 0.25
```

This is the simplified **CME FedWatch** methodology.

### Assumptions (they're baked in — know them)
1. Exactly **one meeting** in the contract month.
2. The effective rate equals the policy target **before** the meeting.
3. Settlement = simple average of daily rates (ignores intra-month weighting
   subtleties).

Near the 0% / 100% edges, treat the number as approximate. For cuts, the tool
flips `hike_size` negative and reports the fraction of a cut priced.

### Worked example
Current rate 5.00%, meeting on day 15 of a 30-day month, future at 94.8667:

```
avg_implied = 100 − 94.8667 = 5.1333
rate_after  = (5.1333×30 − 5.00×14) / 16 = 5.25
prob        = (5.25 − 5.00) / 0.25 = 1.00  → 100% priced
```

A 25bp hike is fully priced. If the Fed delivers exactly that, the decision is a
non-event — the reaction will come from the **guidance**, which is why this tool
pairs with the Statement Diff section.

---

## How to actually use it

The workflow that turns this into an edge:

1. **Before a meeting, read what's priced in** (this tool). Fully priced hike?
   Then the hike itself won't move the currency much.
2. **Form your own view** on whether that's right — cross-check the **Economic
   Surprise** tab. If data is running hot but the market only prices a partial
   hike, the market may be *under-pricing* → upside surprise risk.
3. **On the day, the signal is the gap.** Outcome vs priced-in = the move. And
   once the decision is priced, the **Statement Diff** wording shift is where
   the remaining information lives.

### Reading the output
- **~100% priced** → decision is a non-event; trade the guidance, and beware a
  no-change (large surprise).
- **~0% priced** → if it happens, it's a genuine shock; expect a sharp move.
- **Partial** → the outcome itself still carries information; the market is
  undecided.

### What this is *not*
It's not a forecast of what the central bank will do, and not a fair-value
model. It tells you what the market currently believes. Your edge is judging
whether that belief is mispriced — the tool just makes the belief explicit and
quantified instead of a vibe.

---

# Part 2 · Risk Sentiment & Capital Flows

## The idea in one line
Beyond rates and data, the market has a **mood**: risk-on or risk-off. In
risk-off, money stampedes into safe havens (USD, JPY, CHF) *regardless of the
fundamentals*. Knowing the regime often matters more than the micro data,
because it can override everything in the other tabs.

## Why it overrides fundamentals
Capital flows dominate. When fear spikes, large pools of money de-risk at once —
selling equities, EM, and commodity currencies, and buying havens and
government bonds. That flow swamps a good CPI print or a favourable rate
differential. A currency with great fundamentals can still fall hard in a
risk-off wave simply because it's a "risk" currency (AUD, NZD, ZAR). So you read
the regime *first*, then the fundamentals.

## The two sides

| | Risk-ON (greed) | Risk-OFF (fear) |
|---|---|---|
| Mood | Confident, chasing return | Scared, protecting capital |
| Equities | Up | Down |
| Volatility (VIX) | Low | Spiking |
| Credit spreads | Tight | Widening |
| Bought | AUD, NZD, CAD, EM, copper | USD, JPY, CHF, gold, Treasuries |
| Sold | JPY, CHF (funders) | AUD, NZD, EM |

**Why JPY and CHF are havens:** low domestic rates make them *funding
currencies* — borrowed cheaply to buy higher-yielding assets (the carry trade).
When risk-off hits, those trades unwind: traders buy back JPY/CHF to repay, so
the havens surge. **AUD/JPY** captures both sides at once (high-yield commodity
currency vs haven funder), which is why it's the classic risk barometer.

## How the gauge is built
Five market signals, each turned into a **z-score** (how unusual today is vs the
last ~year) and **oriented so positive always means risk-on**:

| Signal | Source | kind | orient | Reads |
|---|---|---|---|---|
| S&P 500 | yfinance `^GSPC` | ~1m momentum | +1 | equity appetite |
| VIX | yfinance `^VIX` | level | −1 | fear (high = off) |
| AUD/JPY | yfinance `AUDJPY=X` | ~1m momentum | +1 | direct risk barometer |
| Copper/Gold | `HG=F` / `GC=F` | ~1m momentum | +1 | growth vs safety |
| HY credit spread | FRED `BAMLH0A0HYM2` | level | −1 | stress (wide = off) |

```
component_z = rolling_zscore( momentum_or_level ) × orient   # + = risk-on
composite   = mean( available component_z's )
```

Classification:

```
composite > +0.4  → Risk-On
composite < −0.4  → Risk-Off
otherwise         → Neutral / Mixed
```

The gauge also plots the composite over ~2 years (so you can see regimes swing)
and a bar of each component's latest contribution (so you can see *what's*
driving it — e.g. "VIX and credit spreads are pulling it risk-off while equities
lag").

## How to read and use it
1. **Check the regime before anything else.** It's the weather; the other tabs
   are the terrain.
2. **Risk-off** → lean with havens. Cleanest expressions: **short AUD/JPY,
   short NZD/JPY**; USD/JPY is muddy because both are havens.
3. **Risk-on** → lean with growth/yield. Cleanest expression: **long AUD/JPY**.
4. **Neutral** → risk flows aren't driving; *now* the rate-differential,
   economic-surprise and priced-in tabs carry the signal.

The big trap this helps you avoid: being long a fundamentally strong but
risk-sensitive currency right as a risk-off wave hits. The fundamentals were
right and you still lose, because the regime overrode them.

## Limitations
- Momentum z-scores are ~1-month lookbacks — good for the prevailing regime, but
  they lag sudden intraday risk shocks. For fast moves watch VIX live.
- The composite is an **equal-weight** average. You can weight signals (e.g.
  lean on credit spreads and VIX) by editing `RISK_SIGNALS` and the averaging in
  `composite_from_components`.
- It describes the regime; it does not predict the *turn*. Regimes can persist or
  flip fast on a headline.