# Dashboard-Pro

A modular, professional Forex Macro Dashboard combining technical analysis, macro fundamentals, and entry signal generation.

---

## Project Structure

```
Dashboard-Pro/
├── app.py                         # Production dashboard entry point
├── src/
│   └── core/
│       ├── analyzer.py            # Technical indicators (via ta library)
│       ├── data_provider.py       # Data fetching — Yahoo Finance, QuantConnect, FRED
│       ├── signals.py             # Entry signals and trading idea generation
│       └── config.py              # Centralised application configuration
└── archive/                       # Legacy and experimental versions
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

## Notes

- The `archive/` folder contains earlier iterations and is not imported by the production dashboard.
- Macro fallback data in `data_provider.py` is static as of 2024-Q1. Add a FRED API key to get live values.
- The Trading Ideas tab auto-refreshes every 5 minutes using Streamlit's `@st.fragment` — no manual refresh needed.