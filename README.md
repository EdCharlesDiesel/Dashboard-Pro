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


Your Daily Trading Routine — SAST (UTC+2)

  The good news: 09:00 SAST = 07:00 UTC — you wake up exactly at the London Kill Zone open. That's
  your highest-probability window of the day.

  ---
  Kill Zone Schedule (SAST)

  ┌──────────────────┬─────────────┬─────────────┬──────────────┐
  │     Session      │     UTC     │    SAST     │   Priority   │
  ├──────────────────┼─────────────┼─────────────┼──────────────┤
  │ London Kill Zone │ 07:00–09:00 │ 09:00–11:00 │ 🟢 Prime     │
  ├──────────────────┼─────────────┼─────────────┼──────────────┤
  │ NY Kill Zone     │ 12:00–14:00 │ 14:00–16:00 │ 🟢 Prime     │
  ├──────────────────┼─────────────┼─────────────┼──────────────┤
  │ London Close     │ 15:00–17:00 │ 17:00–19:00 │ 🟡 Secondary │
  ├──────────────────┼─────────────┼─────────────┼──────────────┤
  │ Tokyo            │ 00:00–03:00 │ 02:00–05:00 │ 🔴 Avoid     │
  └──────────────────┴─────────────┴─────────────┴──────────────┘

  You have two prime windows — morning and afternoon. Most days one clean trade is enough.

  ---
  Daily Routine — Page by Page

  ☕  08:30–09:00 SAST — Pre-market prep (30 min)

  Work top-down through the higher timeframes before price starts moving:

  ┌──────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │       Page       │                            What you're deciding                             │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 01. Macro Bias   │ Is the fundamental backdrop bullish or bearish for your target pairs today? │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 02. News Filter  │ Any red-folder news in the next 2 hours? If yes, wait or skip               │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 03. Correlations │ Is DXY, Gold, S&P moving in a way that confirms your bias?                  │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 05. Weekly EMA   │ Which pairs have clean weekly trend alignment?                              │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 06. Weekly RSI   │ Is RSI extended (avoid) or has room (favour)?                               │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 08. Daily Trend  │ Daily EMA tells you the directional bias for today                          │
  ├──────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ 09. Daily MACD   │ MACD histogram — is momentum building or fading?                            │
  └──────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  Output: You should have 2–4 pairs on your watchlist and a clear LONG or SHORT bias for each.

  ---
  🔎 09:00–09:15 SAST — Find the best setup

  ┌────────────────────────┬────────────────────────────────────────────────────────────┐
  │          Page          │                    What you're deciding                    │
  ├────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 18. Setup Ranker       │ Run it — pick the pairs scoring 7+/10, your bias direction │
  ├────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 10. 4H Confluence Zone │ Is price at a 4H zone right now? PDH/PDL in the area?      │
  ├────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 04. ATR Volatility     │ Is spread/ATR ratio ≤5%? Is volatility ok to trade?        │
  ├────────────────────────┼────────────────────────────────────────────────────────────┤
  │ 11. Confluence Checker │ Do 2 of 3 key confluences line up?                         │
  └────────────────────────┴────────────────────────────────────────────────────────────┘

  Output: One pair, one direction, confirmed at a key level.

  ---
  ⚡  09:15–11:00 SAST — Watch for the entry (London Kill Zone)

  This is the execution window. Open the checklist on your target pair.

  Step 1 — Fill the checklist (00. Checklist)
  - Checks 1–10 you can tick from your pre-market work
  - The MTF Alignment strip tells you instantly if Weekly/Daily/4H all agree
  - Watch the correlation exposure warning — if you already have a correlated position open, skip

  Step 2 — Wait for the trigger

  ┌───────────────────┬───────────────────────────────────────────────────────────────────────────┐
  │       Page        │                           What fires the entry                            │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ 12. 15M Rejection │ A pin bar, engulf, or tweezer at the 4H zone + PDH/PDL sweep marker 🔄    │
  ├───────────────────┼───────────────────────────────────────────────────────────────────────────┤
  │ 13. 15M Entry     │ Stochastic crossover + RSI reset — ideally LONG CONFIRMED / SHORT         │
  │ Signal            │ CONFIRMED (vol spike too)                                                 │
  └───────────────────┴───────────────────────────────────────────────────────────────────────────┘

  Rule: Only take the trade when the checklist hits 16/18+ AND all checks 11–16 are ticked. The signal
   chip must show 🟢 GO.

  ---
  🛡️Before you click — 2-minute checks

  ┌────────────────────┬───────────────────────────────────────────────────────────────────┐
  │        Page        │                            Quick check                            │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 14. Stop Structure │ Is the stop behind a real structure level?                        │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ 15. R:R Calculator │ Is R:R ≥ 2:1 to TP1?                                              │
  ├────────────────────┼───────────────────────────────────────────────────────────────────┤
  │ Daily Loss Limit   │ Already 1 loss today? Trade smaller. Already 2? Close the laptop. │
  └────────────────────┴───────────────────────────────────────────────────────────────────┘

  ---
  📋 Log the trade

  In 00. Checklist → Save Trade Setup to PostgreSQL. This captures everything for the journal.

  ---
  🌇 14:00–16:00 SAST — NY Kill Zone (second opportunity)

  If the London window gave no clean GO signal:
  - Re-run pages 10 → 11 → 12 → 13 for the NY open
  - Same checklist process
  - Price often re-tests the London session level — this is a second entry chance

  ---
  📓 19:00 SAST — End of day review (10 min)

  ┌───────────────────────┬────────────────────────────────────────────────────────────────┐
  │         Page          │                         What to review                         │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ 16. Trade Journal     │ Did equity curve move up? Win rate trending toward 66%?        │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ 17. Market Structure  │ Did the pairs you traded break structure as expected?          │
  ├───────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Close any open trades │ 00. Checklist → Close Trade — record outcome, pips, R multiple │
  └───────────────────────┴────────────────────────────────────────────────────────────────┘

  ---
  The 66% Win Rate Formula

  The system is designed around this logic:

  High-probability setup =
    Weekly bias aligned  ✅
    Daily trend confirmed ✅
    4H zone + PDH/PDL    ✅
    MTF FULLY ALIGNED    ⭐
    London/NY kill zone  ✅
    Rejection candle     ✅
    Entry signal CONFIRMED (vol spike) ✅
    ≥16/18 checks + critical path      ✅

  When all of those are present: take the trade.
  When even one critical check (11–16) is missing: sit on your hands.

  The Trade Journal page tracks your win rate in real time against the 66% target — if you're below
  it, filter the journal by session and pair to find where you're leaking (wrong session, wrong pair,
  skipping the checklist).

  ---
  Discipline rules baked into the system

  - 2 losses today → system blocks new entries (daily loss limit tracker)
  - Correlated pairs open → system warns you (stacked risk)
  - Not in a kill zone → Check #10 shows red (session enforcement)
  - Setup ranks below 7/10 → skip it (Setup Ranker filter)

  One focused trade in the London window, reviewed that evening — done consistently, that's the path
  to 66%.
