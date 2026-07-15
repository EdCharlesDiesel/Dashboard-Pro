# Live Session Runbook — Analysis → Trade

*A literal, in-order checklist for one live trading session: open each page in
this sequence, do the one thing listed, then move to the next. This is the
same workflow as the README's "Daily routine" and the System Guide, condensed
into something you follow step-by-step in real time rather than read as
reference. Mirrors `src/pages_lib/navigation.py` — if a page number here ever
disagrees with the sidebar, the sidebar (code) wins.*

**Rule for tonight: one focused trade, taken only at 🟢 GO on the Daily
Checklist (≥16/18 checks, all of 11–16 clear). If nothing reaches GO, the
correct trade is no trade.**

---

## 0 · Before you open anything

- [ ] `.streamlit/secrets.toml` has real values under `[database]` (Neon) —
      confirms the whole session's work journals to Postgres.
- [ ] If you want Telegram alerts tonight: `[telegram] bot_token`/`chat_id`
      are filled in (test via the Surprise Awareness page's sidebar button).
- [ ] `streamlit run app.py` — the Checklist (page 00) is the master; every
      other page below feeds one of its 18 checks.
- [ ] Know which session window you're in right now (SAST, UTC+2):

| Session | UTC | SAST | Priority |
|---|---|---|---|
| London Kill Zone | 07:00–09:00 | 09:00–11:00 | 🟢 Prime |
| NY Kill Zone | 12:00–14:00 | 14:00–16:00 | 🟢 Prime |
| London Close | 15:00–17:00 | 17:00–19:00 | 🟡 Secondary |
| Tokyo | 00:00–03:00 | 02:00–05:00 | 🔴 Avoid — dead zone |

If you're in the 🔴 Dead Zone, this is a research/prep session, not an entry
session — work through Steps 1–4 below, set your watchlist, and wait for the
next 🟢/🟡 window to actually pull the trigger (Step 5 onward).

---

## Step 1 — Macro backdrop & risk regime

| Do this | Page |
|---|---|
| One-screen regime + rate bias + fresh setups across the full registry | [01 · Daily Cockpit](../pages/daily_cockpit_tab.py) |
| Morning snapshot — headline KPIs, price table | [02 · Market Overview](../pages/market-overview.py) |
| Any high-impact event in the next few hours? If yes, note it — it may gate your instrument later (Step 6a) | [13 · News Filter](../pages/news-filter.py) |
| **New** — is there an active surprise/event-gate/regime flag on your instrument right now? | [38 · Surprise Awareness](../pages/surprise_tab.py) |

**Output of this step:** a 1-sentence fundamental bias per instrument you're
watching — *"[pair] is [Long/Short/Neutral] because [reason]."* Only look for
setups that agree with it.

---

## Step 2 — Build the shortlist (scan)

| Do this | Page |
|---|---|
| Run **Both** directions; 70%+ = candidate, 90%+ fires an email alert | [03 · Setup Ranker](../pages/setup-ranker.py) |
| A/M/D phase scan (1H × 1 month) | [04 · AMD Scanner](../pages/amd-scanner.py) |
| 6-condition trend scan (50/200 EMA, RSI, MACD, ADX) | [05 · Trend Signals](../pages/trend-signals.py) |
| Trend across timeframes, one grid | [05a · MTF Matrix](../pages/mtf-matrix.py) |
| Donchian 20-day breakout candidates | [06 · 20-Day Breakout](../pages/twenty_day_breakout_tab.py) |
| **New** — structure (BoS/CHoCH) + order blocks + trendlines + MTF bias + a graded trade plan, gold/silver/majors/WTI/BTC | [09a · ABR Toolkit](../pages/abr_toolkit_tab.py) |

**Output:** 1–3 candidate pairs, each with a direction.

---

## Step 3 — Second opinion (synthesize)

| Do this | Page |
|---|---|
| Composite read: Setup Score + Trend Signal + Currency Strength + COT — label, confidence, agreement | [10 · Instrument Predictor](../pages/instrument-predictor.py) |

Drop each Step-2 candidate in here. If confidence is Low or agreement is
split, that candidate is weaker than it looked — don't force it.

**Output:** candidates ranked by how much independent evidence actually agrees.

---

## Step 4 — Filter the day

| Do this | Page |
|---|---|
| Dollar vs Gold inverse cross-check | [11 · DXY vs Gold](../pages/dxy-gold.py) |
| Strongest vs weakest currency — does it match your candidate's direction? | [12 · Currency Strength](../pages/currency-strength.py) |
| Stacked-exposure check before adding correlated risk | [14 · Correlations](../pages/correlations.py) |

**Output:** candidates that survive the daily macro/risk backdrop.

---

## Step 5 — Bias confirmation (weekly → daily → 4H)

Work top to bottom; each level either confirms or vetoes the one above it.

| Level | Page | Rule |
|---|---|---|
| Weekly | [15 · Weekly EMA](../pages/weekly-ema.py) / [16 · Weekly Swing](../pages/weekly-swing.py) | EMA20 > EMA50 + price above both = weekly uptrend. Never fight it. |
| Daily | [17 · Daily Trend](../pages/daily-trend.py) / [18 · Daily MACD](../pages/daily-macd.py) / [19 · Market Structure](../pages/market-structure.py) | Daily should agree with weekly; a daily pullback against a bullish weekly is a buying opportunity, not a reversal. |
| 4H zone | [20 · 4H Confluence Zone](../pages/4H-confluence-zone.py) / [21 · 2/3 Confluence Check](../pages/confluence-checker.py) | Fib + Pivot + EMA20 overlap = the execution zone. Need ≥2 of 3. |

**Output:** one instrument, one direction, price sitting at (or approaching) a
real confluence zone.

---

## Step 6 — Wait for the trigger

| Do this | Page |
|---|---|
| Retrace into the 0.382–0.618 zone + a confirming candle (≥2 of: Stoch cross, RSI reset off 40, BB touch) | [22 · 15M Fib Entry](../pages/15m-fib-entry.py) |

**Do not enter** if price is still slicing through the zone, the daily shows
⚠️ Conflicting, or spread is abnormally wide ahead of news.

### Step 6a — If News Filter or Surprise Awareness flagged an event
Check the event-proximity gate for your instrument on [38 · Surprise
Awareness](../pages/surprise_tab.py) before triggering. A red-folder event
inside the gate window (default ±2–4h) means **stand aside**, not signal —
the geopolitical/data-surprise premium can unwind in hours, not weeks.

---

## Step 7 — Size the risk

| Do this | Page |
|---|---|
| Stop behind a real structure level, ≥1× ATR | [23 · Stop Structure](../pages/stop-structure.py) |
| R:R ≥ 2:1 to TP1 (engine filters at 1.5:1) | [24 · R:R Calculator](../pages/rr-calculator.py) |
| Position size = (Account × Risk%) ÷ (SL pips × pip value); risk 1–2%/trade | [25 · Account Risk](../pages/account-risk.py) |

Risk model, spelled out: **SL = 1.5 × ATR14**, **TP1 = 2R**, **TP2 = 3R**.
Reject anything under 2:1 to TP1.

---

## Step 8 — The gate: Daily Checklist

Open [00 · Daily Checklist](../app.py). Tick every check your Steps 1–7 work
already answered — this is the only place a trade actually gets a verdict.

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

**Verdict is 🟢 GO only when ≥16/18 are checked AND all critical checks
(11–16) pass.** The engine also blocks new entries at 2 losses today
(Postgres-backed daily loss limit) and warns on stacked correlated exposure.

- **🟢 GO** → proceed to Step 9.
- **Anything less** → no trade. Log why in the checklist notes and move on;
  don't force a partial setup because you're already several steps in.

---

## Step 9 — Execute and save

1. Place the trade in your broker (or Deriv MT5 — this is where the [ABR
   Toolkit](../pages/abr_toolkit_tab.py)'s MT5-sourced candles matter: they
   match what you actually see in the platform, not yfinance's futures/spot
   basis offset).
2. **Save Trade Setup** in [00 · Daily Checklist](../app.py) — this is what
   populates the Trade Journal and the daily-loss counter.
3. Management, once in:
   - Stop to **breakeven** at +1R.
   - Take **50% at TP1**, let the rest run to TP2.
   - Exit without hesitation if a 4H closes the wrong side of EMA20, daily
     RSI crosses back through 50 against you, or a weekly closes through the
     pivot.

---

## Step 10 — Review (same evening or next morning)

| Do this | Page |
|---|---|
| Equity curve, win rate vs 66% target, close the trade | [26 · Trade Journal](../pages/trade-journal.py) |
| Did structure stay intact through the hold? | [19 · Market Structure](../pages/market-structure.py) |
| Any journaled forecast maturing? Check the self-scoring track record | [37a · Forecast](../pages/forecast_tab.py) |

---

## If nothing reaches GO tonight

That is a valid, successful session. The system exists to filter out weak
trades — most sessions produce zero or one clean setup, not several. Use the
time instead for the **Weekly & Research Lab** (section 10 of the sidebar):
COT positioning if it's a Monday, [37 · Quant Models
Lab](../pages/quant_models_tab.py) for a stat-arb read, or [37a ·
Forecast](../pages/forecast_tab.py) to journal a longer-horizon view on your
watchlist for next session.

---

## Quick reference — what's new since the last session

- **[09a · ABR Toolkit](../pages/abr_toolkit_tab.py)** — structure/order-block/trendline
  scanner with a graded trade plan; tries your Deriv MT5 terminal first, falls
  back to yfinance.
- **[37a · Forecast](../pages/forecast_tab.py)** — GARCH volatility cone
  (now including a **1-week** horizon) + transparent driver score + narrative,
  self-scored against realized outcomes.
- **[38 · Surprise Awareness](../pages/surprise_tab.py)** — economic surprise
  index + event gate + gold~oil regime inversion, covering the full
  22-instrument registry (not just gold/oil) with live Telegram alerts.
- **Neon Postgres** is now the primary database for the whole app — journal,
  signals, and the three pages above all persist there.
