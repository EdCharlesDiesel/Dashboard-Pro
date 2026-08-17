# OscillatorConsensus.mq5 — One Overbought/Oversold Read — Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** An MT5 sub-window indicator that answers "is this stretched?" with the **agreement of four oscillators**, not one. RSI, Stochastic %K, Williams %R and CCI all already exist in the Python side (`src/indicators/leading.py`) and none of them is on the chart. A single oscillator whipsaws; four agreeing is a signal worth a look.

**Architecture:** A standalone `mt5/OscillatorConsensus.mq5` in the house style — one sub-window, stable buffer order, `iCustom`-readable. It computes nothing exotic: it wraps MT5's own `iRSI`, `iStochastic`, `iWPR` and `iCCI` handles, normalises each to a 0–100 scale, and plots (a) the four normalised lines faintly and (b) a **consensus histogram**: how many of the four are simultaneously beyond their band, signed. Nothing in the Python app changes.

**Tech Stack:** MQL5 (MetaTrader 5 build on this machine), MetaEditor CLI at `C:\Program Files\MetaTrader 5 EXNESS\metaeditor64.exe`, Python 3.14 venv for the parity check, Docker Compose unaffected.

**Spec:** `mt5/FiboRibbon.mq5` and `mt5/BiasedPivots.mq5` are the house conventions this follows — buffer order documented in the header and never renumbered, the "why" recorded above the code, inputs exposed rather than hardcoded.

## Global Constraints

- Never commit. Make changes only; the repository owner reviews and commits.
- **A plan gets its own bump too.** `VERSION` read **1.10.11**, so this plan takes **1.10.12** — the patch, plus one. Each task then bumps to whatever `VERSION` reads at the moment it completes, plus one. Never a minor bump, never a reserved block, never a skipped number.
- Run tests as `PYTHONIOENCODING=utf-8 python -m pytest`.
- Return complete implementations — no TODO comments, no placeholder code.
- **Buffer order is a public API.** Signal code reads buffers by index via `iCustom`. Append new plots; never renumber existing ones.
- The `.mq5` source lives in `mt5/` (tracked). Installing copies it to the terminal's `MQL5\Indicators\`; the `.ex5` is a build artifact and is not the source of truth.
- Compile with **0 errors, 0 warnings** before calling a task done, the same bar `FiboRibbon.mq5` cleared.

---

## Measured starting state (2026-08-15, v1.10.11)

**The Python side already has every oscillator, and nothing plots them:**

| Indicator | Location |
|---|---|
| RSI | `src/indicators/technical.py:24`, `src/core/signals.py:294` |
| Stochastic %K/%D | `src/indicators/leading.py:28` |
| Williams %R | `src/indicators/leading.py:39` |
| CCI | `src/indicators/leading.py:46` |
| DeMarker | `src/indicators/leading.py:55` |
| RSI divergence | `src/indicators/leading.py:154` |

**The threshold problem, and the decision this plan makes.**
`src/core/config.py` sets:

```python
rsi_os: float = 40.0        # classic mean-reversion band is 30
rsi_ob: float = 60.0        # classic is 70
stoch_os: float = 25.0
stoch_ob: float = 75.0
```

RSI 40/60 is a **trend-continuation** band — buy the pullback to 40 in an uptrend. It is not an overbought/oversold band, and only `signals.py` consumes it.

**Decision: this indicator uses the classic mean-reversion bands (RSI 30/70, Stoch 20/80, Williams −80/−20, CCI ±100), exposed as inputs defaulting to those values.** Rationale: the indicator answers "is this stretched", which is a mean-reversion question; borrowing a trend-pullback band would make it say "overbought" at RSI 61 on a pair that is simply trending. Measured on the live universe 2026-08-15: EUR/USD sat at RSI 61.3 — "overbought" under the app's 60, mid-range under 70. **Do not change `config.py`.** That band belongs to the signal engine and changing it is a separate decision with its own blast radius.

**Live readings the finished indicator must reproduce** (daily, via the spine, 2026-08-15):

| pair | RSI14 | Stoch %K | Will %R | CCI20 | expected consensus |
|---|---|---|---|---|---|
| XAU/USD | 66.6 | 90.3 | −4.0 | 106.5 | **+3** (Stoch, W%R, CCI) |
| USD/CAD | 30.2 | 11.5 | −97.1 | −122.9 | **−3** (Stoch, W%R, CCI) |
| USD/JPY | 43.1 | 56.2 | −27.4 | −21.6 | **0** |

These are the parity targets for Task 2.

**House conventions to follow** (from `mt5/FiboRibbon.mq5`):
- A header block explaining *why*, not just *what*, including the buffer map.
- `BUFFER ORDER (stable — signal code reads these by index via iCustom)`.
- Inputs for every period and threshold; nothing magic inline.

**Terminal paths on this machine:**
- Indicators: `C:\Users\LENOVO\AppData\Roaming\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\MQL5\Indicators\` (verified 2026-08-15 — holds `FiboRibbon.ex5`, `BiasedPivots.ex5`)
- Compiler: `C:\Program Files\MetaTrader 5 EXNESS\metaeditor64.exe`

---

## File structure

- **Create** `mt5/OscillatorConsensus.mq5` — the indicator. Sub-window, 5 buffers.
- **Create** `mt5/OscillatorConsensus.md` — what it plots and how to read it, matching `mt5/FiboRibbon.md`.
- **Create** `tests/test_oscillator_consensus_parity.py` — proves the Python and MQL5 definitions agree on the *consensus rule*, so the chart and the app cannot drift.
- **Unchanged:** `src/core/config.py`, and every Python page. This adds a chart tool, not an app feature.

---

### Task 1: The indicator

**Files:**
- Create: `mt5/OscillatorConsensus.mq5`

**Interfaces (the public API — buffer order is stable):**

```
0 = RSI                       (0-100, raw)
1 = Stochastic %K             (0-100, raw)
2 = Williams %R normalised    (raw W%R is -100..0; plotted as W%R + 100)
3 = CCI normalised            (raw is unbounded; plotted as 50 + CCI/4, clipped 0-100)
4 = consensus histogram       (-4..+4, signed count of oscillators beyond their band)
```

- [x] **Step 1: Write the indicator**

`mt5/OscillatorConsensus.mq5`. Header first, in the house style — state the buffer map, the normalisation, and why the bands differ from `config.py`.

```mql5
//+------------------------------------------------------------------+
//|                                        OscillatorConsensus.mq5   |
//|   Four oscillators, one overbought/oversold answer.              |
//+------------------------------------------------------------------+
//  A single oscillator whipsaws. Four agreeing is worth a look. This plots
//  RSI, Stochastic %K, Williams %R and CCI on one 0-100 scale, plus a
//  histogram counting how many are simultaneously beyond their band.
//
//  NORMALISATION (so four different scales can share one sub-window)
//    RSI          0..100 already.
//    Stochastic   0..100 already.
//    Williams %R  natively -100..0  -> plotted as W%R + 100.
//    CCI          unbounded         -> plotted as 50 + CCI/4, clipped to 0..100,
//                                      so the classic +/-100 lands on 75/25.
//  The histogram is computed from the RAW values, never the normalised ones:
//  clipping CCI would otherwise silently cap the count.
//
//  BANDS -- deliberately the classic mean-reversion levels, NOT the 40/60 in
//  the Python app's config.py. That 40/60 is a trend-PULLBACK band (buy the dip
//  to 40 in an uptrend); this indicator answers "is this stretched", which is a
//  mean-reversion question. Measured 2026-08-15: EUR/USD at RSI 61.3 reads
//  "overbought" on 60 and "mid-range" on 70. Both are defensible; they are
//  answers to different questions. All four bands are inputs -- change them
//  here, not in the app.
//
//  BUFFER ORDER (stable — signal code reads these by index via iCustom):
//    0 = RSI   1 = Stoch %K   2 = Williams %R + 100   3 = CCI normalised
//    4 = consensus histogram (-4..+4)
//  New plots are APPENDED, never renumbered: consumers read by index.
//
//  READING IT: the histogram is the indicator. +3 or +4 means three or four
//  oscillators agree the market is stretched up; -3 or -4, stretched down.
//  Anything between +/-2 is noise and is drawn dim on purpose.
//+------------------------------------------------------------------+
#property copyright "Dashboard-Pro"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 5
#property indicator_plots   5

#property indicator_minimum -10
#property indicator_maximum 110

input int    RSIPeriod        = 14;    // RSI period
input int    StochKPeriod     = 14;    // Stochastic %K period
input int    StochDPeriod     = 3;     // Stochastic %D (slowing)
input int    WPRPeriod        = 14;    // Williams %R period
input int    CCIPeriod        = 20;    // CCI period

input double RSIOverbought    = 70.0;  // RSI overbought (classic; app uses 60)
input double RSIOversold      = 30.0;  // RSI oversold  (classic; app uses 40)
input double StochOverbought  = 80.0;  // Stochastic overbought
input double StochOversold    = 20.0;  // Stochastic oversold
input double WPROverbought    = -20.0; // Williams %R overbought (raw scale)
input double WPROversold      = -80.0; // Williams %R oversold  (raw scale)
input double CCIOverbought    = 100.0; // CCI overbought
input double CCIOversold      = -100.0;// CCI oversold

double BufRSI[], BufStoch[], BufWPR[], BufCCI[], BufConsensus[];
int hRSI, hStoch, hWPR, hCCI;

int OnInit()
  {
   SetIndexBuffer(0, BufRSI,       INDICATOR_DATA);
   SetIndexBuffer(1, BufStoch,     INDICATOR_DATA);
   SetIndexBuffer(2, BufWPR,       INDICATOR_DATA);
   SetIndexBuffer(3, BufCCI,       INDICATOR_DATA);
   SetIndexBuffer(4, BufConsensus, INDICATOR_DATA);

   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_LINE);
   PlotIndexSetInteger(4, PLOT_DRAW_TYPE, DRAW_HISTOGRAM);

   PlotIndexSetString(0, PLOT_LABEL, "RSI");
   PlotIndexSetString(1, PLOT_LABEL, "Stoch %K");
   PlotIndexSetString(2, PLOT_LABEL, "Williams %R");
   PlotIndexSetString(3, PLOT_LABEL, "CCI");
   PlotIndexSetString(4, PLOT_LABEL, "Consensus");

   PlotIndexSetInteger(4, PLOT_LINE_WIDTH, 3);

   hRSI   = iRSI(NULL, 0, RSIPeriod, PRICE_CLOSE);
   hStoch = iStochastic(NULL, 0, StochKPeriod, StochDPeriod, 3,
                        MODE_SMA, STO_LOWHIGH);
   hWPR   = iWPR(NULL, 0, WPRPeriod);
   hCCI   = iCCI(NULL, 0, CCIPeriod, PRICE_TYPICAL);

   if(hRSI == INVALID_HANDLE || hStoch == INVALID_HANDLE ||
      hWPR == INVALID_HANDLE || hCCI == INVALID_HANDLE)
     {
      Print("OscillatorConsensus: could not create an indicator handle");
      return(INIT_FAILED);
     }

   IndicatorSetString(INDICATOR_SHORTNAME, "OscillatorConsensus");
   IndicatorSetInteger(INDICATOR_LEVELS, 2);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 0, RSIOverbought);
   IndicatorSetDouble(INDICATOR_LEVELVALUE, 1, RSIOversold);
   return(INIT_SUCCEEDED);
  }

//  CCI is unbounded; squeeze it onto the shared 0-100 axis so the classic
//  +/-100 lands on 75/25. Only the PLOT is clipped -- the consensus count
//  below reads the raw value, or a CCI of 400 would count the same as 101.
double NormaliseCCI(const double raw)
  {
   double v = 50.0 + raw / 4.0;
   if(v > 100.0) v = 100.0;
   if(v <   0.0) v =   0.0;
   return(v);
  }

int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
  {
   int need = MathMax(MathMax(RSIPeriod, StochKPeriod),
                      MathMax(WPRPeriod, CCIPeriod)) + 1;
   if(rates_total < need)
      return(0);

   int start = (prev_calculated > 1) ? prev_calculated - 1 : 0;
   int count = rates_total - start;

   double rsi[], stoch[], wpr[], cci[];
   if(CopyBuffer(hRSI,   0, 0, rates_total, rsi)   <= 0) return(0);
   if(CopyBuffer(hStoch, 0, 0, rates_total, stoch) <= 0) return(0);
   if(CopyBuffer(hWPR,   0, 0, rates_total, wpr)   <= 0) return(0);
   if(CopyBuffer(hCCI,   0, 0, rates_total, cci)   <= 0) return(0);

   for(int i = start; i < rates_total; i++)
     {
      BufRSI[i]   = rsi[i];
      BufStoch[i] = stoch[i];
      BufWPR[i]   = wpr[i] + 100.0;      // -100..0 -> 0..100
      BufCCI[i]   = NormaliseCCI(cci[i]);

      int score = 0;
      if(rsi[i]   > RSIOverbought)   score++;
      if(stoch[i] > StochOverbought) score++;
      if(wpr[i]   > WPROverbought)   score++;
      if(cci[i]   > CCIOverbought)   score++;

      if(rsi[i]   < RSIOversold)     score--;
      if(stoch[i] < StochOversold)   score--;
      if(wpr[i]   < WPROversold)     score--;
      if(cci[i]   < CCIOversold)     score--;

      BufConsensus[i] = (double)score;
     }
   return(rates_total);
  }
```

- [x] **Step 2: Compile it — 0 errors, 0 warnings**

```bash
"/c/Program Files/MetaTrader 5 EXNESS/metaeditor64.exe" /compile:"C:\x\Dashboard-Pro\mt5\OscillatorConsensus.mq5" /log
```
MetaEditor writes a `.log` beside the source; read it. Anything other than `0 errors, 0 warnings` is a failure — `FiboRibbon.mq5` cleared that bar and so must this.

- [x] **Step 3: Install to the terminal**

Copy **both** the `.mq5` and the compiled `.ex5` into:

```
C:\Users\LENOVO\AppData\Roaming\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\MQL5\Indicators\ 
```

Verified 2026-08-15 — that is the folder already holding `FiboRibbon.ex5` and `BiasedPivots.ex5`. Re-list it rather than trusting this line if the terminal has been reinstalled.

- [x] **Step 4: Look at it on a chart**

Attach to **XAU/USD daily**. Against the 2026-08-15 readings above the histogram should print **+3**. Attach to **USD/CAD daily**: **−3**. Attach to **USD/JPY daily**: **0**.

Screenshot each. If a value disagrees, do not adjust the indicator to match — find out which side is wrong first.

**Verified 2026-08-15, so do not chase these:** Python's `cci()` uses typical price `(H+L+C)/3` with the 0.015 scale, matching `iCCI(..., PRICE_TYPICAL)`. Python's `williams_r()` is the standard `-100*(HH-C)/(HH-LL)`, matching `iWPR`.

**The real divergence risk is Stochastic smoothing.** Python's `stochastic(k=14, d=3, smooth=3)` takes fast %K, smooths by 3 for slow %K, then smooths again by 3 for %D. The MQL5 call is `iStochastic(NULL, 0, 14, 3, 3, MODE_SMA, STO_LOWHIGH)`, whose 2nd and 3rd arguments are %D period and slowing. Confirm MT5 buffer 0 (main/%K) is what corresponds to Python's `slow_k` before deciding either side is wrong — buffer 0 is the one this indicator reads.

- [x] **Step 5: Bump, `verify_deploy.py` (the app is untouched but the version must still move), show the owner the diff. Do not commit.**

---

### Task 2: Parity — the chart and the app must not drift

The whole point of the house convention is that a signal can read this by `iCustom` and get the same answer the Python app would. That only holds if both define the consensus identically.

**Files:**
- Create: `tests/test_oscillator_consensus_parity.py`
- Create: `mt5/OscillatorConsensus.md`

- [x] **Step 1: Write the parity test**

It cannot run MQL5, so it pins the *rule* the MQL5 implements, against the Python oscillators on real spine data:

```python
"""The chart indicator and the app must agree on 'stretched'.

`OscillatorConsensus.mq5` counts how many of RSI/Stoch/W%R/CCI are beyond
their band. This pins the same rule in Python against the same bands, so a
future edit to either side that changes the answer fails here rather than
silently putting a different number on the chart from the one the signal
engine sees.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.indicators.leading import cci, stochastic, williams_r
from src.indicators.technical import TechnicalIndicators as TI

# The MQL5 inputs, mirrored. Classic mean-reversion bands -- deliberately NOT
# config.py's 40/60, which is a trend-pullback band. See the .mq5 header.
BANDS = {"rsi": (30.0, 70.0), "stoch": (20.0, 80.0),
         "wpr": (-80.0, -20.0), "cci": (-100.0, 100.0)}


def consensus(df: pd.DataFrame) -> int:
    """Signed count of oscillators beyond their band, on the last bar."""
    ...  # implement to mirror the MQL5 loop exactly


class TestConsensus:
    def test_all_four_stretched_up_is_plus_four(self): ...
    def test_all_four_stretched_down_is_minus_four(self): ...
    def test_mid_range_is_zero(self): ...
    def test_raw_cci_is_counted_not_the_clipped_plot(self):
        # A CCI of 400 must count once, exactly like 101 -- but the plot clips
        # to 100. Counting the clipped value would make an extreme reading
        # vanish from the histogram.
        ...
```

**Fill every `...` when implementing** — this block is the shape, not the deliverable.

- [x] **Step 2: Run it against the three live parity targets** (XAU/USD +3, USD/CAD −3, USD/JPY 0) using `daily_ohlc`. Record the numbers in the `.md`.

- [x] **Step 3: Write `mt5/OscillatorConsensus.md`** in the style of `mt5/FiboRibbon.md`: what it plots, the buffer map, how to read the histogram, and the band decision with its reasoning.

- [x] **Step 4: Full suite, bump, show the diff. Do not commit.**

---

## Out of scope, deliberately

- **Changing `config.py`'s 40/60.** That band belongs to the signal engine; altering it changes every scored signal and needs its own plan.
- **A Python OB/OS page.** That was option 2 of the three offered; this plan builds the chart indicator only.
- **Divergence plotting.** `leading.py:154` already computes RSI divergence and it is the natural next buffer — appended as buffer 5, in a later plan, so the buffer order stays stable.
- **Alerts.** No `Alert()` calls: the desk's alerting lives in `alert_service.py`, and a second channel firing from the terminal would double every notification.

## Verification for the whole plan

- [x] `metaeditor64.exe /compile` reports **0 errors, 0 warnings**.
- [x] `.mq5` and `.ex5` both present in the terminal's `MQL5\Indicators\`.
- [x] Screenshots: XAU/USD **+3**, USD/CAD **−3**, USD/JPY **0** on the daily.
- [x] `tests/test_oscillator_consensus_parity.py` green, reproducing those three.
- [x] Full suite: no failures beyond the 3 known.
- [x] `mt5/OscillatorConsensus.md` written, buffer map matching the source.

---

Module map: [[Architecture]] · Docs index: [[README]]
