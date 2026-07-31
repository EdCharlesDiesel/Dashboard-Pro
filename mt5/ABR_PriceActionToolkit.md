# ABR Price Action Toolkit — MT5 indicator

`ABR_PriceActionToolkit.mq5` is the MetaTrader 5 port of the dashboard's
**🧱 ABR Toolkit** page (`pages/abr_toolkit_tab.py :: run_engine`). Same swing
rule, same BoS/CHoCH state machine, same order-block lifecycle, same quality
formula — so the chart in front of you and the journal in the app can't tell
you two different stories.

**It is an indicator, not an EA. It never places, modifies or closes an order.**

---

## What it draws

| Element | Meaning |
|---|---|
| Small arrows | Confirmed swing highs / lows (strict unique extreme over ±`SwingLength` bars) |
| Dashed level + label | **BoS** — break of structure, the trend continues |
| Solid level + label | **CHoCH** — change of character, the trend just flipped |
| `Trend Strong` / `Trend Weak` | Penetration ≥ `StrongBreakATR` × ATR = Strong |
| Shaded boxes | Live **order blocks** — the last opposing candle before the impulse |
| Dotted rays | Auto trendlines through the last two swings, projected forward |
| Horizontal lines | Entry / SL / TP1–3 for the current plan |
| Corner panel | Signal, grade, HTF bias, live OB count, lot size, risk |

An order block stops being drawn once it is **mitigated** (price closes back
through it), once it ages past `OBMaxAgeBars`, or when it falls outside the
newest `MaxLiveOBs` on its side.

## The quality score

Identical to the dashboard's formula:

```
quality = 25
        + 15 x (higher timeframes whose EMA bias agrees with the signal)
        + 15  if a live order block supports the direction
        + 15  if a trendline broke within the last TLLookbackBars

A+ >= 85    A >= 70    B >= 55    C >= 40    D otherwise
```

Maximum is 100 (25 + 45 + 15 + 15). A bare structural break with no HTF
agreement and no order block scores **25 → grade D** — which is the point: the
score is telling you structure fired without confluence.

## Install

1. In MT5: **File → Open Data Folder**, copy `ABR_PriceActionToolkit.mq5` into
   `MQL5\Indicators\`.
2. Open it in MetaEditor and press **Compile / F7** (or **Navigator →
   Indicators → right-click → Refresh**).
3. Drag **ABR_PriceActionToolkit** onto a chart. No DLLs, no AutoTrading — it
   only reads bars and draws objects.

## Inputs

### Structure

| Input | Default | Meaning |
|---|---|---|
| `SwingLength` | `5` | Bars each side of a pivot. Higher = fewer, larger swings. |
| `MaxBars` | `1500` | Bars analysed (0 = all). Lower it if the chart feels heavy. |
| `ATRPeriod` | `14` | Drives break strength and the SL buffer. |
| `StrongBreakATR` | `0.5` | Break ≥ this × ATR is labelled *Trend Strong*. |

### Order blocks

| Input | Default | Meaning |
|---|---|---|
| `OBSearchBars` | `15` | How far back to look for the origin candle. |
| `OBMaxAgeBars` | `300` | Auto-expire an untouched block after this many bars. |
| `MaxLiveOBs` | `2` | Live blocks kept per side. |

### Multi-timeframe bias

| Input | Default | Meaning |
|---|---|---|
| `HTF1` / `HTF2` / `HTF3` | H4 / D1 / W1 | Set any to `PERIOD_CURRENT` to disable that leg. |
| `MAFast` / `MASlow` | 21 / 50 | EMA pair. Bias is bullish when fast > slow **and** price > fast. |

Each agreeing timeframe adds 15 points to the quality score, so disabling a leg
lowers the achievable maximum — deliberate, not a bug.

### Trade plan

| Input | Default | Meaning |
|---|---|---|
| `SLBufferATR` | `0.5` | SL sits this × ATR beyond the structure level. |
| `TP1R` / `TP2R` / `TP3R` | 1.0 / 2.0 / 2.83 | R multiples (same as the dashboard). |
| `RiskPerTrade` | `30.0` | Risk in **account currency**. |

## Reading the panel

```
ABR PRICE ACTION TOOLKIT
AUDJPYi   H1
Signal:  BUY    Grade A  (70/100)
Structure: UP     EMA bias: UP
HTF:  H4:UP  D1:UP  W1:flat  (2 aligned)
Live OBs: 2 bull / 0 bear    TL break: no
Entry: 114.506
SL:    113.982   (0.524)
TP1:   115.030
Lots:  0.34    risk 30.00 USD
```

`Structure` is the BoS/CHoCH state machine; `EMA bias` is the 21/50 read. When
they disagree, structure has turned before the averages have — earlier, and
lower confidence.

## Using the plan from an EA

Signal, entry, stops and targets are published on `DRAW_NONE` buffers, so an EA
can read them without duplicating the engine:

```mql5
double sig[1], entry[1], sl[1], tp1[1], q[1];
int h = iCustom(_Symbol, PERIOD_H1, "ABR_PriceActionToolkit");
CopyBuffer(h, 0, 0, 1, sig);    // 0 = signal (+1 buy / -1 sell / 0 none)
CopyBuffer(h, 1, 0, 1, entry);  // 1 = entry
CopyBuffer(h, 2, 0, 1, sl);     // 2 = stop loss
CopyBuffer(h, 3, 0, 1, tp1);    // 3 = TP1   (4 = TP2, 5 = TP3)
CopyBuffer(h, 6, 0, 1, q);      // 6 = quality 0-100
```

Every bar except the most recent returns `EMPTY_VALUE` — the buffers publish the
*current* plan, not a history of past ones.

## One deliberate difference from the Python page

**Lot sizing.** The dashboard uses a static `point_value` table keyed by
instrument. This indicator reads the broker's own contract spec for whatever
symbol the chart is on:

```
value per 1.0 price move per lot = SYMBOL_TRADE_TICK_VALUE / SYMBOL_TRADE_TICK_SIZE
lots = RiskPerTrade / (SL distance x that value)
```

then snaps to `SYMBOL_VOLUME_STEP`, clamps to the broker's min/max, and **rounds
down** so a rounding step can never risk more than you asked for. That is why
suffixed broker symbols (`AUDJPYi`, `XAUUSDm`, …) size correctly with nothing to
configure.

## Caveats worth knowing before you trade the numbers

- **The newest bar is still forming.** Structure confirms on close, so a break
  printed on the live candle can un-print. Treat the newest signal as
  provisional until that bar closes.
- **It recalculates on each new bar**, not tick by tick. The structure detector
  is path-dependent, so a full pass over the window is both simpler and safer
  than patching state incrementally.
- **`Signal: NONE` is information.** It means no BoS or CHoCH fired inside the
  analysed window — there is no structural trade, which is a reason to stay out
  rather than a failure of the tool.
- **This is MQL5 and will not compile in MT4.** MQL4 has a different API
  (`OnCalculate` signature, indicator handles, `SymbolInfoDouble` constants). If
  your terminal is MT4, the same engine can be ported to `.mq4`.

## Matching the dashboard exactly

Both implementations share the same defaults (`SwingLength 5`, `OBSearchBars
15`, `OBMaxAgeBars 300`, `MaxLiveOBs 2`, `TLLookbackBars 10`, `SLBufferATR 0.5`,
R multiples 1.0 / 2.0 / 2.83, EMA 21/50). The remaining sources of small
differences are **data, not maths**: the dashboard pulls yfinance candles
(futures/spot basis, different session boundaries) unless MetaTrader5 is
installed on the machine running it, while the indicator uses your broker's own
bars. Structure and grades normally agree; exact prices can differ by the basis.
