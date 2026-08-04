# FiboRibbon — the desk's chart MA stack, as an indicator

`FiboRibbon.mq5` reproduces the moving-average stack that is actually on the
trader's MT5 charts. It was **read out of the terminal's saved chart profile**,
not reconstructed from memory:

```
MQL5\Profiles\Charts\(usd) AUDUSD\chart02.chr
```

The identical stack sits on `(usd) NZDUSD`, so this is the standard template
rather than a one-off setting on a single symbol.

| Buffer | Plot | Period | Method | Applied price |
| --- | --- | --- | --- | --- |
| 0 | EMA200 | 200 | EMA | Close |
| 1 | EMA55 | 55 | EMA | Close |
| 2 | EMA21 | 21 | EMA | Close |
| 3 | EMA5 | 5 | EMA | Close |
| 4 | SMA8 | 8 | **SMA** | **Open** |
| 5 | Bull cross arrow | — | — | — |
| 6 | Bear cross arrow | — | — | — |

A Fibonacci-spaced EMA ribbon (5 / 21 / 55 / 200) plus a fast SMA of the
**open**, and the gated cross signal built on top of them.

## The cross signal (buffers 5 / 6)

```
bull = EMA5 crosses UP through SMA8(open)
       AND EMA21 > EMA55 > EMA200
       AND close > EMA21
bear = the mirror
```

**The gate is the point.** A bare 5/8 cross fires constantly in chop; the ribbon
ordering is what separates a fast cross going *with* the larger trend from one
fighting it. Measured live on 2026-08-04 across the 22-instrument registry: one
cross fired (GBP/ZAR, bearish), the ribbon disagreed, and the gate rejected it —
0 signals from 22 instruments. A quiet scanner is the gate working, not a broken
one. `InpRequireStack=false` disables it, which is for measuring the gate's worth,
not for trading.

**Closed bars only.** The newest bar is skipped in `OnCalculate`, and the Python
twin reads `[-2]` rather than `[-1]`. An arrow that appears intrabar and vanishes
by the close back-tests as a winner and trades as a loss.

## The Python twin

`src/core/fibo_ribbon.py` implements the identical logic for the dashboard —
same periods, same applied prices, same gate, same closed-bar rule — so the chart
and the scanner cannot disagree about whether a signal fired. It is scanned across
the universe by `pages/fibo-ribbon.py` (nav code **RIBN**) and persisted under
`source='fibo_ribbon'`, swept like every other signal page.

## The mixed applied price is deliberate

Four EMAs read the close; the 8-period SMA reads the **open**. That came
straight out of the chart file (`apply=2` = `PRICE_OPEN`, `method=0` = SMA) and
is not a typo to tidy up. A fast EMA of the close crossing a fast SMA of the
open is the trigger pair, and it only behaves as intended while the two applied
prices stay different — align them and the cross degenerates into noise around
a single price series.

## Buffer order is a contract

Signal code reads these by **index** through `iCustom()`, so inserting a plot in
the middle would silently repoint every consumer:

```mql5
int h = iCustom(_Symbol, PERIOD_CURRENT, "FiboRibbon");
double bull[];
CopyBuffer(h, 5, 0, 3, bull);           // buffer 5 = bull cross arrow
// a fired signal is a price; no signal is EMPTY_VALUE
bool fired = (bull[1] != EMPTY_VALUE);  // [1] = last closed bar
```

Append new plots at the end; never renumber.

## Install

Copy to the terminal's indicator folder and compile in MetaEditor (F7):

```
%APPDATA%\MetaQuotes\Terminal\53785E099C927DB68A545C249CDBCE06\MQL5\Indicators\
```

That hash is the **Exness MT5** terminal (`C:\Program Files\MetaTrader 5 EXNESS`).
`46A834A4BD020127C05B0DA2582F8F5C` is a TradersWay **MT4** install — this will
not compile there.

## What is deliberately not in here

Two indicators sit on the same charts and are **not** reproduced:

- **`PivotsNew.ex5`** — compiled only; there is no `.mq5` source on this
  machine, so it cannot be reproduced faithfully. Inventing a pivot formula
  would draw levels that disagree with the chart, which is worse than having no
  pivots at all. `BiasedPivots.mq5` *does* ship with source and is already on
  the live EUR/CAD chart — the obvious candidate if pivots are wanted.
- **`Watermark.ex5`** — cosmetic, computes nothing.

## How the chart stack was detected

MT5 `.chr` chart files are UTF-16LE INI text, so applied indicators and their
parameters can be read straight off disk. This is the only route short of
running MQL5 inside the terminal: the `MetaTrader5` Python package exposes **no
chart or indicator API at all** (269 symbols in its surface; the only "chart"
matches are `SYMBOL_CHART_MODE_BID/LAST`, which describe a symbol's price basis,
not applied indicators).

**Caveat worth repeating:** the terminal only flushes chart state to disk when
the profile is saved or the terminal closes, so what you read is last-saved
state, never live. Check the file's mtime before trusting it — the
`(usd) AUDUSD` profile was last written in **2021**, and the profile that was
live when this was written (`(jpy) AUDJPY`) contained no AUD/USD window at all.
