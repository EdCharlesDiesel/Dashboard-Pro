# OscillatorConsensus

Four oscillators, one overbought/oversold answer.

A single oscillator whipsaws. Four agreeing is worth a look. This indicator plots
RSI, Stochastic %K, Williams %R and CCI on one 0–100 sub-window, and draws a
**consensus histogram** counting how many are simultaneously beyond their band.

## How to read it

**The histogram is the indicator.** The four lines are reference; the bars are the
signal.

| Histogram | Meaning |
|---|---|
| 3–4 bars above the midline | three or four oscillators agree: stretched **up** |
| 3–4 bars below | stretched **down** |
| 1–2 either way | noise — ignore it |

### The coloured bands

| Line | Level | Meaning |
|---|---|---|
| **Green**, solid, 2px | 30 | **Oversold — the buy side** |
| **Red**, solid, 2px | 70 | **Overbought — the sell side** |
| Grey, dotted, 1px | 50 | The histogram's zero. A reference, not a decision. |

The midline is deliberately quiet. Drawing it as loudly as the other two would
make the pane read as three equal decisions instead of one.

**A band being touched is a condition, not a trigger.** Markets stay oversold for
weeks in a downtrend — which is exactly why the histogram exists. Green *with
three or four bars below the midline* is the read worth acting on; green on its
own is just a low number.

## Buffers

Stable and public — signal code reads these by index via `iCustom`. New plots are
**appended**, never renumbered.

| # | Buffer | Range | Notes |
|---|---|---|---|
| 0 | RSI | 0–100 | raw |
| 1 | Stochastic %K | 0–100 | raw, main buffer (not %D) |
| 2 | Williams %R | 0–100 | plotted as raw `%R + 100` |
| 3 | CCI | 0–100 | plotted as `50 + CCI/4`, clipped |
| 4 | histogram baseline | 50 | constant |
| 5 | histogram top | 0–100 | `50 + count × 12.5` |
| **6** | **consensus count** | **−4…+4** | **read this one from `iCustom`** |

Buffer 6 is `INDICATOR_CALCULATIONS` — it is not drawn. It carries the honest
integer so nothing downstream has to undo the display scaling.

### Why the histogram needs two buffers

The count is −4…+4 and the pane is 0–100. Drawn raw with `DRAW_HISTOGRAM` it
would be a 4%-tall smudge, and `DRAW_HISTOGRAM` always draws *from zero*, so it
cannot simply be shifted up. `DRAW_HISTOGRAM2` draws between two buffers instead:
buffer 4 pins a baseline at the pane's midpoint, buffer 5 carries the scaled
value. −4…+4 maps onto 0…100 exactly, and the bars grow up or down from 50 the
way a signed count should.

## Bands — and why they differ from the app

The indicator uses the **classic mean-reversion bands**:

| Oscillator | Oversold | Overbought |
|---|---|---|
| RSI (14) | 30 | 70 |
| Stochastic %K (14, 3, 3) | 20 | 80 |
| Williams %R (14) | −80 | −20 |
| CCI (20) | −100 | +100 |

`src/core/config.py` uses **RSI 40/60**, and this indicator deliberately does not.
That 40/60 is a trend-**pullback** band — buy the dip to 40 in an uptrend. This
indicator answers "is this stretched", which is a mean-reversion question.

Measured 2026-08-15: EUR/USD sat at RSI 61.3 — "overbought" on 60, mid-range on
70. Both readings are defensible; they answer different questions. All eight
bands are inputs, so change them here rather than in the app.

## Normalisation

So four different scales can share one pane:

- **RSI**, **Stochastic** — already 0–100.
- **Williams %R** — natively −100…0, plotted as `%R + 100`.
- **CCI** — unbounded, plotted as `50 + CCI/4` clipped to 0–100, so the classic
  ±100 lands on 75/25.

**The count is computed from raw values, never the normalised ones.** Clipping CCI
for display must not cap the count, or a CCI of 400 would score the same as 101
— and worse, an extreme reading would vanish from the histogram just as it became
interesting. `tests/test_oscillator_consensus_parity.py` pins this.

## Verified

Compiled **0 errors, 0 warnings** (MetaEditor, 2026-08-15). Installed to
`MQL5\Indicators\` alongside `FiboRibbon.ex5` and `BiasedPivots.ex5`.

Parity targets, daily bars, 2026-08-15:

| Pair | RSI | Stoch %K | W %R | CCI | Consensus | Histogram top |
|---|---|---|---|---|---|---|
| XAU/USD | 66.6 | 90.3 | −4.0 | 106.5 | **+3** | 87.5 |
| USD/CAD | 30.2 | 11.5 | −97.1 | −122.9 | **−3** | 12.5 |
| USD/JPY | 43.1 | 56.2 | −27.4 | −21.6 | **0** | 50.0 |

Note XAU/USD: RSI at 66.6 is *below* 70, so it does not count — three of four,
not four. That is the point of the consensus rather than a single oscillator.

## MT5 / Python equivalence

Checked against MT5's built-ins so the chart and the app cannot disagree:

- `cci()` uses typical price `(H+L+C)/3` with the 0.015 scale → matches
  `iCCI(..., PRICE_TYPICAL)`.
- `williams_r()` is `-100*(HH-C)/(HH-LL)` → matches `iWPR`.
- `stochastic(k, d, smooth)` takes fast %K then smooths by `smooth` → matches
  `iStochastic(k, d, slowing, MODE_SMA, STO_LOWHIGH)` **buffer 0**. Buffer 1 is
  %D and is not what this reads.

## No alerts, by design

The desk's alerting lives in `src/services/alert_service.py`. A second channel
firing from the terminal would double every notification.
