> **This folder holds two MT5 indicators.** This file documents the Fixed Range
> Activity Profile; see [`ABR_PriceActionToolkit.md`](ABR_PriceActionToolkit.md)
> for the ABR Price Action Toolkit (structure, order blocks, graded trade plan).

# Fixed Range Activity Profile — MT5 indicator

`FixedRangeActivityProfile.mq5` is the MetaTrader 5 port of the dashboard's
**Fixed Range VP** (`src/core/volume_profile.py`, rendered on the AMD Scanner
page). Same anchoring, same value-area maths, same honesty caveats — drawn on
an MT5 chart instead of Plotly.

It answers one question about a *completed* session: **at which prices did that
range actually trade?** The answer is three levels that stay useful long after
the range closes:

| Level | Meaning |
|---|---|
| **POC** | Point of Control — the highest-activity price bin. The range's fair value; price tends to rotate back to it. |
| **VAH / VAL** | The contiguous band around the POC holding 70% of the range's activity. Outside it is where the range was *rejected*, not accepted. |

## Why the range is a session break, not a drag

TradingView's FRVP profiles whatever you drag a box over, so two traders never
get the same POC. This one anchors to a **session break** — the previous FX day
(rolling at 21:00 UTC ≈ the 17:00 New York close) or the previous occurrence of
a named kill zone. The levels are reproducible, and they match what the
dashboard shows for the same instrument.

**The session still forming is never profiled.** A half-formed session's POC
moves on every tick, which is the opposite of a level you can trade.

## Install

1. In MT5: **File → Open Data Folder**. Copy `FixedRangeActivityProfile.mq5`
   into `MQL5\Indicators\`.
2. Open it in MetaEditor and press **Compile / F7** (or **Navigator →
   Indicators → right-click → Refresh**).
3. Drag **FixedRangeActivityProfile** onto a chart. No DLLs, no AutoTrading —
   it only reads bars and draws objects.

Defaults are tuned for a **H1 chart profiling yesterday's FX day**. Attach it to
an M15 or M5 chart with `ProfileTimeframe = PERIOD_H1` if you want to trade
against the level on a lower timeframe.

## Inputs

### Fixed range

| Input | Default | Meaning |
|---|---|---|
| `ProfileTimeframe` | `PERIOD_CURRENT` | Timeframe the profile is built on. Set it explicitly to keep the same levels while you change chart timeframes. |
| `RangeMode` | `Previous FX day` | Previous FX day, Tokyo, London KZ, NY KZ, London Close, or Last N bars. |
| `SessionsBack` | `1` | 1 = the most recent *completed* session, 2 = the one before it, … |
| `LastNBars` | `24` | Range length in *Last N bars* mode only. |
| `LookbackBars` | `1500` | How much history is scanned to find the range. Raise it if a large `SessionsBack` reports "not enough completed sessions". |
| `DayBreakHourUTC` | `21` | FX day roll. 21 is correct during US daylight time; use 22 in northern winter, or 0 for a plain UTC-midnight break. |

### Profile

| Input | Default | Meaning |
|---|---|---|
| `Bins` | `48` | Price bins across the range's high–low. |
| `ValueAreaPercent` | `70` | The Market Profile convention. |
| `ActivitySource` | `Auto` | `Auto` = real volume → tick volume → true range. See the caveat below. |

### Broker time

MT5 bar times are **broker server time** (commonly UTC+2/+3); every session
window here is UTC. Getting this wrong shifts the whole indicator by hours.

| Input | Default | Meaning |
|---|---|---|
| `AutoDetectUTCOffset` | `true` | Derives the offset from `TimeCurrent()` vs `TimeGMT()`, rounded to the half hour. Correct whenever the terminal is connected. |
| `ServerUTCOffsetHours` | `3` | Manual override used when auto-detect is off. |

Check it once: the info panel's range label should name the session you expect.

### Drawing

| Input | Default | Meaning |
|---|---|---|
| `ProfileAnchor` | `Range start` | TradingView/dashboard style. Switch to `Range end` to keep the range's own candles unobscured. |
| `ProfileWidthPct` | `100` | Scales the widest row. |
| `ShowBuySellSplit` | `true` | Cyan (buy) + pink (sell) stacked per row. |
| `ShowRangeBox` | `true` | Shades the fixed range itself. |
| `ShowSessionBreaks` | `true` | Dotted FX-day separators. |
| `ExtendLevelsRight` | `true` | Rays POC/VAH/VAL to the right edge. |
| `ShowInfoPanel` | `true` | Corner readout: range, source, levels, distance in pips, buy share, verdict. |
| `ObjectPrefix` | `FRAP_` | Change it to run two instances on one chart. |

Rows **inside** the value area are drawn solid; the tails are dimmed toward the
chart background — the eye should land on the 70% band, not the extremes.

## Reading it

The info panel's last line classifies the current price:

- **ABOVE VALUE** (bullish) — buyers moved price out of balance and held it.
  VAH is the first support to defend; losing it puts price back inside value and
  kills the breakout thesis, with the POC as the downside magnet.
- **BELOW VALUE** (bearish) — the mirror image. VAL is the first resistance.
- **IN VALUE** (neutral) — price is back where the market already agreed on
  price. Rotational: fade the edges rather than chase breaks, and expect the POC
  to act as a magnet. **Trend entries taken inside value are the ones that chop
  out.**

## Using the levels from an EA

POC/VAH/VAL are published on three `DRAW_NONE` buffers, so an EA can read them
without duplicating the maths:

```mql5
double poc[1], vah[1], val[1];
int h = iCustom(_Symbol, PERIOD_H1, "FixedRangeActivityProfile");
CopyBuffer(h, 0, 0, 1, poc);   // 0 = POC
CopyBuffer(h, 1, 0, 1, vah);   // 1 = VAH
CopyBuffer(h, 2, 0, 1, val);   // 2 = VAL
```

Bars before the range starts return `EMPTY_VALUE`.

## Two conventions worth knowing before you trade the numbers

- **Activity, not always volume.** Spot FX trades OTC, so there is no true
  volume. `Auto` uses tick volume where it exists (a decent proxy in MT5) and
  falls back to per-bar **true range**. The panel names the source, and flags
  `[proxy]` when the bar lengths are ranges rather than contracts. The *prices*
  the POC and value area identify are meaningful either way.
- **The buy/sell split is inferred from bar direction** (up-closing = buy), not
  from the tape. That is the standard retail approximation — it is **not** real
  order-flow delta.

## Matching the dashboard exactly

The dashboard runs on Yahoo data, which reports zero volume for FX, so it always
uses the true-range proxy. To reproduce its numbers on the same instrument and
timeframe, set `ActivitySource = True range proxy`. Left on `Auto`, MT5 will use
tick volume for FX and the POC will differ — usually slightly, occasionally by a
bin or two.

The maths is a line-for-line port: the two-row CBOT value-area expansion (not a
one-row greedy walk), activity spread across every bin a bar's high–low touches,
and the same FX-day/kill-zone resolvers. Verified against
`src/core/volume_profile.py` over 1392 synthetic comparisons with zero
mismatches.

## Notes

- Kill-zone hours mirror `src/services/session_service.py` (Tokyo 00–03, London
  KZ 07–09, NY KZ 12–14, London Close 15–17 UTC). Change them in one place and
  update the other.
- A kill zone is 2–3 hours, so on H1 bars it profiles into a handful of rows.
  The panel warns below 4 bars — read that as rough context, not a tradeable
  level. Drop to M15/M5 for kill-zone profiles.
- Objects are recreated on each new bar of the profile timeframe and on zoom.
  Removing the indicator deletes every object it created.
