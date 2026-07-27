"""Fixed-range volume profile — TradingView's FRVP, anchored to session breaks.

TradingView's *Fixed Range Volume Profile* takes an arbitrary bar range and
answers one question: **at which prices did this range actually trade?** The
answer is three levels that carry forward as support/resistance long after the
range closes:

- **POC** (Point of Control) — the single price bin with the most activity.
  The range's fair value; price tends to rotate back to it.
- **VAH / VAL** (Value Area High/Low) — the contiguous band around the POC
  holding ``value_area_pct`` (70% by convention) of the range's activity.
  Outside the value area is where the range was *rejected*, not accepted.

This module fixes the range to a **session break** rather than a hand-dragged
selection: the previous FX day (21:00 UTC ≈ 17:00 New York, the standard FX
day roll) or the previous occurrence of one named kill zone. That makes the
levels reproducible — two traders reading the same page get the same POC,
which a dragged range can never guarantee.

Everything here is pure (frames in, dataclasses out): no Streamlit, no
network, no plotting. The AMD scanner's *Fixed Range VP* tab is the renderer.

Two conventions worth knowing before you read the numbers:

- **Activity, not always volume.** Spot FX (``=X`` tickers) trades OTC, so
  Yahoo reports zero volume. When volume is unusable the profile falls back to
  per-bar **true range** as an activity proxy — the same substitution the AMD
  scanner's existing volume-by-price panel makes. ``VolumeProfile.is_proxy``
  says which one you got; a proxy profile still locates the prices where the
  market spent its energy, but the bar magnitudes are ranges, not contracts.
- **Buy/sell split is inferred from bar direction**, not from the tape. An
  up-closing bar's activity counts as buy, a down-closing bar's as sell. This
  is the standard approximation every retail "up/down volume" profile makes —
  it is not real delta.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.services.session_service import SessionService

# The FX day rolls at the New York close (17:00 ET). 21:00 UTC is that roll
# during US daylight time; the hour is a parameter so the caller can move it
# to 22:00 in northern winter, or to 0 for a plain UTC-midnight break.
FX_DAY_BREAK_HOUR = 21

# Named intraday windows, sourced from SessionService so the kill-zone hours
# stay defined in exactly one place (src/services/session_service.py).
SESSION_WINDOWS: Dict[str, Tuple[float, float]] = {
    "Tokyo": SessionService.TOKYO,
    "London KZ": SessionService.LONDON_KZ,
    "NY KZ": SessionService.NY_KZ,
    "London Close": SessionService.LONDON_CLOSE,
}

DEFAULT_BINS = 48
DEFAULT_VALUE_AREA = 0.70


# --------------------------------------------------------------------------- #
# Activity source
# --------------------------------------------------------------------------- #
def true_range(df: pd.DataFrame) -> pd.Series:
    """Per-bar true range (no smoothing). Used as a volume-like activity proxy."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def has_usable_volume(s: pd.Series) -> bool:
    """Volume is usable when it has at least one non-NaN, non-zero value.

    Yahoo Finance returns all-zero Volume for FX pairs — treat that as "no
    volume" and fall back to the range-based activity proxy instead.
    """
    vol = pd.to_numeric(s, errors="coerce")
    return bool(vol.fillna(0).abs().sum() > 0)


def activity_series(df: pd.DataFrame,
                    use_volume: Optional[bool] = None) -> Tuple[pd.Series, bool]:
    """Return ``(series, is_proxy)`` — traded volume when it exists, else true range.

    ``use_volume`` forces the choice (``True`` = insist on volume even if it is
    all zeros, ``False`` = insist on the proxy); ``None`` auto-detects.
    """
    if use_volume is None:
        use_volume = "Volume" in df.columns and has_usable_volume(df["Volume"])
    if use_volume and "Volume" in df.columns:
        return pd.to_numeric(df["Volume"], errors="coerce").fillna(0), False
    return true_range(df).fillna(0), True


# --------------------------------------------------------------------------- #
# Session ranges
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RangeWindow:
    """A fixed bar range: the profile's anchor."""
    start: pd.Timestamp
    end: pd.Timestamp
    label: str

    @property
    def duration(self) -> pd.Timedelta:
        return self.end - self.start


def _utc_index(index) -> pd.DatetimeIndex:
    """Coerce any bar index to tz-aware UTC (naive timestamps are assumed UTC)."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``df`` with its index coerced to tz-aware UTC.

    Renderers need this: the window timestamps this module hands back are
    always UTC, so a chart drawn against a naive (or exchange-local) index
    would place the range box and its levels at the wrong x positions.
    """
    out = df.copy()
    out.index = _utc_index(out.index)
    return out


def session_day(index, day_break_hour: int = FX_DAY_BREAK_HOUR) -> pd.DatetimeIndex:
    """Map each bar to the FX day it belongs to.

    Bars from ``day_break_hour`` onward belong to the *next* calendar day's
    session, which is why the whole index is shifted back before flooring:
    21:00 Monday and 08:00 Tuesday are the same FX day.
    """
    return (_utc_index(index) - pd.Timedelta(hours=day_break_hour)).normalize()


def session_breaks(index,
                   day_break_hour: int = FX_DAY_BREAK_HOUR) -> List[pd.Timestamp]:
    """Bar timestamps where a new FX day starts — the chart's vertical separators."""
    idx = _utc_index(index)
    if len(idx) == 0:
        return []
    days = session_day(idx, day_break_hour)
    breaks: List[pd.Timestamp] = []
    prev = None
    for ts, day in zip(idx, days):
        if prev is not None and day != prev:
            breaks.append(ts)
        prev = day
    return breaks


def previous_day_range(index, back: int = 1,
                       day_break_hour: int = FX_DAY_BREAK_HOUR
                       ) -> Optional[RangeWindow]:
    """The FX day ``back`` sessions ago (``back=1`` = yesterday, the default).

    The in-progress day is never returned — a profile over a half-formed
    session has a POC that moves every rerun, which is the opposite of the
    reproducible level this indicator exists to provide.
    """
    idx = _utc_index(index)
    if len(idx) == 0 or back < 1:
        return None
    days = session_day(idx, day_break_hour)
    unique = sorted(set(days))
    # unique[-1] is the day still forming; step back from there.
    target_pos = len(unique) - 1 - back
    if target_pos < 0:
        return None
    target = unique[target_pos]
    mask = days == target
    bars = idx[mask]
    if len(bars) == 0:
        return None
    return RangeWindow(start=bars[0], end=bars[-1],
                       label=f"FX day {pd.Timestamp(target).date()}")


def _session_blocks(index, window: Tuple[float, float],
                    day_break_hour: int = FX_DAY_BREAK_HOUR) -> List[RangeWindow]:
    """Every contiguous occurrence of an intraday window, oldest first."""
    idx = _utc_index(index)
    if len(idx) == 0:
        return []
    lo, hi = window
    hours = idx.hour + idx.minute / 60.0
    in_window = (hours >= lo) & (hours < hi)
    days = session_day(idx, day_break_hour)
    blocks: List[RangeWindow] = []
    current_day = None
    start = end = None
    for ts, day, inside in zip(idx, days, in_window):
        if not inside:
            continue
        if current_day is None or day != current_day:
            if start is not None:
                blocks.append(RangeWindow(start, end, ""))
            current_day, start = day, ts
        end = ts
    if start is not None:
        blocks.append(RangeWindow(start, end, ""))
    return blocks


def previous_session_range(index, session: str, back: int = 1,
                           day_break_hour: int = FX_DAY_BREAK_HOUR
                           ) -> Optional[RangeWindow]:
    """The previous complete occurrence of a named kill zone (see SESSION_WINDOWS).

    A block whose last bar is also the last bar of the whole index is still
    forming and is dropped, same reasoning as :func:`previous_day_range`.
    """
    if session not in SESSION_WINDOWS or back < 1:
        return None
    idx = _utc_index(index)
    blocks = _session_blocks(idx, SESSION_WINDOWS[session], day_break_hour)
    if blocks and len(idx) and blocks[-1].end == idx[-1]:
        blocks = blocks[:-1]          # in progress right now
    if len(blocks) < back:
        return None
    block = blocks[-back]
    return RangeWindow(
        start=block.start, end=block.end,
        label=f"{session} {block.start.date()}",
    )


def last_n_bars_range(index, n: int) -> Optional[RangeWindow]:
    """Plain fixed range over the most recent ``n`` bars — the manual escape hatch."""
    idx = _utc_index(index)
    if len(idx) == 0 or n < 1:
        return None
    bars = idx[-n:]
    return RangeWindow(start=bars[0], end=bars[-1], label=f"last {len(bars)} bars")


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VolumeProfile:
    """A fixed-range volume profile and its derived levels."""
    edges: np.ndarray          # bin edges, length bins+1
    centers: np.ndarray        # bin centers, length bins
    total: np.ndarray          # activity per bin
    buy: np.ndarray            # activity from up-closing bars
    sell: np.ndarray           # activity from down-closing bars
    poc: float                 # price of the highest-activity bin
    vah: float                 # value area high
    val: float                 # value area low
    poc_index: int
    va_low_index: int
    va_high_index: int
    window: RangeWindow
    is_proxy: bool             # True when built on true range, not real volume
    value_area_pct: float
    bars: int

    @property
    def bin_width(self) -> float:
        return float(self.edges[1] - self.edges[0]) if len(self.edges) > 1 else 0.0

    @property
    def range_high(self) -> float:
        return float(self.edges[-1])

    @property
    def range_low(self) -> float:
        return float(self.edges[0])

    @property
    def buy_share(self) -> float:
        """Fraction of the range's activity from up-closing bars (0..1)."""
        tot = float(self.total.sum())
        return float(self.buy.sum() / tot) if tot > 0 else 0.5

    @property
    def value_area_width(self) -> float:
        return float(self.vah - self.val)


def value_area(volumes: np.ndarray,
               pct: float = DEFAULT_VALUE_AREA) -> Tuple[int, int, int]:
    """Return ``(low_index, poc_index, high_index)`` for the value area.

    The standard Market Profile expansion: start at the POC and repeatedly
    annex whichever *pair* of adjacent bins (two above vs two below) carries
    more activity, until ``pct`` of the total is enclosed. Taking two rows at a
    time is the CBOT convention TradingView follows — a one-row-at-a-time
    greedy walk drifts the band toward whichever side happens to have the
    noisier single bin.
    """
    n = len(volumes)
    if n == 0:
        return 0, 0, 0
    poc = int(np.argmax(volumes))
    total = float(volumes.sum())
    if total <= 0:
        return poc, poc, poc
    target = total * pct
    acc = float(volumes[poc])
    lo = hi = poc
    while acc < target and (lo > 0 or hi < n - 1):
        # -1 is a "no room left on this side" sentinel; activity is never < 0.
        if hi < n - 1:
            up_hi = min(hi + 2, n - 1)
            up = float(volumes[hi + 1:up_hi + 1].sum())
        else:
            up_hi, up = hi, -1.0
        if lo > 0:
            dn_lo = max(lo - 2, 0)
            dn = float(volumes[dn_lo:lo].sum())
        else:
            dn_lo, dn = lo, -1.0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            acc += up
            hi = up_hi
        else:
            acc += dn
            lo = dn_lo
    return lo, poc, hi


def fixed_range_profile(df: pd.DataFrame,
                        window: Optional[RangeWindow] = None,
                        bins: int = DEFAULT_BINS,
                        value_area_pct: float = DEFAULT_VALUE_AREA,
                        use_volume: Optional[bool] = None
                        ) -> Optional[VolumeProfile]:
    """Build the profile for ``window`` (defaults to the whole frame).

    Each bar's activity is spread evenly across every price bin its high–low
    range touches — the same distribution the AMD scanner's volume-by-price
    panel uses, so the two panels agree on where the market traded.

    Returns ``None`` when the range has no bars or no price extent.
    """
    if df is None or df.empty or bins < 2:
        return None
    frame = df.copy()
    frame.index = _utc_index(frame.index)
    if window is not None:
        frame = frame.loc[(frame.index >= window.start) & (frame.index <= window.end)]
    if frame.empty:
        return None
    if window is None:
        window = RangeWindow(frame.index[0], frame.index[-1], "full range")

    lo = float(frame["Low"].min())
    hi = float(frame["High"].max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return None

    activity, is_proxy = activity_series(frame, use_volume=use_volume)
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = np.zeros(bins)
    buy = np.zeros(bins)

    highs = frame["High"].to_numpy()
    lows = frame["Low"].to_numpy()
    opens = frame["Open"].to_numpy()
    closes = frame["Close"].to_numpy()
    vals = activity.to_numpy()

    for h, l, o, c, v in zip(highs, lows, opens, closes, vals):
        if not (np.isfinite(v) and np.isfinite(h) and np.isfinite(l)) or v <= 0:
            continue
        lo_i = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, bins - 1))
        hi_i = int(np.clip(np.searchsorted(edges, h, side="right") - 1, 0, bins - 1))
        span = hi_i - lo_i + 1
        total[lo_i:hi_i + 1] += v / span
        if np.isfinite(c) and np.isfinite(o) and c >= o:
            buy[lo_i:hi_i + 1] += v / span

    if total.sum() <= 0:
        return None

    va_lo, poc_i, va_hi = value_area(total, value_area_pct)
    return VolumeProfile(
        edges=edges, centers=centers, total=total, buy=buy, sell=total - buy,
        poc=float(centers[poc_i]),
        vah=float(edges[va_hi + 1]),
        val=float(edges[va_lo]),
        poc_index=poc_i, va_low_index=va_lo, va_high_index=va_hi,
        window=window, is_proxy=is_proxy, value_area_pct=value_area_pct,
        bars=int(len(frame)),
    )


# --------------------------------------------------------------------------- #
# Reading the levels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProfileRead:
    """Plain-language interpretation of where price sits vs the prior range."""
    state: str          # ABOVE VALUE / IN VALUE / BELOW VALUE
    lean: str           # Bullish / Bearish / Neutral
    headline: str
    detail: str
    dist_to_poc: float  # signed: price - POC


def read_profile(profile: VolumeProfile, price: float) -> ProfileRead:
    """Classify ``price`` against the profile's value area.

    The three states are the whole point of carrying a prior session's profile
    forward. Acceptance *outside* the previous value area is directional
    information; rotation *inside* it is the market agreeing with yesterday's
    price and is the classic chop warning.
    """
    dist = float(price - profile.poc)
    if price > profile.vah:
        return ProfileRead(
            state="ABOVE VALUE", lean="Bullish",
            headline=f"Price is accepting above {profile.window.label} value",
            detail=(
                "Trading above the prior range's value area high — buyers have "
                "moved price out of balance and held it there. VAH "
                f"({profile.vah:,.5f}) is the first support to defend; losing it "
                "puts price back inside value and kills the breakout thesis. "
                f"The POC at {profile.poc:,.5f} is the downside magnet if that "
                "happens."
            ),
            dist_to_poc=dist,
        )
    if price < profile.val:
        return ProfileRead(
            state="BELOW VALUE", lean="Bearish",
            headline=f"Price is accepting below {profile.window.label} value",
            detail=(
                "Trading below the prior range's value area low — sellers have "
                "moved price out of balance and held it there. VAL "
                f"({profile.val:,.5f}) is the first resistance; reclaiming it "
                "puts price back inside value and kills the breakdown thesis. "
                f"The POC at {profile.poc:,.5f} is the upside magnet if that "
                "happens."
            ),
            dist_to_poc=dist,
        )
    near_poc = abs(dist) <= profile.bin_width * 1.5
    return ProfileRead(
        state="IN VALUE", lean="Neutral",
        headline=(f"Price is rotating inside {profile.window.label} value"
                  + (" — sitting on the POC" if near_poc else "")),
        detail=(
            "Price is back inside the prior range's value area, which is where "
            "the market already agreed on price. Rotational conditions: fade "
            f"the edges ({profile.val:,.5f} / {profile.vah:,.5f}) rather than "
            "chase breaks, and expect the POC "
            f"({profile.poc:,.5f}) to act as a magnet. Trend entries taken "
            "inside value are the ones that chop out."
        ),
        dist_to_poc=dist,
    )
