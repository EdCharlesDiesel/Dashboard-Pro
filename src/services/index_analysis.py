"""Correlation / relative-strength math for two price series (e.g. S&P 500 vs
Dow Jones). Pure — no Streamlit, no network, no DB — so the maths is unit-tested
without a chart or a live fetch. The page (pages/indices-correlation.py) feeds
it two Close series from the canonical spine and renders the result.

Everything aligns the two series on their **common** dates first, so a market
holiday that hits one index but not the other never skews a correlation.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _aligned(a_close: pd.Series, b_close: pd.Series) -> pd.DataFrame:
    """Two Close series joined on their shared timestamps, NaNs dropped."""
    df = pd.concat({"a": a_close, "b": b_close}, axis=1).dropna()
    return df


def daily_returns(a_close: pd.Series, b_close: pd.Series):
    """Aligned simple daily returns for both series (index-matched)."""
    df = _aligned(a_close, b_close)
    return df["a"].pct_change(fill_method=None).dropna(), df["b"].pct_change(fill_method=None).dropna()


def rolling_correlation(a_close: pd.Series, b_close: pd.Series,
                        window: int) -> pd.Series:
    """Rolling Pearson correlation of the two series' daily returns."""
    df = _aligned(a_close, b_close)
    ra = df["a"].pct_change(fill_method=None)
    rb = df["b"].pct_change(fill_method=None)
    return ra.rolling(window).corr(rb)


def ratio(a_close: pd.Series, b_close: pd.Series) -> pd.Series:
    """Relative-strength ratio a/b on the common dates (rising = a outperforms)."""
    df = _aligned(a_close, b_close)
    return df["a"] / df["b"].replace(0, np.nan)


def correlation_summary(a_close: pd.Series, b_close: pd.Series,
                        windows: Sequence[int] = (20, 60)) -> Dict[str, Optional[float]]:
    """Headline numbers for the two series:

    - ``full`` — correlation of returns over the whole overlapping window
    - ``w{n}`` — the latest value of the n-day rolling correlation
    - ``beta`` — sensitivity of ``a``'s returns to ``b``'s (cov/var), i.e. how
      much S&P moves per unit Dow move
    - ``n`` — number of overlapping return observations
    """
    ra, rb = daily_returns(a_close, b_close)
    out: Dict[str, Optional[float]] = {"n": int(len(ra))}
    out["full"] = float(ra.corr(rb)) if len(ra) > 2 else None
    for w in windows:
        rc = rolling_correlation(a_close, b_close, w).dropna()
        out[f"w{w}"] = float(rc.iloc[-1]) if len(rc) else None
    var_b = float(rb.var()) if len(rb) > 2 else 0.0
    out["beta"] = float(ra.cov(rb) / var_b) if var_b > 0 else None
    return out
