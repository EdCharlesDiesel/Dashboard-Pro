"""Static technical-indicator helpers.

These are byte-equivalent to the functions previously inlined in
daily-trading-checklist.py — names, formulas, and parameters are preserved.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Stateless namespace for indicator math (EMA, RSI, MACD, ADX, ATR)."""

    # ── EMA ────────────────────────────────────────────────────────────────
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    # ── RSI ────────────────────────────────────────────────────────────────
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ── MACD ───────────────────────────────────────────────────────────────
    @staticmethod
    def macd(
        series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        e_fast = TechnicalIndicators.ema(series, fast)
        e_slow = TechnicalIndicators.ema(series, slow)
        macd_line = e_fast - e_slow
        sig_line = TechnicalIndicators.ema(macd_line, sig)
        hist = macd_line - sig_line
        return macd_line, sig_line, hist

    # ── ADX ────────────────────────────────────────────────────────────────
    @staticmethod
    def adx(
        df: pd.DataFrame, period: int = 14
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
        minus_dm = down_move.where((down_move > 0) & (down_move > up_move), 0.0)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * (
            plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        )
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
        return adx_val, plus_di, minus_di

    # ── ATR (Wilder, EWM) — matches original fetch_atr helper ──────────────
    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        prev = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # ── Convenience: enrich an OHLCV frame with the trend set ──────────────
    @staticmethod
    def enrich_trend_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA50, EMA200, RSI, MACD trio, ADX trio in-place (returns a copy)."""
        df = df.copy()
        df["EMA50"] = TechnicalIndicators.ema(df["Close"], 50)
        df["EMA200"] = TechnicalIndicators.ema(df["Close"], 200)
        df["RSI"] = TechnicalIndicators.rsi(df["Close"])
        df["MACD"], df["MACDSig"], df["MACDHist"] = TechnicalIndicators.macd(df["Close"])
        df["ADX"], df["PlusDI"], df["MinusDI"] = TechnicalIndicators.adx(df)
        return df
