"""ATR fetcher + derived SL/TP calculation.

Logic is byte-equivalent to the original `fetch_atr()` function from
daily-trading-checklist.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from src.indicators.technical import TechnicalIndicators


@dataclass(frozen=True)
class ATRReading:
    atr14: float
    atr20: float
    atr14_pips: float
    atr20_pips: float
    sl_pips: float
    tp1_pips: float
    tp2_pips: float
    atr_ok: bool

    def as_dict(self) -> dict:
        return {
            "atr14": self.atr14, "atr20": self.atr20,
            "atr14_pips": self.atr14_pips, "atr20_pips": self.atr20_pips,
            "sl_pips": self.sl_pips, "tp1_pips": self.tp1_pips,
            "tp2_pips": self.tp2_pips, "atr_ok": self.atr_ok,
        }


class ATRService:
    """Pulls daily OHLCV and computes ATR14 / ATR20 plus stop & target pips."""

    SL_ATR_MULT = 1.5
    TP1_R = 2.0
    TP2_R = 3.0

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch(ticker: str, pip_size: float) -> Optional[dict]:
        """Returns a dict shaped exactly like the legacy fetch_atr output."""
        try:
            df = yf.download(
                ticker, period="60d", interval="1d",
                progress=False, auto_adjust=True,
            )
            if df.empty or len(df) < 22:
                return None
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            df["atr14"] = TechnicalIndicators.atr(df["High"], df["Low"], df["Close"], 14)
            df["atr20"] = TechnicalIndicators.atr(df["High"], df["Low"], df["Close"], 20)
            df = df.dropna(subset=["atr14", "atr20"])
            if df.empty:
                return None

            a14 = float(df["atr14"].iloc[-1])
            a20 = float(df["atr20"].iloc[-1])
            a14_pips = round(a14 / pip_size, 1)
            a20_pips = round(a20 / pip_size, 1)
            sl_pips = round(a14_pips * ATRService.SL_ATR_MULT, 1)
            tp1_pips = round(sl_pips * ATRService.TP1_R, 1)
            tp2_pips = round(sl_pips * ATRService.TP2_R, 1)

            return ATRReading(
                atr14=round(a14, 5), atr20=round(a20, 5),
                atr14_pips=a14_pips, atr20_pips=a20_pips,
                sl_pips=sl_pips, tp1_pips=tp1_pips, tp2_pips=tp2_pips,
                atr_ok=a14 > a20,
            ).as_dict()
        except Exception:
            return None
