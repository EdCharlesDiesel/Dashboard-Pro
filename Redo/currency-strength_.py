from __future__ import annotations

import pandas as pd
import streamlit as st

from src.instruments.registry import INSTRUMENTS

_WINDOWS: tuple[int, ...] = (5, 20, 60, 120)

_PRIMARY_OPTIONS: dict[str, int] = {
    "5 Day": 5, "20 Day": 20, "60 Day": 60, "120 Day": 120,
}

_IDEA_THRESHOLD = 0.15


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_pair_closes() -> pd.DataFrame:
    from src.db.market_cache import cached_closes

    pairs = INSTRUMENTS.forex_pairs()
    tickers = [INSTRUMENTS[p]["ticker"] for p in pairs]
    try:
        close = cached_closes(tickers, period="6mo", interval="1d", ttl=600)
        return close if close is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

