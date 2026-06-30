"""
app_demo.py — minimal Streamlit demo of the data_backbone pipeline.

Standalone entry point (kept out of the main multipage app, which is driven by
``DailyTradingPage`` via the root ``app.py``). Shows a ticker served through the
Postgres -> yfinance read path:

    streamlit run src/data_backbone/app_demo.py
"""
import sys
from pathlib import Path

# streamlit run puts this file's dir on sys.path, not the repo root — add the
# root so the absolute `src.data_backbone` package import resolves.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.data_backbone import db
from src.data_backbone import data_access as da

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📊 Trading Dashboard — data-backed")

# Ensure tables exist (safe to call repeatedly).
try:
    db.init_db()
except Exception as e:
    st.warning(f"DB not reachable yet: {e}")

ticker = st.text_input("Ticker", "SPY")
if ticker:
    df = da.get_ohlcv(ticker)
    if df.empty:
        st.warning("No data (DB empty and source unreachable).")
    else:
        st.caption(f"{len(df):,} bars — served from Postgres → yfinance.")
        st.line_chart(df["Close"])
        st.dataframe(df.tail(10), use_container_width=True)

with st.expander("FRED series"):
    sid = st.text_input("FRED series id", "DFF")
    if sid:
        s = da.get_fred(sid)
        if not s.empty:
            st.line_chart(s.tail(500))
