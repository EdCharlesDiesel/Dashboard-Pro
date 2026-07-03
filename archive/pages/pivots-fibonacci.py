"""Pivots & Fibonacci — session key levels (broken out of Market Overview)."""
import streamlit as st

from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="Pivots & Fibonacci · Trading System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("🛒 Pivots & Fibonacci")

data = mol.ensure_loaded()
mol.render_trading_view_tab(data)
