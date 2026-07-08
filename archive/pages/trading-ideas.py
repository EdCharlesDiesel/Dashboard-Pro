"""Trading Ideas — live, auto-refreshed setups (broken out of Market Overview)."""
import streamlit as st

from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="Trading Ideas · Trading System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("🎯 Trading Ideas")

mol.ensure_loaded()
mol.render_trading_ideas_tab()
