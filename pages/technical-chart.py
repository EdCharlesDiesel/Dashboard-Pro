"""Technical Chart — Perfect Order SMA stack (broken out of Market Overview)."""
import streamlit as st

from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="Technical Chart · Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("📈 Technical Chart")

data = mol.ensure_loaded()
mol.render_technical_chart_tab(data)
