"""Volume Profile — volume-by-price, POC & value areas (broken out of Market Overview)."""
import streamlit as st

from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="Volume Profile · Trading System",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("🔊 Volume Profile")

data = mol.ensure_loaded()
mol.render_volume_profile_tab(data)
