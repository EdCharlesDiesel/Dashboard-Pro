"""FRED Macro Grid — 8-series macro dashboard (broken out of Market Overview)."""
import streamlit as st

from src.core import secrets
from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="FRED Macro Grid · Trading System",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()
mol.subpage_sidebar("🏛 FRED Macro Grid")

mol.render_macro_pro_tab(secrets.fred_api_key())
