"""Market Overview — the landing snapshot.

Once a ten-tab monolith, this page is now a lean landing: headline KPIs, a price
table, and the background alert engine. Each unique analysis tab it used to host
was broken out (Trading Ideas, MTF Matrix, Technical Chart, Pivots & Fibonacci,
Volume Profile, FRED Macro Grid); only MTF Matrix is still on the live sidebar
nav (see pages/mtf-matrix.py) — the rest were later archived. All the shared
logic lives in ``src/pages_lib/market_overview_lib.py`` so those pages stay thin.
"""
import logging
import traceback
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.core import secrets
from src.core.config import default_config as config
from src.pages_lib import market_overview_lib as mol
from src.ui.theme import BloombergTheme

st.set_page_config(
    page_title="Market Overview · Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()
mol.inject_css()

logger = logging.getLogger("ForexDashboard")


def main():
    st.title(f"💹 Macro Dashboard Pro v{config.version}")

    # ── Auto-refresh every 5 minutes ─────────────────────────────────────────
    # st_autorefresh returns a monotonically increasing counter (0 on first load).
    # On every tick after the first we bust the data cache so fresh prices are
    # fetched and signals re-evaluated — identical to clicking ↺ Refresh.
    _refresh_count = st_autorefresh(interval=300_000, key="dashboard_autorefresh")
    if _refresh_count > 0 and st.session_state.get("_last_refresh_count", -1) != _refresh_count:
        st.session_state["_last_refresh_count"] = _refresh_count
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.last_refresh = datetime.now()

    mol.init_notification_state()

    fred_api_key = secrets.fred_api_key()
    mol.render_full_sidebar(fred_api_key)

    data_by_timeframe = mol.ensure_loaded()

    # BACKGROUND ALERT ENGINE — keep high-conviction ideas firing even when the
    # user is on the landing page rather than the Trading Ideas page.
    ideas = st.session_state.get('latest_ideas', [])
    if ideas:
        mol.check_and_notify(ideas)

    daily_data = data_by_timeframe.get('Daily', {})
    if daily_data:
        mol.render_kpis(daily_data)

    st.divider()
    mol.render_overview_tab(daily_data)

    st.caption(
        "This is the market snapshot. The MTF Matrix that used to live here as "
        "a tab is now its own page — see the sidebar: 🧭 MTF Matrix. The other "
        "former tabs (Trading Ideas, Technical Chart, Pivots & Fibonacci, "
        "Volume Profile, FRED Macro Grid) were archived."
    )


# Streamlit runs this file as a page (imported, not __main__), so call
# unconditionally — matching every other page in pages/.
try:
    main()
except Exception as e:
    st.error(f"Application error: {e}")
    logger.error(traceback.format_exc())
