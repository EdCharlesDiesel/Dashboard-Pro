"""
oil_cot_tab.py
==============
WTI crude COT page — thin wrapper over the shared disaggregated-COT renderer.

Entry point: render_oil_cot_tab()
Standalone:  streamlit run pages/oil_cot_tab.py
"""
from __future__ import annotations

import streamlit as st

from src.pages_lib.commodity_cot_lib import render_commodity_cot_tab


def render_oil_cot_tab() -> None:
    render_commodity_cot_tab("WTI")


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================

def _page() -> None:
    from src.ui.theme import BloombergTheme
    from src.pages_lib.navigation import render_sidebar_nav

    st.set_page_config(
        page_title="OCOT · Oil COT",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()
    with st.sidebar:
        st.markdown("### 🛢️ Oil COT")
        st.caption("Disaggregated Managed Money positioning")
        st.divider()
        render_sidebar_nav()
    render_oil_cot_tab()


_page()
