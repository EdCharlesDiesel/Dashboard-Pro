"""platinum_tab.py — Platinum (XPTUSD) monitor.

Answers: does the rand carry information about platinum *after* the dollar
factor is removed? A raw corr(XPTUSD, USDZAR) cannot say — DXY weakness lifts EM
currencies and dollar-denominated metals at the same time — so the engine runs a
nested incremental test instead.

Follows the `render(engine)` / `*_tab.py` pattern. Engine lives in
`src/core/platinum.py`; the tab body in `src/pages_lib/platinum.py`; the
collector in `src/data_backbone/platinum_jobs.py`.
"""
from __future__ import annotations

import streamlit as st

from src.pages_lib.platinum import render

if __name__ == "__main__":
    st.set_page_config(
        page_title="Platinum Monitor · Trading System",
        page_icon="⚪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from src.ui.theme import BloombergTheme

    BloombergTheme.apply()

    with st.sidebar:
        st.caption("Platinum monitor — XPTUSD vs the dollar factor")
        st.divider()
        from src.pages_lib.navigation import render_sidebar_nav

        render_sidebar_nav()

    from src.db.connection import current_db_config

    _cfg = current_db_config()
    if not _cfg.password:
        # Same contract as the other DB-backed tabs: say what is missing rather
        # than rendering an empty page that looks like "no signal".
        st.info("🔌 Connect to PostgreSQL in the sidebar to load the Platinum Monitor.")
        st.stop()

    from sqlalchemy import create_engine

    _url = (f"postgresql+psycopg2://{_cfg.user}:{_cfg.password}"
            f"@{_cfg.host}:{_cfg.port}/{_cfg.dbname}")
    _engine = create_engine(_url, pool_pre_ping=True)

    # render() takes the engine itself, not a connection: the tab opens its own
    # short-lived connections behind st.cache_data.
    render(_engine)
