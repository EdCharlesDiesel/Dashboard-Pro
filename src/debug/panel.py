"""Shared debug panel — data-source switching + the historical time machine.

Renders nothing outside DASHBOARD_ENV=dev (see src/debug/env.py). Sets
``dbg_source``/``dbg_asof`` in session_state, which src/db/market_cache.py's
``cached_ohlc`` reads as its defaults — so wiring a page into this panel
never requires threading those params through the page's own fetch calls.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.debug.env import is_dev


def debug_panel(page_name: str, dataframes: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    if not is_dev():
        return
    with st.sidebar.expander(f"🐛 DEBUG — {page_name}", expanded=False):
        source = st.radio(
            "Data source", ["live", "duka"], horizontal=True,
            index=0 if st.session_state.get("dbg_source", "live") == "live" else 1,
            help="'duka' reads the Dukascopy archive (src/data/dukascopy_backfill.py). "
                 "Falls back to live with a warning for any symbol/interval "
                 "that hasn't been backfilled (or has no Dukascopy mapping — "
                 "e.g. GBP/ZAR).",
        )
        st.session_state["dbg_source"] = source

        time_machine_on = st.toggle(
            "Time machine", value=st.session_state.get("dbg_asof") is not None,
            help="Replay this page as of a past date — every cached_ohlc call "
                 "on the page truncates to bars on or before it.",
        )
        if time_machine_on:
            st.session_state["dbg_asof"] = st.date_input(
                "Render as of",
                value=st.session_state.get("dbg_asof") or pd.Timestamp.utcnow().date(),
            )
        else:
            st.session_state["dbg_asof"] = None

        for name, df in (dataframes or {}).items():
            prov = df.attrs.get("provenance") if df is not None else None
            st.caption(f"**{name}** — {prov if prov else 'no provenance (not fetched via cached_ohlc)'}")
            if df is not None and not df.empty:
                st.dataframe(df.tail(5), height=150)

        if st.button("📸 Snapshot state to Postgres", key=f"dbg_snapshot_{page_name}"):
            from src.services.tool_log import log_tool_usage
            payload = {
                "page": page_name,
                "source": st.session_state.get("dbg_source"),
                "asof": str(st.session_state.get("dbg_asof")),
                "dataframes": {
                    name: (df.attrs.get("provenance") if df is not None else None)
                    for name, df in (dataframes or {}).items()
                },
            }
            ok = log_tool_usage("debug_snapshot", payload)
            (st.toast if ok else st.error)(
                "Snapshot saved." if ok else "Snapshot failed — DB not configured?"
            )
