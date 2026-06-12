"""Sidebar controllers for the Daily Trading page."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

from src.db import DBConfig, TradeRepository
from src.instruments import INSTRUMENTS, TREND_COMMODITIES, TREND_TIMEFRAMES
from src.pages_lib.daily_trading.state import CHECKS_TOTAL, SessionStateBootstrap
from src.services import ATRService


class ChecklistSidebar:
    """Sidebar for CHECKLIST mode — instrument, risk, ATR, DB config."""

    def render(self) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;margin-bottom:6px;">'
            'TRADE SETUP</div>',
            unsafe_allow_html=True,
        )

        inst_keys = INSTRUMENTS.keys()
        if st.session_state.selected_instrument not in inst_keys:
            st.session_state.selected_instrument = inst_keys[0]
        selected = st.selectbox(
            "Instrument", inst_keys,
            index=inst_keys.index(st.session_state.selected_instrument),
        )
        st.session_state.selected_instrument = selected
        inst = INSTRUMENTS[selected]
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#00e0ff;margin:4px 0 2px 0;">▸ {inst["ticker"]}</div>'
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;'
            f'color:#9a9a9a;">{inst["corr"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        st.session_state.trade_direction = st.radio(
            "Direction", ["LONG", "SHORT"], horizontal=True,
            index=0 if st.session_state.trade_direction == "LONG" else 1,
        )
        st.session_state.session = st.selectbox(
            "Session", ["London", "New York", "Tokyo", "Sydney"],
            index=["London", "New York", "Tokyo", "Sydney"].index(st.session_state.session)
            if st.session_state.session in ["London", "New York", "Tokyo", "Sydney"] else 0,
        )

        st.markdown("---")
        st.markdown("**◆ POSITION SIZING**")
        st.session_state.account_bal = st.number_input(
            "Account Balance ($)", value=float(st.session_state.account_bal),
            step=500.0, format="%.2f",
        )
        st.session_state.risk_pct = st.slider(
            "Risk per Trade (%)", 0.25, 3.0,
            float(st.session_state.risk_pct), 0.25,
        )

        st.markdown("---")
        st.markdown("**◆ ATR & LEVELS**")
        if st.button("Fetch ATR + Levels", use_container_width=True):
            with st.spinner("Fetching market data…"):
                ATRService.fetch.clear()
                result = ATRService.fetch(inst["ticker"], inst["pip_size"])
                st.session_state[f"atr_data_{selected}"] = result

        atr_data = st.session_state.get(f"atr_data_{selected}", None)
        if atr_data is None:
            atr_data = ATRService.fetch(inst["ticker"], inst["pip_size"])
            st.session_state[f"atr_data_{selected}"] = atr_data

        if atr_data:
            sl_pips = atr_data["sl_pips"]
            tp1_pips = atr_data["tp1_pips"]
            tp2_pips = atr_data["tp2_pips"]
            st.session_state.trend_sl_pips = sl_pips
            st.session_state.trend_tp1_pips = tp1_pips
            st.session_state.trend_tp2_pips = tp2_pips
        else:
            st.warning("⚠️ No live data — manual entry.")
            sl_pips = st.number_input("Stop Loss (pips)", value=20.0, step=1.0)
            tp1_pips = st.number_input("TP1 (pips)", value=40.0, step=1.0)
            tp2_pips = st.number_input("TP2 (pips)", value=60.0, step=1.0)
            st.session_state.trend_sl_pips = sl_pips
            st.session_state.trend_tp1_pips = tp1_pips
            st.session_state.trend_tp2_pips = tp2_pips

        st.markdown("---")
        st.markdown("**◆ POSTGRESQL**")
        st.session_state.db_host = st.text_input("Host", value=st.session_state.db_host)
        st.session_state.db_port = st.number_input(
            "Port", value=int(st.session_state.db_port), step=1,
        )
        st.session_state.db_name = st.text_input("Database", value=st.session_state.db_name)
        st.session_state.db_user = st.text_input("User", value=st.session_state.db_user)
        st.session_state.db_pass = st.text_input(
            "Password", value=st.session_state.db_pass, type="password",
        )

        if st.button("Connect & Init DB", use_container_width=True):
            cfg = DBConfig.from_mapping({
                "host": st.session_state.db_host,
                "port": st.session_state.db_port,
                "dbname": st.session_state.db_name,
                "user": st.session_state.db_user,
                "password": st.session_state.db_pass,
            })
            ok, msg = TradeRepository(cfg).init_schema()
            st.session_state.db_ok = ok
            st.session_state.db_msg = msg

        badge_color = "#00ff66" if st.session_state.db_ok else "#ff3344"
        badge_text = (
            "● CONNECTED" if st.session_state.db_ok
            else f"○ {st.session_state.db_msg[:24]}"
        )
        st.markdown(
            f'<div style="color:{badge_color};font-family:\'JetBrains Mono\',monospace;'
            f'font-size:10px;letter-spacing:0.1em;margin-top:4px;">{badge_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if st.button("Reset All Checks", use_container_width=True):
            SessionStateBootstrap.reset_checks()
            st.rerun()


class TrendSidebar:
    """Sidebar for TREND SIGNALS mode."""

    def render(self) -> None:
        st.markdown(
            '<div style="color:#00ff41;font-weight:700;letter-spacing:0.15em;'
            'text-transform:uppercase;font-size:11px;margin-bottom:6px;">'
            'TREND SIGNALS</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("All Forex", use_container_width=True):
                st.session_state.trend_selected_pairs = [
                    p for p in INSTRUMENTS.keys() if p not in TREND_COMMODITIES
                ]
        with col2:
            if st.button("All Metals", use_container_width=True):
                st.session_state.trend_selected_pairs = [
                    p for p in INSTRUMENTS.keys() if p in TREND_COMMODITIES
                ]

        selected = st.multiselect(
            "Select Markets", INSTRUMENTS.keys(),
            default=st.session_state.trend_selected_pairs,
            key="trend_pair_selector",
        )
        st.session_state.trend_selected_pairs = selected

        tf_keys = list(TREND_TIMEFRAMES.keys())
        st.session_state.trend_timeframe = st.selectbox(
            "Timeframe", tf_keys,
            index=tf_keys.index(st.session_state.trend_timeframe),
        )

        st.divider()
        st.markdown("**◆ SENSITIVITY**")
        st.session_state.trend_min_conds = st.slider(
            "Min conditions", 3, 6,
            st.session_state.trend_min_conds,
            help="Lower = more signals; higher = stronger but fewer.",
        )

        st.divider()
        st.markdown("**◆ AUTO REFRESH**")
        auto = st.toggle("Enable", value=True)
        mins = st.slider("Interval (min)", 1, 60, 15, disabled=not auto)

        if st.button("Refresh Now", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.caption(f"◷ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if auto:
            st.caption(f"Next refresh in ~{mins} min")
