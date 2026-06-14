"""Sidebar controllers for the Daily Trading page."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import streamlit as st

from src.db import DBConfig, TradeRepository
from src.instruments import INSTRUMENTS, TREND_COMMODITIES, TREND_TIMEFRAMES
from src.pages_lib.daily_trading.state import CHECKS_TOTAL, SessionStateBootstrap
from src.services import ATRService, account_state


@st.cache_data(ttl=60, show_spinner=False)
def _cached_realized_pnl(host, port, dbname, user, password):
    """Realised P/L from the journal, cached 60s so it isn't re-queried on every
    sidebar rerun. Keyed on the connection so changing DB invalidates it."""
    cfg = DBConfig.from_mapping({
        "host": host, "port": port, "dbname": dbname,
        "user": user, "password": password,
    })
    return TradeRepository(cfg).realized_pnl()


class ChecklistSidebar:
    """Sidebar for CHECKLIST mode — instrument, risk, ATR, DB config."""

    def _realized_pnl(self):
        """Realised P/L from the journal DB (cached 60s), or None if it can't be
        read (not connected, or the schema predates the `profit` column)."""
        try:
            return _cached_realized_pnl(
                st.session_state.db_host, st.session_state.db_port,
                st.session_state.db_name, st.session_state.db_user,
                st.session_state.db_pass,
            )
        except Exception:
            return None

    def _render_balance(self) -> None:
        """Account balance with three sources: computed from trade history (the
        true source of data), the live MT4-statement balance, or manual."""
        sources = ["Trade history", "Live (MT4 statement)", "Manual"]
        source = st.radio(
            "Balance source", sources,
            index=sources.index(st.session_state.get("dt_bal_source", "Trade history")),
            help="Trade history = starting balance + realised P/L from the journal. "
                 "Live = the balance off the latest imported MT4 statement.",
        )
        st.session_state.dt_bal_source = source

        acct = account_state.get()
        live_bal = account_state.get_balance(0.0)
        has_live = bool(acct) and live_bal > 0

        if source == "Trade history":
            start_bal = st.number_input(
                "Starting balance ($)", min_value=0.0, step=500.0,
                value=float(st.session_state.get("dt_start_bal", live_bal or 10000.0)),
                help="Account balance before the first trade in the journal.",
            )
            st.session_state.dt_start_bal = start_bal
            info = self._realized_pnl()
            if info is None:
                st.session_state.account_bal = start_bal
                st.caption("⚠ Connect to PostgreSQL below to compute balance from history.")
            else:
                bal = start_bal + info["pnl"]
                st.session_state.account_bal = bal
                pnl_color = "#00ff41" if info["pnl"] >= 0 else "#ff3344"
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                    f'BAL <span style="color:#00ff41;font-weight:700;">${bal:,.2f}</span>'
                    f'<span style="color:#9a9a9a;"> = ${start_bal:,.0f} </span>'
                    f'<span style="color:{pnl_color};font-weight:700;">{info["pnl"]:+,.2f}</span>'
                    f'<span style="color:#9a9a9a;"> P/L · {info["counted"]} trades</span></div>',
                    unsafe_allow_html=True,
                )
                if info["skipped"]:
                    st.caption(f"{info['skipped']} closed trade(s) skipped "
                               "(missing P/L / lot size / unmapped instrument).")
        elif source == "Live (MT4 statement)":
            if has_live:
                st.session_state.account_bal = live_bal
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                    f'BAL <span style="color:#00ff41;font-weight:700;">${live_bal:,.2f}</span>'
                    f'<span style="color:#9a9a9a;"> · {acct.get("source", "—")} · '
                    f'{acct.get("updated_at", "—")}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No live balance yet — import an MT4 statement in the Trade Journal.")
                st.session_state.account_bal = st.number_input(
                    "Account Balance ($)", value=float(st.session_state.account_bal),
                    step=500.0, format="%.2f",
                )
        else:  # Manual
            st.session_state.account_bal = st.number_input(
                "Account Balance ($)", value=float(st.session_state.account_bal),
                step=500.0, format="%.2f",
            )

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
        self._render_balance()
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
