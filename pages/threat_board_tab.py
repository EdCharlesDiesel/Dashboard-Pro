"""
threat_board_tab.py — position-aware risk monitor.

Answers: "what threatens MY open trades right now?"
Follows the render(conn) / *_tab.py pattern. Requires threat_core.py.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from src.core import threat_core as tc

STATE_COLOURS = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}


def _positions_editor(conn):
    st.subheader("Open positions")
    st.caption("Mirror your MT5 positions here (pair, direction, lots, entry, SL). "
               "This is what makes every gauge position-aware.")
    df = pd.read_sql("SELECT id, pair, direction, lots, entry, stop "
                     "FROM threat_positions ORDER BY id", conn)
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="threat_pos_editor",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "direction": st.column_config.SelectboxColumn(options=["long", "short"]),
        })
    if st.button("Save positions", key="threat_save"):
        conn.execute(text("DELETE FROM threat_positions"))
        for _, r in edited.iterrows():
            if pd.isna(r["pair"]) or not str(r["pair"]).strip():
                continue
            conn.execute(text(
                "INSERT INTO threat_positions (pair, direction, lots, entry, stop) "
                "VALUES (:p, :d, :l, :e, :s)"),
                {"p": str(r["pair"]).upper().strip(), "d": r["direction"] or "long",
                 "l": float(r["lots"]), "e": float(r["entry"]), "s": float(r["stop"])})
        conn.commit()
        st.rerun()


def _component_chart(components: dict):
    names = list(components.keys())
    vals = [components[n] for n in names]
    colours = [STATE_COLOURS[tc.band(v)] for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=[n.title() for n in names],
                           orientation="h", marker_color=colours,
                           text=[f"{v:.0f}" for v in vals], textposition="outside"))
    fig.add_vline(x=40, line_dash="dot", line_color="grey")
    fig.add_vline(x=70, line_dash="dot", line_color="grey")
    fig.update_layout(xaxis_range=[0, 110], height=260,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Threat score (0-100)")
    st.plotly_chart(fig, use_container_width=True)


def render(conn):
    st.header("Threat Board")
    tc.ensure_tables(conn)

    c1, c2, c3 = st.columns(3)
    equity = c1.number_input("Account equity (USD)", min_value=1.0,
                             value=935.0, step=10.0, key="threat_equity")
    zone_lo = c2.number_input("JPY sensitive zone low", value=tc.JPY_ZONE_LOW,
                              step=0.5, key="threat_zone_lo")
    zone_hi = c3.number_input("JPY sensitive zone high", value=tc.JPY_ZONE_HIGH,
                              step=0.5, key="threat_zone_hi")

    _positions_editor(conn)
    positions = tc.load_positions(conn)
    if not positions:
        st.info("Add at least one position to activate the board.")
        return

    with st.spinner("Evaluating threats..."):
        try:
            rep = tc.build_report(positions, equity, (zone_lo, zone_hi))
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            return

    # ---- headline state ---------------------------------------------------
    colour = STATE_COLOURS[rep.state]
    st.markdown(
        f"<div style='background:{colour};padding:14px;border-radius:10px;"
        f"text-align:center;font-size:1.3rem;color:white;font-weight:700'>"
        f"{rep.state.upper()} — composite threat {rep.score}/100</div>",
        unsafe_allow_html=True)

    d = rep.detail
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("USDJPY", d["usdjpy_last"], f"{d['usdjpy_roc5_pct']}% / 5d")
    m2.metric("Worst correlated stop-out",
              f"${d['worst_cluster_usd']:,.0f}",
              f"{d['worst_cluster_pct_equity']}% of equity ({d['worst_cluster_ccy']})",
              delta_color="inverse")
    cot = d["jpy_cot_percentile"]
    m3.metric("JPY spec net (3y pctile)",
              f"{cot:.0f}%" if cot is not None else "n/a",
              "specs crowded short" if cot is not None and cot <= 25 else "")
    m4.metric("Regime", d["regime"] or "n/a")

    st.caption(f"Net currency exposure (lots): {d['exposure']}")
    _component_chart(rep.components)

    # ---- supporting detail ------------------------------------------------
    if d["headline_hits"]:
        st.warning("Verbal intervention language detected:\n\n- "
                   + "\n- ".join(d["headline_hits"][:5]))
    if d["red_events"]:
        st.subheader("Red-impact events on your currencies (7 days)")
        st.dataframe(pd.DataFrame(d["red_events"]), use_container_width=True)
    elif d["red_events"] is None:
        st.caption("Calendar feed unavailable — check news_fetcher wiring.")

    # ---- journal ----------------------------------------------------------
    if st.button("Journal this reading", key="threat_journal_btn"):
        tc.journal(conn, rep)
        st.success("Journaled.")
    hist = pd.read_sql(
        "SELECT ts, score, state FROM threat_journal ORDER BY ts DESC LIMIT 200", conn)
    if not hist.empty:
        fig = go.Figure(go.Scatter(x=hist["ts"], y=hist["score"], mode="lines+markers",
                                   marker=dict(color=[STATE_COLOURS[s] for s in hist["state"]])))
        fig.add_hrect(y0=70, y1=100, fillcolor="#e74c3c", opacity=0.08, line_width=0)
        fig.add_hrect(y0=40, y1=70, fillcolor="#f39c12", opacity=0.08, line_width=0)
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Composite threat", yaxis_range=[0, 100])
        st.subheader("Threat history")
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE ENTRY — multipage (auto-run) + standalone (streamlit run)
# ===========================================================================
# Guarded behind __main__ (Streamlit runs each page script as __main__) so
# render()/the pure threat_core functions stay importable headlessly — same
# contract as pages/liquidity_hunt_tab.py / pages/disconnect_monitor_tab.py.

if __name__ == "__main__":
    from src.pages_lib.navigation import render_sidebar_nav
    from src.ui.theme import BloombergTheme

    st.set_page_config(
        page_title="THRT · Threat Board",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    BloombergTheme.apply()  # also runs auto_connect() for this session

    with st.sidebar:
        st.markdown("### 🛡️ Threat Board")
        st.caption("Position-aware risk monitor")
        st.divider()
        render_sidebar_nav()

    from src.db.connection import current_db_config

    _cfg = current_db_config()
    if not _cfg.password:
        st.info("🔌 Connect to PostgreSQL in the sidebar to load the Threat Board.")
        st.stop()

    from sqlalchemy import create_engine

    _url = (f"postgresql+psycopg2://{_cfg.user}:{_cfg.password}"
           f"@{_cfg.host}:{_cfg.port}/{_cfg.dbname}")
    _engine = create_engine(_url, pool_pre_ping=True)
    with _engine.connect() as _conn:
        render(_conn)