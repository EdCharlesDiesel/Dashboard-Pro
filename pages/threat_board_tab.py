"""
threat_board_tab.py — position-aware risk monitor.

Answers: "what threatens MY open trades right now?"
Follows the render(conn) / *_tab.py pattern. Requires threat_core.py.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core import threat_core as tc
from src.services import open_positions

STATE_COLOURS = {"green": "#2ecc71", "amber": "#f39c12", "red": "#e74c3c"}


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

    # Equity, not balance: this board reports cluster risk as a % of equity,
    # and the two differ by floating P/L ($3,844.15 vs $3,552.45 on the day
    # this was written). The hardcoded $935 it used before - a balance from
    # months earlier - overstated every percentage about 4x.
    #
    # Equity lives only in the book's account snapshot, which an MT4-statement
    # feed does not supply, so a missing equity falls back to manual entry
    # rather than quietly substituting the balance.
    _snap = open_positions.account_snapshot() or {}
    _live_eq = float(_snap.get("equity") or 0.0)
    _has_live_eq = _live_eq > 0
    st.session_state.setdefault("threat_equity", 935.0)

    _use_live_eq = st.checkbox(
        "🔗 Use live equity from MT5",
        value=bool(st.session_state.get("threat_use_live_eq", _has_live_eq)) and _has_live_eq,
        disabled=not _has_live_eq,
        help="Reads equity from the stored MT5 book. Uncheck to enter one manually.",
    )
    st.session_state["threat_use_live_eq"] = _use_live_eq

    c1, c2, c3 = st.columns(3)
    if _use_live_eq and _has_live_eq:
        equity = _live_eq
        _age = open_positions.age_minutes()
        c1.metric("Account equity (USD)", f"${_live_eq:,.2f}",
                  help="Live from the stored MT5 book"
                       + (f" · {_age:.0f} min old" if _age is not None else ""))
        if _age is not None and _age > 15:
            st.error(f"⚠ Equity is **{_age:.0f} min old** - the MT5 sync is "
                     f"not running, so every risk percentage below is stale.")
    else:
        # No value= here: threat_equity is a keyed widget, so Streamlit reads
        # it from session state, seeded once above.
        equity = c1.number_input("Account equity (USD)", min_value=1.0,
                                 step=10.0, key="threat_equity")
    zone_lo = c2.number_input("JPY sensitive zone low", value=tc.JPY_ZONE_LOW,
                              step=0.5, key="threat_zone_lo")
    zone_hi = c3.number_input("JPY sensitive zone high", value=tc.JPY_ZONE_HIGH,
                              step=0.5, key="threat_zone_hi")

    # The book, not a hand-typed table. This page used to read
    # `threat_positions`, which you filled in yourself - so it showed nothing
    # at all while nine real positions were open. `positions_from_book`
    # absorbs the format mismatches (slash, case, missing stop); see its
    # docstring for why each one is dangerous rather than merely untidy.
    book = open_positions.load()
    positions, no_stop = tc.positions_from_book(book)

    st.subheader("Open positions")
    _age = open_positions.age_minutes()
    st.caption(
        f"{len(book)} position(s) from the MT5 book"
        + (f" · synced {_age:.0f} min ago" if _age is not None else "")
        + (f" · {len(no_stop)} without a stop" if no_stop else ""))
    if _age is not None and _age > 15:
        st.error(f"⚠ Book is **{_age:.0f} min old** - the MT5 sync is not "
                 f"running, so these gauges describe a stale book. "
                 f"See logs/mt5_sync.log.")

    if no_stop:
        # Named, never silently dropped: a position with no stop has unbounded
        # risk, which is the one thing a threat board must not leave out. It is
        # excluded from the cluster maths only because it cannot be quantified.
        _names = ", ".join(f"{r.get('pair')} {r.get('direction')}" for r in no_stop)
        st.error(f"⚠ **{len(no_stop)} position(s) with NO STOP: {_names}** - "
                 f"risk is unbounded, so none of the cluster figures below "
                 f"account for them.")

    if positions:
        st.dataframe(
            pd.DataFrame([{"Pair": p.pair, "Dir": p.direction, "Lots": p.lots,
                           "Entry": p.entry, "Stop": p.stop} for p in positions]),
            use_container_width=True, hide_index=True)

    if not positions:
        st.info("No stopped positions in the MT5 book. If you are holding "
                "trades, the sync may not be running - see logs/mt5_sync.log.")
        return

    with st.spinner("Evaluating threats..."):
        try:
            rep = tc.build_report(positions, equity, (zone_lo, zone_hi))
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            return

    # ---- headline state ---------------------------------------------------
    # The headline follows the worst single component, not the weighted mean.
    # Without that rule a maxed component is averaged away: concentration is
    # worth 30 of 100 points, so on 2026-08-20 a correlated stop-out of 173% of
    # equity scored 100/100 and still printed GREEN at a composite of 30.
    colour = STATE_COLOURS[rep.state]
    _driver = rep.detail.get("state_driver") or []
    _vetoed = rep.state != tc.band(rep.score)
    _by = (" · driven by " + ", ".join(n.title() for n in _driver)) if _driver else ""
    st.markdown(
        f"<div style='background:{colour};padding:14px;border-radius:10px;"
        f"text-align:center;font-size:1.3rem;color:white;font-weight:700'>"
        f"{rep.state.upper()} — composite threat {rep.score}/100{_by}</div>",
        unsafe_allow_html=True)
    if _vetoed:
        st.caption(
            f"The headline follows the **worst single component**, not the "
            f"average. The composite of {rep.score}/100 alone would read "
            f"*{tc.band(rep.score)}* — a component worth {max(tc.WEIGHTS.values())} "
            f"of 100 points cannot move it on its own, so a maxed component "
            f"sets the state instead of being averaged away.")

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