"""System Logs — in-app viewer for everything observability records.

Reads the structured event log (logs/events.jsonl), the text log
(logs/app.log) and the Postgres app_events table, with filtering and download.
All recording is wired up in src/core/observability.py; this page is read-only.
"""
import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.core import observability
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="System Logs · Trading System",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

st.markdown("""
<style>
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer{visibility:hidden;}
    .block-container{padding-top:1.5rem;max-width:1500px;}
    .ll-hero{border:1px solid #2a2a2a;background:#0a0a0a;padding:14px 18px;margin-bottom:14px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧾 System Logs")
    st.caption("Everything the app records — page views, data fetches, errors.")

    max_rows = st.slider("Rows to load", 100, 5000, 1000, step=100)

    if st.button("🔄 Refresh", use_container_width=True, type="primary"):
        st.rerun()

    if st.button("🔌 Retry Postgres sink", use_container_width=True):
        observability.reset_pg()
        st.rerun()

    paths = observability.log_paths()
    st.divider()
    st.caption("Log files")
    st.code(str(paths["dir"]), language=None)

    st.divider()
    render_sidebar_nav()

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ll-hero">
  <div style="font-size:24px;font-weight:700;color:#e6e6e6;">🧾 System Logs</div>
  <div style="color:#9a9a9a;font-size:13px;margin-top:4px;">
    Structured events · Application log · Postgres event store
  </div>
  <div style="font-size:12px;color:#00ff41;margin-top:6px;">
    🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M:%S')}
  </div>
</div>
""", unsafe_allow_html=True)

events = observability.read_events(limit=max_rows)
app_log = observability.read_app_log(limit=max_rows)
pg_events = observability.read_pg_events(limit=max_rows)

# ── Top metrics ──────────────────────────────────────────────────────────
ev_df = pd.DataFrame(events) if events else pd.DataFrame()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events (file)", len(events))
c2.metric("Page views", int((ev_df["event_type"] == "page_view").sum()) if not ev_df.empty else 0)
c3.metric("Data fetches", int((ev_df["event_type"] == "data_fetch").sum()) if not ev_df.empty else 0)
c4.metric("Errors", int((ev_df["event_type"] == "exception").sum()) if not ev_df.empty else 0)

tab_events, tab_log, tab_pg = st.tabs(["📊 Events", "📜 App Log", "🗄️ Postgres"])

# ── Events tab ──────────────────────────────────────────────────────────
with tab_events:
    st.markdown("""
           <style>
               span[data-baseweb="tag"]{
                   background-color:#00ff41 !important;
               }
               span[data-baseweb="tag"] span{
                   color:#000000 !important;
                   font-weight:600 !important;
               }
               span[data-baseweb="tag"] svg{
                   fill:#000000 !important;
               }
               span[data-baseweb="tag"] [role="button"]:hover{
                   background-color:#00cc34 !important;
               }
           </style>
           """, unsafe_allow_html=True)
    if ev_df.empty:
        st.info("No events recorded yet. Navigate the app and refresh — events "
                "are written to logs/events.jsonl.")
    else:
        types = sorted(ev_df["event_type"].dropna().unique().tolist())
        levels = sorted(ev_df["level"].dropna().unique().tolist()) if "level" in ev_df else []
        fc1, fc2, fc3 = st.columns([2, 2, 3])
        sel_types = fc1.multiselect("Event type", types, default=types)
        sel_levels = fc2.multiselect("Level", levels, default=levels)
        search = fc3.text_input("Search (page / message / data)", "")

        view = ev_df.copy()
        if sel_types:
            view = view[view["event_type"].isin(sel_types)]
        if sel_levels and "level" in view:
            view = view[view["level"].isin(sel_levels)]
        if search:
            s = search.lower()
            mask = view.apply(
                lambda r: s in str(r.get("page", "")).lower()
                or s in str(r.get("message", "")).lower()
                or s in str(r.get("data", "")).lower(),
                axis=1,
            )
            view = view[mask]

        # newest first for display
        view = view.iloc[::-1].reset_index(drop=True)
        st.caption(f"Showing {len(view)} of {len(ev_df)} events")
        st.dataframe(
            view[[c for c in ("ts", "level", "event_type", "page",
                              "session_id", "message", "data") if c in view]],
            use_container_width=True, height=460,
        )

    st.download_button(
        "⬇️ Download events.jsonl",
        data="\n".join(__import__("json").dumps(e) for e in events),
        file_name="events.jsonl", mime="application/json",
        disabled=not events,
    )

# ── App log tab ──────────────────────────────────────────────────────────
with tab_log:
    if not app_log:
        st.info("logs/app.log is empty or not created yet.")
    else:
        st.code(app_log, language="log")
    st.download_button(
        "⬇️ Download app.log", data=app_log or "",
        file_name="app.log", mime="text/plain", disabled=not app_log,
    )

# ── Postgres tab ──────────────────────────────────────────────────────────
with tab_pg:
    if not pg_events:
        st.warning("No Postgres events — the database sink is unavailable "
                   "(check [database] credentials in secrets.toml, then use "
                   "'Retry Postgres sink' in the sidebar).")
    else:
        pg_df = pd.DataFrame(pg_events)
        st.caption(f"{len(pg_df)} rows from app_events (newest first)")
        st.dataframe(pg_df, use_container_width=True, height=500)
