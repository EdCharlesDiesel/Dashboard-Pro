"""Bloomberg-terminal palette + global CSS injection.

Call `BloombergTheme.apply()` once at the top of every Streamlit page after
`st.set_page_config(...)`. It hides Streamlit chrome and lays down the
green-on-black terminal palette plus a responsive grid system.
"""
from __future__ import annotations

import streamlit as st


class BloombergTheme:
    """Centralised colour tokens + CSS injection."""

    # ── palette ────────────────────────────────────────────────────────────
    BG          = "#000000"
    BG_ELEV     = "#0a0a0a"
    BG_PANEL    = "#0f0f0f"
    BG_HEADER   = "#001a08"
    BORDER      = "#2a2a2a"
    BORDER_HOT  = "#003a14"

    AMBER       = "#00ff41"     # primary action / accent — terminal phosphor green
    AMBER_DIM   = "#00a32a"
    CYAN        = "#00e0ff"     # info
    GREEN       = "#00ff66"     # bullish / pass
    RED         = "#ff3344"     # bearish / fail
    YELLOW      = "#ffcc00"     # warn / wait
    GREY        = "#9a9a9a"     # secondary text
    WHITE       = "#e6e6e6"
    PURPLE      = "#b266ff"

    FONT_MONO   = "'JetBrains Mono', 'Fira Code', 'Consolas', monospace"
    FONT_UI     = "'IBM Plex Sans', 'Inter', sans-serif"

    # ── public API ─────────────────────────────────────────────────────────
    @classmethod
    def apply(cls) -> None:
        st.markdown(cls._css(), unsafe_allow_html=True)

    # ── internal CSS builder ───────────────────────────────────────────────
    @classmethod
    def _css(cls) -> str:
        return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ── base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: {cls.FONT_MONO};
    font-size: 12px;
    letter-spacing: 0.02em;
}}
.stApp {{ background: {cls.BG} !important; color: {cls.WHITE}; }}
.block-container {{
    padding: 0.6rem 1rem 4rem 1rem !important;
    max-width: 100% !important;
}}

/* hide streamlit chrome */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stSidebarCollapsedControl"] {{ visibility: visible !important; display: flex !important; }}
[data-testid="stSidebarCollapseButton"] {{ visibility: visible !important; display: flex !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {cls.BG_ELEV} !important;
    border-right: 1px solid {cls.BORDER};
}}
section[data-testid="stSidebar"] * {{
    color: {cls.WHITE} !important;
    font-family: {cls.FONT_MONO} !important;
    font-size: 11px !important;
}}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    color: {cls.AMBER} !important;
    font-size: 11px !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid {cls.BORDER};
    padding-bottom: 4px;
}}

/* anything sitting on the green fill gets BLACK text — readability first.
   (The sidebar `*` rule above paints inner spans white, which made green
   primary buttons and the active page link unreadable.) */
.stButton button[kind="primary"],
.stButton button[kind="primary"] *,
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[kind="primary"] * {{
    color: {cls.BG} !important;
}}

/* sidebar page navigation links */
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
    padding: 2px 8px;
    border: 1px solid transparent;
    border-radius: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover {{
    background: {cls.BG_HEADER} !important;
    border-color: {cls.AMBER_DIM} !important;
}}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"]:hover * {{
    color: {cls.AMBER} !important;
}}
/* current page: solid green chip, black font */
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
    background: {cls.AMBER} !important;
}}
section[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] * {{
    color: {cls.BG} !important;
}}

/* widgets — flat, bordered, monospace */
.stTextInput input, .stNumberInput input, .stSelectbox > div,
.stTextArea textarea {{
    background: {cls.BG_PANEL} !important;
    border: 1px solid {cls.BORDER} !important;
    color: {cls.WHITE} !important;
    border-radius: 0 !important;
    font-family: {cls.FONT_MONO} !important;
}}
.stButton button {{
    background: {cls.BG_PANEL} !important;
    color: {cls.AMBER} !important;
    border: 1px solid {cls.AMBER_DIM} !important;
    border-radius: 0 !important;
    font-family: {cls.FONT_MONO} !important;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 11px !important;
}}
.stButton button:hover {{
    background: {cls.BG_HEADER} !important;
    border-color: {cls.AMBER} !important;
    color: {cls.AMBER} !important;
}}
.stButton button[kind="primary"] {{
    background: {cls.AMBER} !important;
    color: {cls.BG} !important;
    border: 1px solid {cls.AMBER} !important;
}}
[data-baseweb="radio"] label, [data-baseweb="checkbox"] label {{
    color: {cls.WHITE} !important;
    font-family: {cls.FONT_MONO} !important;
}}
.stCheckbox > label {{ font-size: 11px !important; }}

/* tab bar */
.stTabs [data-baseweb="tab-list"] {{
    background: {cls.BG_PANEL};
    border-bottom: 2px solid {cls.AMBER_DIM};
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {cls.GREY};
    border-radius: 0;
    padding: 8px 16px;
    font-family: {cls.FONT_MONO};
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 11px;
}}
.stTabs [aria-selected="true"] {{
    background: {cls.BG_HEADER} !important;
    color: {cls.AMBER} !important;
    border-bottom: 2px solid {cls.AMBER};
}}

/* dataframes */
[data-testid="stDataFrame"] {{
    background: {cls.BG_PANEL};
    border: 1px solid {cls.BORDER};
}}

/* ── Bloomberg-terminal components ────────────────────────────── */

/* Top ticker strip — scrolling marquee feel */
.bb-ticker-strip {{
    background: {cls.BG_HEADER};
    border-bottom: 1px solid {cls.AMBER_DIM};
    padding: 6px 12px;
    font-family: {cls.FONT_MONO};
    font-size: 11px;
    color: {cls.AMBER};
    letter-spacing: 0.08em;
    display: flex;
    gap: 18px;
    align-items: center;
    overflow-x: auto;
    white-space: nowrap;
    scrollbar-width: thin;
}}
.bb-ticker-pill {{
    color: {cls.WHITE};
}}
.bb-ticker-pill .bb-up   {{ color: {cls.GREEN}; }}
.bb-ticker-pill .bb-down {{ color: {cls.RED}; }}
.bb-ticker-pill .bb-sym  {{ color: {cls.AMBER}; font-weight: 600; }}

/* Command bar — Bloomberg's iconic input strip */
.bb-cmd-bar {{
    display: flex;
    gap: 0;
    align-items: stretch;
    background: {cls.BG_PANEL};
    border: 1px solid {cls.AMBER_DIM};
    margin: 6px 0 10px 0;
}}
.bb-cmd-bar .bb-cmd-label {{
    background: {cls.AMBER};
    color: {cls.BG};
    padding: 4px 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    font-size: 11px;
}}
.bb-cmd-bar .bb-cmd-cell {{
    padding: 4px 12px;
    color: {cls.AMBER};
    border-left: 1px solid {cls.BORDER};
    font-family: {cls.FONT_MONO};
    font-size: 11px;
    flex: 1;
}}
.bb-cmd-bar .bb-cmd-cell.bb-cyan  {{ color: {cls.CYAN}; }}
.bb-cmd-bar .bb-cmd-cell.bb-green {{ color: {cls.GREEN}; }}
.bb-cmd-bar .bb-cmd-cell.bb-red   {{ color: {cls.RED}; }}

/* Panel — bordered window with a labelled header */
.bb-panel {{
    border: 1px solid {cls.BORDER};
    background: {cls.BG_PANEL};
    margin-bottom: 10px;
    display: flex;
    flex-direction: column;
}}
.bb-panel-header {{
    background: {cls.BG_HEADER};
    color: {cls.AMBER};
    padding: 4px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    border-bottom: 1px solid {cls.AMBER_DIM};
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.bb-panel-header .bb-tag {{ color: {cls.GREY}; font-weight: 500; }}
.bb-panel-body {{
    padding: 8px 12px;
    color: {cls.WHITE};
    font-size: 12px;
    line-height: 1.45;
    flex: 1;
}}

/* Metric cell — compact KPI tile */
.bb-metric {{
    border: 1px solid {cls.BORDER};
    background: {cls.BG_PANEL};
    padding: 8px 10px;
    text-align: left;
    min-height: 60px;
}}
.bb-metric-label {{
    color: {cls.GREY};
    font-size: 9px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
.bb-metric-value {{
    color: {cls.AMBER};
    font-family: {cls.FONT_MONO};
    font-size: 18px;
    font-weight: 700;
    line-height: 1.1;
}}
.bb-metric-value.bb-green {{ color: {cls.GREEN}; }}
.bb-metric-value.bb-red   {{ color: {cls.RED}; }}
.bb-metric-value.bb-cyan  {{ color: {cls.CYAN}; }}
.bb-metric-value.bb-yellow {{ color: {cls.YELLOW}; }}
.bb-metric-value.bb-white  {{ color: {cls.WHITE}; }}
.bb-metric-delta {{
    font-size: 10px;
    font-family: {cls.FONT_MONO};
    color: {cls.GREY};
    margin-top: 2px;
}}

/* Chips — status pills */
.bb-chip {{
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid;
    font-family: {cls.FONT_MONO};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
.bb-chip-go    {{ color: {cls.GREEN};  border-color: {cls.GREEN};  background: rgba(0,255,102,0.06); }}
.bb-chip-wait  {{ color: {cls.YELLOW}; border-color: {cls.YELLOW}; background: rgba(255,204,0,0.06); }}
.bb-chip-no    {{ color: {cls.RED};    border-color: {cls.RED};    background: rgba(255,51,68,0.06); }}
.bb-chip-info  {{ color: {cls.CYAN};   border-color: {cls.CYAN};   background: rgba(0,224,255,0.06); }}
.bb-chip-amber {{ color: {cls.AMBER};  border-color: {cls.AMBER};  background: rgba(0,255,65,0.06); }}

/* Status bar — page-bottom strip */
.bb-status-bar {{
    position: sticky;
    bottom: 0;
    background: {cls.BG_HEADER};
    border-top: 1px solid {cls.AMBER_DIM};
    padding: 4px 12px;
    color: {cls.AMBER};
    font-family: {cls.FONT_MONO};
    font-size: 10px;
    letter-spacing: 0.1em;
    display: flex;
    gap: 24px;
    align-items: center;
    z-index: 50;
}}
.bb-status-bar .bb-sep {{ color: {cls.BORDER}; }}

/* Function code grid — Bloomberg <PgUp>/<F1>-style shortcut blocks */
.bb-fn-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 4px;
    margin: 6px 0 14px 0;
}}
.bb-fn-tile {{
    background: {cls.BG_PANEL};
    border: 1px solid {cls.BORDER};
    padding: 6px 8px;
    cursor: pointer;
    transition: all .12s;
}}
.bb-fn-tile:hover {{
    border-color: {cls.AMBER};
    background: {cls.BG_HEADER};
}}
.bb-fn-tile a {{ color: inherit; text-decoration: none; display: block; }}
.bb-fn-code {{
    color: {cls.AMBER};
    font-family: {cls.FONT_MONO};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
}}
.bb-fn-label {{
    color: {cls.WHITE};
    font-size: 11px;
    margin-top: 2px;
}}

/* Progress bar */
.bb-progress {{
    background: {cls.BG_PANEL};
    border: 1px solid {cls.BORDER};
    height: 8px;
    overflow: hidden;
}}
.bb-progress-fill {{ height: 100%; transition: width .3s; }}

/* Data row — table row with monospace alignment */
.bb-data-row {{
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    border-bottom: 1px dotted {cls.BORDER};
    font-family: {cls.FONT_MONO};
    font-size: 11px;
}}
.bb-data-row:last-child {{ border-bottom: none; }}
.bb-data-key   {{ color: {cls.GREY}; }}
.bb-data-val   {{ color: {cls.WHITE}; font-weight: 600; }}
.bb-data-val.bb-green {{ color: {cls.GREEN}; }}
.bb-data-val.bb-red   {{ color: {cls.RED};   }}
.bb-data-val.bb-amber {{ color: {cls.AMBER}; }}
.bb-data-val.bb-cyan  {{ color: {cls.CYAN};  }}

/* Mosaic grid container */
.bb-mosaic {{
    display: grid;
    gap: 10px;
}}
.bb-mosaic-2 {{ grid-template-columns: 1fr 1fr; }}
.bb-mosaic-3 {{ grid-template-columns: repeat(3, 1fr); }}
.bb-mosaic-4 {{ grid-template-columns: repeat(4, 1fr); }}

/* Responsive collapse */
@media (max-width: 1100px) {{
    .bb-mosaic-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .bb-mosaic-3 {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 720px) {{
    .bb-mosaic-2, .bb-mosaic-3, .bb-mosaic-4 {{ grid-template-columns: 1fr; }}
    .bb-cmd-bar {{ flex-wrap: wrap; }}
    .bb-status-bar {{ flex-wrap: wrap; gap: 8px; }}
    .block-container {{ padding: 0.4rem 0.4rem 5rem 0.4rem !important; }}
}}

/* ── global flatten — terminals have no rounded corners ──────── */
*, *::before, *::after {{ border-radius: 0 !important; }}

/* ── legacy page classes harmonised to the terminal look ─────── */
.hero {{
    background: linear-gradient(90deg, {cls.BG_HEADER} 0%, {cls.BG} 100%) !important;
    border: 1px solid {cls.AMBER_DIM} !important;
    border-left: 4px solid {cls.AMBER} !important;
}}
.card, .metric-box, .ccy-card, .ccy-bank, .stat-box {{
    background: {cls.BG_PANEL} !important;
    border: 1px solid {cls.BORDER} !important;
}}
.section-title, .card-header {{ color: {cls.AMBER} !important; }}

/* Bloomberg-tinted Plotly background passes through automatically */
.js-plotly-plot {{ border: 1px solid {cls.BORDER}; background: {cls.BG_PANEL}; }}

/* Hero header */
.bb-hero {{
    background: linear-gradient(90deg, {cls.BG_HEADER} 0%, {cls.BG} 100%);
    border: 1px solid {cls.AMBER_DIM};
    border-left: 4px solid {cls.AMBER};
    padding: 10px 16px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}}
.bb-hero-title {{
    color: {cls.AMBER};
    font-family: {cls.FONT_MONO};
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}
.bb-hero-sub {{
    color: {cls.GREY};
    font-family: {cls.FONT_MONO};
    font-size: 11px;
    margin-top: 2px;
}}
.bb-hero-right {{
    text-align: right;
    font-family: {cls.FONT_MONO};
}}
.bb-hero-symbol {{
    color: {cls.WHITE};
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
.bb-hero-ticker {{
    color: {cls.CYAN};
    font-size: 11px;
}}
.bb-hero-side {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.15em;
    margin-top: 4px;
}}
.bb-hero-side.bb-long  {{ color: {cls.GREEN}; }}
.bb-hero-side.bb-short {{ color: {cls.RED};   }}
</style>
"""
