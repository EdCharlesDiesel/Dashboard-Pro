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
        # Universal chokepoint: every page (framework + legacy) calls apply()
        # right after set_page_config, so this is where we initialise logging
        # and record a page view. Guarded — observability must never break a
        # page render.
        try:
            from src.core import observability
            observability.init_logging()
            observability.log_page_view()
        except Exception:
            pass
        st.markdown(cls._css(), unsafe_allow_html=True)
        cls._render_connection_banner()

    @classmethod
    def _render_connection_banner(cls) -> None:
        """Site-wide Postgres connectivity indicator: a thick red line pinned to
        the top of the page when the DB isn't reachable, a thin green line when
        it is. ``apply()`` is the one chokepoint every page (framework +
        legacy — 53 call sites) shares, so injecting it here makes it universal
        without touching each page.

        Calls ``auto_connect()`` itself (idempotent after the first call per
        session, never raises) rather than trusting ``db_ok`` to already be
        set — most pages never otherwise trigger the DB bootstrap, so without
        this the banner would read a stale/unset gate on everything except
        the checklist, Trend Signals, and Trade Journal.
        """
        try:
            from src.db.connection import auto_connect
            ok = auto_connect()
        except Exception:
            ok = False
        if ok:
            color, height, glow = cls.GREEN, "2px", "none"
        else:
            color, height, glow = cls.RED, "6px", f"0 0 10px {cls.RED}"
        st.markdown(
            f'<div style="position:fixed;top:0;left:0;right:0;height:{height};'
            f'background:{color};box-shadow:{glow};z-index:1000001;'
            f'pointer-events:none;"></div>',
            unsafe_allow_html=True,
        )

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

/* hide streamlit chrome — but NEVER the header OR toolbar wholesale. In
   Streamlit 1.57 the collapsed-sidebar EXPAND arrow (stExpandSidebarButton) is
   rendered INSIDE stToolbar, as a sibling of the Deploy/menu cluster. A
   `display:none` on stToolbar removes that whole subtree — the arrow included —
   so a collapsed sidebar can never be reopened. Hide only the individual chrome
   items and leave the toolbar + expand arrow intact. */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stMainMenu"],
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stSidebarNav"] {{ display: none; }}
/* Keep the header AND toolbar visible — the sidebar expand arrow lives inside
   the toolbar. Legacy pages hide the header with `header{{visibility:hidden}}`;
   override that so a collapsed sidebar can always be reopened. */
header[data-testid="stHeader"] {{
    background: transparent !important;
    visibility: visible !important;
}}
[data-testid="stToolbar"] {{
    background: transparent !important;
    visibility: visible !important;
    display: flex !important;
}}

/* Sidebar APPEARANCE ONLY — never pin it open. The previous build forced
   `transform: none` + a fixed 250px width, which made the collapse arrow do
   nothing on every viewport (desktop included) and swallowed mobile screens.
   We now leave Streamlit's native collapse/expand intact and only restyle the
   surface. */
section[data-testid="stSidebar"] {{
    background: {cls.BG_ELEV} !important;
    border-right: 1px solid {cls.BORDER};
}}
/* Application logo — rendered once at the very top of the sidebar (above every
   page's own controls) via a ::before on the user-content container, so it is
   uniform across every page from this single rule. The artwork mirrors
   assets/logo.svg, inlined as a data URI so it renders without Streamlit static
   file serving. */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]::before {{
    content: "";
    display: block;
    height: 46px;
    margin: 2px 0 12px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid {cls.BORDER};
    background-repeat: no-repeat;
    background-position: left center;
    background-size: contain;
    background-image: url("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' width='216' height='50' viewBox='0 0 216 50'><rect x='1' y='5' width='40' height='40' fill='%230f0f0f' stroke='%2300ff41' stroke-width='1.5'/><line x1='11' y1='14' x2='11' y2='40' stroke='%2300ff41' stroke-width='1'/><rect x='9' y='22' width='4' height='13' fill='%2300ff41'/><line x1='21' y1='10' x2='21' y2='40' stroke='%2300ff66' stroke-width='1'/><rect x='19' y='15' width='4' height='20' fill='%2300ff66'/><line x1='31' y1='18' x2='31' y2='40' stroke='%23ff3344' stroke-width='1'/><rect x='29' y='25' width='4' height='10' fill='%23ff3344'/><text x='51' y='25' font-family='JetBrains Mono,Consolas,monospace' font-size='16' font-weight='700' letter-spacing='1' fill='%23e6e6e6'>DASHBOARD<tspan fill='%2300ff41'> PRO</tspan></text><text x='52' y='40' font-family='JetBrains Mono,Consolas,monospace' font-size='8.5' letter-spacing='2.5' fill='%239a9a9a'>TRADING TERMINAL</text></svg>");
}}
/* Always-tappable toggle controls so the sidebar can be hidden AND reopened,
   even on touch (the arrow is normally hover-only) and on legacy pages that
   hide the header. Force terminal-green so the COLLAPSED-state expand control —
   which renders OUTSIDE the sidebar, where the sidebar `*` colour rule never
   reaches it — is visible instead of black-on-black. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] button {{
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    color: {cls.AMBER} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {cls.WHITE} !important;
    font-family: {cls.FONT_MONO} !important;
    font-size: 11px !important;
}}
/* re-assert the icon font on sidebar Material icons in general (the monospace
   `*` rule above otherwise breaks every ligature). */
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Symbols Sharp' !important;
}}
/* The collapse/expand ARROW specifically. Two bugs to beat:
   1. the sidebar monospace `*` rule turns the glyph into the literal text
      "keyboard_double_arrow_left";
   2. Streamlit hardcodes the EXPAND arrow's colour to `fadedText60`, which is
      near-invisible on this black theme — so a collapsed sidebar looks like it
      has no reopen control at all.
   Target EVERY descendant (`*`) of the toggle controls — these buttons contain
   only the glyph, so the universal selector is safe and matches whatever
   element (span / i / svg) this Streamlit version wraps the icon in. Re-assert
   the icon font, force terminal-green, and bump the size so it's unmissable. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] *,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"] * {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Symbols Sharp' !important;
    color: {cls.AMBER} !important;
    fill: {cls.AMBER} !important;
    font-size: 1.6rem !important;
    opacity: 1 !important;
}}
/* Give the COLLAPSED-state reopen button a visible chip so it can never blend
   into the background, and pin it on top of page content. */
[data-testid="stExpandSidebarButton"] {{
    background: {cls.BG_PANEL} !important;
    border: 1px solid {cls.AMBER} !important;
    border-radius: 0 !important;
    z-index: 1000000 !important;
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
   primary buttons and the active page link unreadable.) This must cover
   EVERY primary-variant widget, not just .stButton: st.form_submit_button
   renders kind="primaryFormSubmit" inside stFormSubmitButton (the "Apply
   Filter" white-on-green bug), and st.download_button has its own wrapper.
   `kind^="primary"` matches all of them, in and out of the sidebar. */
button[kind^="primary"],
button[kind^="primary"] *,
[data-testid^="stBaseButton-primary"],
[data-testid^="stBaseButton-primary"] *,
section[data-testid="stSidebar"] button[kind^="primary"],
section[data-testid="stSidebar"] button[kind^="primary"] *,
[data-testid="stFormSubmitButton"] button[kind^="primary"],
[data-testid="stFormSubmitButton"] button[kind^="primary"] *,
.stDownloadButton button[kind^="primary"],
.stDownloadButton button[kind^="primary"] * {{
    color: {cls.BG} !important;
}}
/* …and give the form-submit primary the same flat green treatment as
   .stButton primary so the two button kinds are visually identical. */
[data-testid="stFormSubmitButton"] button[kind^="primary"] {{
    background: {cls.AMBER} !important;
    border: 1px solid {cls.AMBER} !important;
    border-radius: 0 !important;
    font-family: {cls.FONT_MONO} !important;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-size: 11px !important;
}}
/* multiselect / tag chips sit on the green accent too — black text and a
   black ✕ icon everywhere (previously only patched page-locally). */
span[data-baseweb="tag"] {{
    background: {cls.AMBER} !important;
}}
span[data-baseweb="tag"],
span[data-baseweb="tag"] *,
section[data-testid="stSidebar"] span[data-baseweb="tag"],
section[data-testid="stSidebar"] span[data-baseweb="tag"] * {{
    color: {cls.BG} !important;
}}
span[data-baseweb="tag"] svg {{ fill: {cls.BG} !important; }}

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

/* ── Responsive / mobile-first collapse ───────────────────────── */
@media (max-width: 1100px) {{
    .bb-mosaic-4 {{ grid-template-columns: repeat(2, 1fr); }}
    .bb-mosaic-3 {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 768px) {{
    .bb-mosaic-2, .bb-mosaic-3, .bb-mosaic-4 {{ grid-template-columns: 1fr; }}
    .bb-cmd-bar {{ flex-wrap: wrap; }}
    .bb-status-bar {{ flex-wrap: wrap; gap: 8px; }}
    .block-container {{ padding: 0.4rem 0.5rem 5rem 0.5rem !important; }}

    /* Stack every Streamlit st.columns() row — by default the horizontal
       block keeps its children side-by-side and squeezes them to unusable
       widths on a phone. Force a single column. (Layout only — no Python
       change to the [3,2] / [0.06,0.94] column ratios.) */
    [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
    [data-testid="stColumn"],
    [data-testid="column"] {{
        flex: 1 1 100% !important;
        min-width: 100% !important;
        width: 100% !important;
    }}

    /* Larger touch targets + scrollable tab bar */
    .stButton button {{ min-height: 42px; }}
    .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; }}
    .stTabs [data-baseweb="tab"] {{ padding: 8px 12px; white-space: nowrap; }}

    /* Trim oversized hero / metric type for small screens */
    .bb-hero-title  {{ font-size: 13px; }}
    .bb-hero-symbol {{ font-size: 18px; }}
    .bb-metric-value {{ font-size: 15px; }}
    .bb-ticker-strip {{ font-size: 10px; gap: 12px; }}
}}

/* Phones & small tablets: make the sidebar an off-canvas drawer that overlays
   the content (instead of reserving a column) and slides fully away when
   collapsed, so it never swallows a narrow screen.

   Two touch-specific fixes (Streamlit 1.57 DOM):
     • The collapse arrow (stSidebarCollapseButton) is only revealed on hover,
       which never fires on touch — force it visible so there IS a control to
       tap to close the drawer.
     • When collapsed, the reopen control is stExpandSidebarButton (lives in
       the header) — float it top-left, above the content, so the drawer can
       be brought back. */
@media (max-width: 768px) {{
    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {{
        position: fixed !important;
        top: 0; bottom: 0; left: 0;
        width: 85vw !important;
        min-width: 85vw !important;
        max-width: 320px !important;
        z-index: 1000 !important;
        box-shadow: 2px 0 18px rgba(0, 0, 0, 0.7);
    }}
    section[data-testid="stSidebar"][aria-expanded="false"] {{
        transform: translateX(-100%) !important;
        margin-left: 0 !important;
        box-shadow: none !important;
    }}
    /* the collapse (close) arrow inside the drawer — always tappable */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button {{
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
    }}
    /* the reopen control floats over the content, top-left */
    [data-testid="stExpandSidebarButton"] {{
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        position: fixed !important;
        top: 6px;
        left: 6px;
        z-index: 1001 !important;
        background: {cls.BG_HEADER} !important;
        border: 1px solid {cls.AMBER_DIM} !important;
    }}
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
