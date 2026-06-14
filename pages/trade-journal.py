import streamlit as st
from src.ui.theme import BloombergTheme
from src.pages_lib.navigation import render_sidebar_nav
from src.instruments.registry import INSTRUMENTS
from src.core import secrets
from src.db.trade_repository import TradeRepository, DBConfig
from src.services import mt4_import
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import psycopg2.extras
from datetime import datetime

st.set_page_config(
    page_title="Trade Journal · Trading System",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)
BloombergTheme.apply()

st.markdown("""
<style>
    /* Theme-adaptive layout vars */
    .stApp {
      --border: color-mix(in srgb, var(--text-color) 12%, transparent);
      --muted:  color-mix(in srgb, var(--text-color) 55%, transparent);
    }
    html,body,[class*="css"]{font-family:'JetBrains Mono','Fira Code',monospace;}
    .stApp{background:var(--background-color);}
    section[data-testid="stSidebar"]{background:var(--secondary-background-color)!important;border-right:1px solid var(--border,#2a2a2a);}
    .card{background:var(--secondary-background-color);border:1px solid var(--border,#2a2a2a);border-radius:12px;padding:20px;margin-bottom:16px;}
    .card-header{font-size:12px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#9a9a9a;margin-bottom:14px;}
    .hero{background:linear-gradient(135deg,#000000 0%,#0a0a0a 50%,#000000 100%);border:1px solid #2a2a2a;border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
    .hero::before{content:'';position:absolute;top:-40%;right:-5%;width:300px;height:300px;background:radial-gradient(circle,rgba(56,139,253,.07) 0%,transparent 70%);border-radius:50%;}
    .metric-box{background:var(--background-color);border:1px solid var(--border,#2a2a2a);border-radius:8px;padding:14px;text-align:center;}
    .metric-value{font-size:22px;font-weight:700;color:#e6e6e6;}
    .metric-label{font-size:10px;color:#9a9a9a;margin-top:2px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;}
    .section-title{font-size:15px;font-weight:700;color:#e6e6e6;margin:20px 0 12px 0;padding-left:4px;border-left:3px solid #00ff41;}
    .outcome-win{background:#0d2f1a;color:#00ff66;border:1px solid #00a32a;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:700;display:inline-block;}
    .outcome-loss{background:#2f0d0d;color:#ff3344;border:1px solid #8b2d2d;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:700;display:inline-block;}
    .outcome-be{background:#1a1a0d;color:#ffcc00;border:1px solid #9e6a03;border-radius:5px;padding:2px 9px;font-size:11px;font-weight:700;display:inline-block;}
    [data-testid="stSidebarNav"]{display:none;}
    #MainMenu,footer,header{visibility:hidden;}
    [data-testid="stSidebarCollapsedControl"]{visibility:visible !important;}
    [data-testid="stSidebarCollapseButton"]{visibility:visible !important;display:flex !important;}
    section[data-testid="stSidebar"]{display:block !important;}
    .block-container{padding-top:1.5rem;max-width:1400px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# DB HELPERS
# ══════════════════════════════════════════════════════════════════

def get_db_connection(cfg):
    return psycopg2.connect(
        host=cfg["host"], port=cfg["port"],
        dbname=cfg["dbname"], user=cfg["user"], password=cfg["password"]
    )


def load_journal_trades(cfg, limit: int = 500):
    try:
        conn = get_db_connection(cfg)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, logged_at, instrument, direction, session,
                   score, verdict, sl_pips, tp1_pips, tp2_pips, rr_tp1,
                   lot_size, risk_amount,
                   entry_price, close_price, outcome, pips_gained, r_multiple,
                   is_open, checks_passed, checks_total, source, notes
            FROM trade_setups
            ORDER BY logged_at ASC
            LIMIT %s
        """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    except Exception:
        return []


def render_mt4_import(cfg):
    """Upload + parse + map + persist an MT4 HTML statement. Imported trades are
    tagged source='mt4_import' so they're filterable and never counted in the
    pre-trade workflow-quality stats."""
    with st.expander("📥 Import MT4 trade history (HTML statement)", expanded=False):
        st.caption("In MT4: **Terminal → Account History → right-click → Save as Report** "
                   "(or *Save as Detailed Report*). Upload the resulting `.htm` file here.")

        up = st.file_uploader("MT4 statement", type=["htm", "html"], key="mt4_file",
                              label_visibility="collapsed")
        offset = st.number_input(
            "Broker server UTC offset (hours)", -12.0, 14.0, 0.0, 0.5,
            help="MT4 times are in broker server time. Most brokers are UTC+2 (or +3 in summer). "
                 "Set this so sessions line up — leave 0 if your broker reports UTC.")
        if up is None:
            return

        try:
            trades = mt4_import.parse_mt4_html(up.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't parse this file: {exc}")
            return
        if not trades:
            st.warning("No closed trades found in this statement. "
                       "Make sure you exported the **Account History** tab.")
            return

        st.success(f"Parsed **{len(trades)}** closed trades.")

        # ── Symbol mapping (auto, user-correctable) ───────────────────
        auto = mt4_import.build_symbol_map([t["item"] for t in trades])
        st.markdown("**Symbol mapping** — fix any row that didn't auto-map:")
        map_df = pd.DataFrame([{"MT4 symbol": k, "Maps to": (v or "")} for k, v in auto.items()])
        edited = st.data_editor(
            map_df, hide_index=True, use_container_width=True, key="mt4_map",
            column_config={"MT4 symbol": st.column_config.TextColumn(disabled=True),
                           "Maps to": st.column_config.SelectboxColumn(
                               options=[""] + list(INSTRUMENTS.keys()), required=False)})
        symbol_map = {r["MT4 symbol"]: (r["Maps to"] or None) for _, r in edited.iterrows()}

        rows, skipped = mt4_import.to_journal_rows(trades, symbol_map, offset)

        # ── Dedupe against what's already imported ────────────────────
        try:
            existing = TradeRepository(DBConfig.from_mapping(cfg)).imported_tickets()
        except Exception:
            existing = set()
        new_rows = [r for r in rows if r["ticket"] not in existing]
        dup = len(rows) - len(new_rows)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mappable", len(rows))
        c2.metric("New to import", len(new_rows))
        c3.metric("Already imported", dup)
        if skipped:
            st.warning("Skipped (unmapped symbol): "
                       + ", ".join(f"{k} × {v}" for k, v in skipped.items()))

        if new_rows:
            prev = pd.DataFrame(new_rows)[
                ["logged_at", "instrument", "direction", "session", "lot_size",
                 "entry_price", "close_price", "pips_gained", "r_multiple", "outcome"]]
            st.dataframe(prev.head(50), use_container_width=True, hide_index=True)
            st.caption("R-multiple is blank where the MT4 row had no Stop Loss (can't infer risk).")

            if st.button(f"✅ Import {len(new_rows)} trades", type="primary", key="mt4_go"):
                try:
                    repo = TradeRepository(DBConfig.from_mapping(cfg))
                    repo.init_schema()          # ensure table + source column exist
                    n = repo.import_mt4_rows(new_rows)
                    st.success(f"Imported {n} trades. Reloading…")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Import failed: {exc}")
        else:
            st.info("Nothing new to import — every parsed trade is already in the journal.")


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📓 Trade Journal")

    st.markdown("**🗄️ PostgreSQL Database**")
    _db = secrets.db_config()  # defaults from .streamlit/secrets.toml [database]
    _host = st.text_input("Host", value=st.session_state.get("db_host", _db["host"]), key="jdb_host")
    _port = st.number_input("Port", value=int(st.session_state.get("db_port", _db["port"])), step=1, key="jdb_port")
    _name = st.text_input("Database", value=st.session_state.get("db_name", _db["dbname"]), key="jdb_name")
    _user = st.text_input("User", value=st.session_state.get("db_user", _db["user"]), key="jdb_user")
    _pass = st.text_input("Password", value=st.session_state.get("db_pass", _db["password"]), type="password",
                          key="jdb_pass")

    db_cfg = {"host": _host, "port": int(_port), "dbname": _name, "user": _user, "password": _pass}

    if st.button("🔌 Connect & Load", use_container_width=True, type="primary"):
        try:
            conn = get_db_connection(db_cfg)
            conn.close()
            st.session_state.journal_db_ok = True
            st.session_state.journal_db_cfg = db_cfg
            st.success("✅ Connected")
        except Exception as e:
            st.session_state.journal_db_ok = False
            st.error(str(e)[:60])

    st.divider()
    show_n = st.slider("Max trades to load", 50, 500, 200, step=50)
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.divider()
    render_sidebar_nav()
# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero">
  <div style="font-size:26px;font-weight:700;color:#e6e6e6;">📓 Trade Journal</div>
  <div style="color:#9a9a9a;font-size:13px;margin-top:4px;">
    Equity Curve · Performance Breakdown · Win Rate Feedback Loop
  </div>
  <div style="font-size:12px;color:#00ff41;margin-top:6px;">
    🕐 {datetime.now().strftime('%A, %d %B %Y  |  %H:%M')} · Target: 66% Win Rate
  </div>
</div>
""", unsafe_allow_html=True)

# ── Guard: DB connection ───────────────────────────────────────────
if not st.session_state.get("journal_db_ok"):
    st.info("🔌 Connect to PostgreSQL in the sidebar to load your trade journal.")
    st.stop()

_cfg = st.session_state.get("journal_db_cfg", db_cfg)

render_mt4_import(_cfg)

raw = load_journal_trades(_cfg, limit=show_n)

if not raw:
    st.warning("No trades found yet. Import an MT4 statement above, or save setups from the Checklist page.")
    st.stop()

df_all = pd.DataFrame(raw)
df_all["logged_at"] = pd.to_datetime(df_all["logged_at"])
if "source" not in df_all.columns:
    df_all["source"] = "checklist"
df_all["source"] = df_all["source"].fillna("checklist")

# Closed trades only for analytics. Imported broker fills without a Stop Loss
# carry a null R — treat as 0R for cumulative math; they still count by outcome.
df_closed = df_all[df_all["outcome"].notna() & (df_all["is_open"] == False)].copy()
df_closed["r_multiple"] = df_closed["r_multiple"].fillna(0)
df_open = df_all[df_all["is_open"] != False].copy()

n_total = len(df_all)
n_closed = len(df_closed)
n_open = len(df_open)

# ── Compute stats ──────────────────────────────────────────────────
if n_closed > 0:
    wins = df_closed[df_closed["outcome"] == "WIN"]
    losses = df_closed[df_closed["outcome"] == "LOSS"]
    be_trades = df_closed[df_closed["outcome"] == "BE"]
    win_rate = len(wins) / n_closed * 100
    avg_win_r = wins["r_multiple"].mean() if len(wins) else 0
    avg_los_r = losses["r_multiple"].abs().mean() if len(losses) else 1
    loss_rate = len(losses) / n_closed
    expectancy = (win_rate / 100 * avg_win_r) - (loss_rate * abs(avg_los_r))
    pf = (len(wins) * avg_win_r) / (len(losses) * abs(avg_los_r)) \
        if len(losses) > 0 and avg_los_r != 0 else 0
    total_r = df_closed["r_multiple"].sum()

    # Equity curve + drawdown
    df_closed = df_closed.sort_values("logged_at").reset_index(drop=True)
    df_closed["cum_r"] = df_closed["r_multiple"].cumsum()
    df_closed["run_max"] = df_closed["cum_r"].cummax()
    df_closed["drawdown"] = df_closed["run_max"] - df_closed["cum_r"]
    max_dd = df_closed["drawdown"].max()
else:
    win_rate = avg_win_r = avg_los_r = expectancy = pf = total_r = max_dd = 0
    wins = pd.DataFrame()
    losses = pd.DataFrame()
    be_trades = pd.DataFrame()

target = 66.0
wr_color = "#00ff66" if win_rate >= target else "#ff3344"

# ══════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
kpis = [
    (k1, f"{win_rate:.1f}%", "Win Rate", wr_color),
    (k2, f"{len(wins) if n_closed else 0}W/{len(losses) if n_closed else 0}L/{len(be_trades) if n_closed else 0}BE",
     "W / L / BE", "#e6e6e6"),
    (k3, f"{n_closed}", "Closed", "#9a9a9a"),
    (k4, f"{n_open}", "Open", "#00ff41"),
    (k5, f"{expectancy:+.2f}R", "Expectancy", "#00ff66" if expectancy > 0 else "#ff3344"),
    (k6, f"{pf:.2f}", "Profit Factor", "#00ff66" if pf >= 1.5 else "#ffcc00" if pf >= 1 else "#ff3344"),
    (k7, f"{total_r:+.1f}R", "Total R", "#00ff66" if total_r > 0 else "#ff3344"),
]
for col, val, lbl, color in kpis:
    with col:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:{color};">{val}</div>'
            f'<div class="metric-label">{lbl}</div></div>',
            unsafe_allow_html=True)

# Win rate vs target banner
banner_c = "#003a14" if win_rate >= target else "#4a0d0d"
banner_b = "#00a32a" if win_rate >= target else "#8b2d2d"
banner_icon = "🎯" if win_rate >= target else "⚠️"
st.markdown(f"""
<div style="background:{banner_c};border:1px solid {banner_b};border-radius:10px;
            padding:10px 20px;margin:16px 0;display:flex;align-items:center;gap:12px;">
  <span style="font-size:20px;">{banner_icon}</span>
  <span style="color:{wr_color};font-weight:700;font-size:14px;">
    {win_rate:.1f}% Win Rate &nbsp;{'≥' if win_rate >= target else '<'} &nbsp;{target:.0f}% target
    &nbsp;·&nbsp; {n_closed} closed trades analysed
    &nbsp;·&nbsp; Max drawdown {max_dd:.2f}R
  </span>
</div>
""", unsafe_allow_html=True)

if n_closed == 0:
    st.info("No closed trades yet — close trades from the Checklist to see performance analytics.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_equity, tab_session, tab_pair, tab_dir, tab_log = st.tabs([
    "📈 Equity Curve", "🕐 By Session", "💱 By Pair", "↕️ By Direction", "📋 Trade Log"
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — EQUITY CURVE
# ──────────────────────────────────────────────────────────────────
with tab_equity:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=["Cumulative R (Equity Curve)", "Drawdown (R)", "Win / Loss per Trade"],
    )

    dates = df_closed["logged_at"]

    # ── Equity line with gradient fill ────────────────────────────
    cum_r = df_closed["cum_r"].tolist()
    x_dates = dates.tolist()
    fill_colors = []
    _s = 0
    for _i in range(1, len(cum_r) + 1):
        _end = _i == len(cum_r)
        _changed = _end or ((cum_r[_i] >= 0) != (cum_r[_s] >= 0))
        if _changed:
            _clr = "rgba(63,185,80,0.12)" if cum_r[_s] >= 0 else "rgba(248,81,73,0.12)"
            fig.add_trace(go.Scatter(
                x=x_dates[_s:_i] + x_dates[_s:_i][::-1],
                y=cum_r[_s:_i] + [0] * (_i - _s),
                fill="toself", fillcolor=_clr,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=1, col=1)
            _s = _i

    fig.add_trace(go.Scatter(
        x=dates, y=df_closed["cum_r"],
        mode="lines+markers",
        line=dict(color="#00ff41", width=2.5),
        marker=dict(
            color=["#00ff66" if o == "WIN" else "#ff3344" if o == "LOSS" else "#ffcc00"
                   for o in df_closed["outcome"]],
            size=7, line=dict(width=1, color="#000000"),
        ),
        name="Cumulative R",
        hovertemplate="<b>%{x|%d %b}</b><br>Cum R: %{y:+.2f}R<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#555555", line_width=1, row=1, col=1)

    # Target line (expectancy projected)
    if n_closed >= 3:
        x_proj = list(range(n_closed))
        trend = np.polyfit(x_proj, df_closed["cum_r"], 1)
        proj_y = [trend[0] * x + trend[1] for x in x_proj]
        fig.add_trace(go.Scatter(
            x=dates, y=proj_y,
            mode="lines", line=dict(color="#ffcc00", width=1, dash="dot"),
            name="Trend", opacity=0.6,
            hoverinfo="skip",
        ), row=1, col=1)

    # ── Drawdown ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates, y=-df_closed["drawdown"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(248,81,73,0.15)",
        line=dict(color="#ff3344", width=1.5),
        name="Drawdown",
        hovertemplate="DD: %{y:.2f}R<extra></extra>",
    ), row=2, col=1)

    # ── Per-trade R bar ───────────────────────────────────────────
    bar_colors = ["#00ff66" if o == "WIN" else "#ff3344" if o == "LOSS" else "#ffcc00"
                  for o in df_closed["outcome"]]
    fig.add_trace(go.Bar(
        x=dates, y=df_closed["r_multiple"],
        marker_color=bar_colors,
        name="Trade R",
        hovertemplate="<b>%{x|%d %b}</b><br>R: %{y:+.2f}R<extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="#555555", line_width=1, row=3, col=1)

    fig.update_layout(
        height=680, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        font=dict(family="JetBrains Mono, monospace", size=11, color="#9a9a9a"),
        legend=dict(bgcolor="rgba(13,17,23,.6)", bordercolor="#2a2a2a", borderwidth=1,
                    font=dict(size=10), orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(showgrid=False, linecolor="#2a2a2a", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a2a", row=row, col=1)

    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # ── Streak analysis ───────────────────────────────────────────
    st.markdown('<div class="section-title">🔥 Streak Analysis</div>', unsafe_allow_html=True)
    outcomes_list = df_closed["outcome"].tolist()
    max_win_streak = max_los_streak = cur_w = cur_l = 0
    for o in outcomes_list:
        if o == "WIN":
            cur_w += 1;
            cur_l = 0
        elif o == "LOSS":
            cur_l += 1;
            cur_w = 0
        else:
            cur_w = cur_l = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_los_streak = max(max_los_streak, cur_l)

    last5 = outcomes_list[-5:] if len(outcomes_list) >= 5 else outcomes_list
    last5_html = "".join(
        f'<span style="display:inline-block;background:{"#0d2f1a" if o == "WIN" else "#2f0d0d" if o == "LOSS" else "#1a1a0d"};'
        f'border:1px solid {"#00a32a" if o == "WIN" else "#8b2d2d" if o == "LOSS" else "#9e6a03"};'
        f'color:{"#00ff66" if o == "WIN" else "#ff3344" if o == "LOSS" else "#ffcc00"};'
        f'font-weight:700;font-size:12px;padding:4px 12px;border-radius:5px;margin-right:6px;">{o}</span>'
        for o in last5
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:#00ff66;">{max_win_streak}</div>'
            f'<div class="metric-label">Max Win Streak</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-value" style="color:#ff3344;">{max_los_streak}</div>'
            f'<div class="metric-label">Max Loss Streak</div></div>', unsafe_allow_html=True)
    with sc3:
        st.markdown(
            f'<div class="metric-box" style="text-align:left;padding:10px 14px;">'
            f'<div class="metric-label" style="margin-bottom:6px;">Last 5 Trades</div>'
            f'{last5_html}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — BY SESSION
# ──────────────────────────────────────────────────────────────────
with tab_session:
    st.markdown('<div class="section-title">🕐 Win Rate by Session</div>', unsafe_allow_html=True)

    sess_group = (
        df_closed.groupby("session")
        .apply(lambda g: pd.Series({
            "Trades": len(g),
            "Wins": (g["outcome"] == "WIN").sum(),
            "Losses": (g["outcome"] == "LOSS").sum(),
            "BE": (g["outcome"] == "BE").sum(),
            "Win Rate %": (g["outcome"] == "WIN").mean() * 100,
            "Avg R": g["r_multiple"].mean(),
            "Total R": g["r_multiple"].sum(),
        }))
        .reset_index()
        .sort_values("Win Rate %", ascending=False)
    )

    # Bar chart
    fig_s = go.Figure()
    colors_s = ["#00ff66" if w >= 66 else "#ffcc00" if w >= 50 else "#ff3344"
                for w in sess_group["Win Rate %"]]
    fig_s.add_trace(go.Bar(
        x=sess_group["session"], y=sess_group["Win Rate %"],
        marker_color=colors_s,
        text=[f"{w:.1f}%" for w in sess_group["Win Rate %"]],
        textposition="outside", textfont=dict(color="#e6e6e6", size=12),
        hovertemplate="<b>%{x}</b><br>Win Rate: %{y:.1f}%<extra></extra>",
        name="Win Rate",
    ))
    fig_s.add_hline(y=66, line_dash="dash", line_color="#00ff66", line_width=1.5,
                    annotation_text="66% target", annotation_font_color="#00ff66",
                    annotation_position="right")
    fig_s.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a", height=320,
        yaxis=dict(range=[0, 110], gridcolor="#2a2a2a", ticksuffix="%"),
        xaxis=dict(showgrid=False), margin=dict(l=10, r=80, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_s, use_container_width=True, config=dict(displayModeBar=False))

    # Table
    st.dataframe(
        sess_group.style.format({
            "Win Rate %": "{:.1f}%", "Avg R": "{:+.2f}R", "Total R": "{:+.1f}R",
        }),
        use_container_width=True, hide_index=True,
    )

    # Kill zone effectiveness
    st.markdown('<div class="section-title">⏰ Trade Hour Distribution (UTC)</div>', unsafe_allow_html=True)
    df_closed["hour"] = df_closed["logged_at"].dt.hour
    hour_group = (
        df_closed.groupby("hour")
        .apply(lambda g: pd.Series({
            "Trades": len(g),
            "Win Rate %": (g["outcome"] == "WIN").mean() * 100,
            "Avg R": g["r_multiple"].mean(),
        }))
        .reset_index()
    )

    if not hour_group.empty:
        fig_h = go.Figure()
        kz_bands = [(7, 9, "#00ff66", "London KZ"), (12, 14, "#00ff66", "NY KZ"),
                    (15, 17, "#ffcc00", "London Close"), (0, 3, "#ff3344", "Tokyo")]
        for h0, h1, color, label in kz_bands:
            fig_h.add_vrect(x0=h0 - 0.4, x1=h1 - 0.6,
                            fillcolor=color, opacity=0.06, line_width=0,
                            annotation_text=label,
                            annotation_font=dict(size=9, color=color),
                            annotation_position="top left")
        bar_colors_h = ["#00ff66" if w >= 66 else "#ffcc00" if w >= 50 else "#ff3344"
                        for w in hour_group["Win Rate %"]]
        fig_h.add_trace(go.Bar(
            x=hour_group["hour"], y=hour_group["Win Rate %"],
            marker_color=bar_colors_h,
            text=[f"{w:.0f}%" for w in hour_group["Win Rate %"]],
            textposition="outside",
            hovertemplate="Hour %{x}:00 UTC<br>Win Rate: %{y:.1f}%<extra></extra>",
        ))
        fig_h.add_hline(y=66, line_dash="dash", line_color="#00ff66", line_width=1)
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a", height=300,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1, showgrid=False, title="Hour (UTC)"),
            yaxis=dict(range=[0, 115], gridcolor="#2a2a2a", ticksuffix="%"),
            margin=dict(l=10, r=80, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_h, use_container_width=True, config=dict(displayModeBar=False))

# ──────────────────────────────────────────────────────────────────
# TAB 3 — BY PAIR
# ──────────────────────────────────────────────────────────────────
with tab_pair:
    st.markdown('<div class="section-title">💱 Win Rate by Instrument</div>', unsafe_allow_html=True)

    pair_group = (
        df_closed.groupby("instrument")
        .apply(lambda g: pd.Series({
            "Trades": len(g),
            "Wins": (g["outcome"] == "WIN").sum(),
            "Losses": (g["outcome"] == "LOSS").sum(),
            "Win Rate %": (g["outcome"] == "WIN").mean() * 100,
            "Avg R": g["r_multiple"].mean(),
            "Total R": g["r_multiple"].sum(),
            "Best R": g["r_multiple"].max(),
            "Worst R": g["r_multiple"].min(),
        }))
        .reset_index()
        .sort_values("Win Rate %", ascending=True)
    )

    fig_p = go.Figure()
    bar_colors_p = ["#00ff66" if w >= 66 else "#ffcc00" if w >= 50 else "#ff3344"
                    for w in pair_group["Win Rate %"]]
    fig_p.add_trace(go.Bar(
        y=pair_group["instrument"], x=pair_group["Win Rate %"],
        orientation="h",
        marker_color=bar_colors_p,
        text=[f"{w:.1f}%" for w in pair_group["Win Rate %"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<br>Trades: %{customdata}<extra></extra>",
        customdata=pair_group["Trades"],
    ))
    fig_p.add_vline(x=66, line_dash="dash", line_color="#00ff66", line_width=1.5,
                    annotation_text="66%", annotation_font_color="#00ff66",
                    annotation_position="top")
    fig_p.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a",
        height=max(320, len(pair_group) * 34),
        xaxis=dict(range=[0, 115], gridcolor="#2a2a2a", ticksuffix="%"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=80, t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig_p, use_container_width=True, config=dict(displayModeBar=False))

    st.dataframe(
        pair_group.sort_values("Win Rate %", ascending=False)
        .style.format({
            "Win Rate %": "{:.1f}%", "Avg R": "{:+.2f}R",
            "Total R": "{:+.1f}R", "Best R": "{:+.2f}R", "Worst R": "{:+.2f}R",
        }),
        use_container_width=True, hide_index=True,
    )

# ──────────────────────────────────────────────────────────────────
# TAB 4 — BY DIRECTION
# ──────────────────────────────────────────────────────────────────
with tab_dir:
    st.markdown('<div class="section-title">↕️ LONG vs SHORT Performance</div>', unsafe_allow_html=True)

    dir_group = (
        df_closed.groupby("direction")
        .apply(lambda g: pd.Series({
            "Trades": len(g),
            "Wins": (g["outcome"] == "WIN").sum(),
            "Losses": (g["outcome"] == "LOSS").sum(),
            "BE": (g["outcome"] == "BE").sum(),
            "Win Rate %": (g["outcome"] == "WIN").mean() * 100,
            "Avg R": g["r_multiple"].mean(),
            "Total R": g["r_multiple"].sum(),
            "Avg Pips": g["pips_gained"].mean() if "pips_gained" in g.columns else 0,
        }))
        .reset_index()
    )

    d1, d2 = st.columns(2)
    for col_, (_, row_) in zip([d1, d2], dir_group.iterrows()):
        direction = row_["direction"]
        wr = row_["Win Rate %"]  # Fixed: use bracket notation instead of ._4
        trades = row_["Trades"]
        wins = row_["Wins"]
        losses = row_["Losses"]
        be = row_["BE"]
        avg_r = row_["Avg R"]  # Fixed: use bracket notation instead of ._6
        total_r = row_["Total R"]  # Fixed: use bracket notation instead of ._7

        d_color = "#00ff41" if direction == "LONG" else "#a371f7"
        d_icon = "🔼" if direction == "LONG" else "🔽"
        wr_c = "#00ff66" if wr >= 66 else "#ffcc00" if wr >= 50 else "#ff3344"
        with col_:
            st.markdown(f"""
            <div class="card" style="border-color:{d_color}33;">
              <div style="font-size:18px;font-weight:700;color:{d_color};margin-bottom:12px;">
                {d_icon} {direction}
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
                <div class="metric-box">
                  <div class="metric-value" style="color:{wr_c};">{wr:.1f}%</div>
                  <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-box">
                  <div class="metric-value">{trades}</div>
                  <div class="metric-label">Trades</div>
                </div>
                <div class="metric-box">
                  <div class="metric-value" style="color:{"#00ff66" if avg_r > 0 else "#ff3344"};">{avg_r:+.2f}R</div>
                  <div class="metric-label">Avg R</div>
                </div>
                <div class="metric-box">
                  <div class="metric-value" style="color:{"#00ff66" if total_r > 0 else "#ff3344"};">{total_r:+.1f}R</div>
                  <div class="metric-label">Total R</div>
                </div>
              </div>
              <div style="margin-top:10px;font-size:12px;color:#9a9a9a;">
                {wins}W / {losses}L / {be}BE
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Direction win rate over time (rolling 10)
    st.markdown('<div class="section-title">📉 Rolling Win Rate by Direction (10-trade window)</div>',
                unsafe_allow_html=True)
    for direction, color in [("LONG", "#00ff41"), ("SHORT", "#a371f7")]:
        df_d = df_closed[df_closed["direction"] == direction].copy()
        if len(df_d) >= 5:
            df_d["rolling_wr"] = (df_d["outcome"] == "WIN").rolling(min(10, len(df_d))).mean() * 100
            fig_dir = go.Figure()
            fig_dir.add_trace(go.Scatter(
                x=df_d["logged_at"], y=df_d["rolling_wr"],
                mode="lines", line=dict(color=color, width=2),
                name=f"{direction} rolling WR",
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
            fig_dir.add_hline(y=66, line_dash="dash", line_color="#00ff66", line_width=1)
            fig_dir.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0a0a", height=200,
                yaxis=dict(range=[0, 105], gridcolor="#2a2a2a", ticksuffix="%"),
                xaxis=dict(showgrid=False),
                margin=dict(l=10, r=40, t=10, b=10), showlegend=False, title_text=direction,
                title_font=dict(color=color, size=12),
            )
            st.plotly_chart(fig_dir, use_container_width=True, config=dict(displayModeBar=False))

# ──────────────────────────────────────────────────────────────────
# TAB 5 — TRADE LOG
# ──────────────────────────────────────────────────────────────────
with tab_log:
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
    st.markdown('<div class="section-title">📋 Full Trade Log</div>', unsafe_allow_html=True)

    # Filter controls
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        f_outcome = st.multiselect("Outcome", ["WIN", "LOSS", "BE", "OPEN"],
                                   default=["WIN", "LOSS", "BE", "OPEN"])
    with fc2:
        f_dir = st.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
    with fc3:
        all_instruments = sorted(df_all["instrument"].dropna().unique().tolist())
        f_pairs = st.multiselect("Instrument", all_instruments, default=all_instruments)
    with fc4:
        all_sessions = sorted(df_all["session"].dropna().unique().tolist())
        f_sessions = st.multiselect("Session", all_sessions, default=all_sessions)
    with fc5:
        all_sources = sorted(df_all["source"].dropna().unique().tolist())
        f_sources = st.multiselect("Source", all_sources, default=all_sources,
                                   help="checklist = logged in-app · mt4_import = imported broker fills")

    # Apply filters
    df_view = df_all.copy()
    if f_outcome:
        mask_out = pd.Series(False, index=df_view.index)
        if "OPEN" in f_outcome:
            mask_out |= (df_view["is_open"] != False)
        for o in [x for x in f_outcome if x != "OPEN"]:
            mask_out |= (df_view["outcome"] == o)
        df_view = df_view[mask_out]
    if f_dir:
        df_view = df_view[df_view["direction"].isin(f_dir)]
    if f_pairs:
        df_view = df_view[df_view["instrument"].isin(f_pairs)]
    if f_sessions:
        df_view = df_view[df_view["session"].isin(f_sessions)]
    if f_sources:
        df_view = df_view[df_view["source"].isin(f_sources)]

    df_view = df_view.sort_values("logged_at", ascending=False).copy()
    df_view["logged_at"] = df_view["logged_at"].dt.strftime("%Y-%m-%d %H:%M")

    st.caption(f"Showing {len(df_view)} of {n_total} trades")

    st.dataframe(
        df_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "logged_at": st.column_config.TextColumn("Date/Time", width="medium"),
            "instrument": st.column_config.TextColumn("Pair"),
            "direction": st.column_config.TextColumn("Dir", width="small"),
            "session": st.column_config.TextColumn("Session"),
            "outcome": st.column_config.TextColumn("Outcome", width="small"),
            "r_multiple": st.column_config.NumberColumn("R", format="%+.2f"),
            "pips_gained": st.column_config.NumberColumn("Pips", format="%+.1f"),
            "entry_price": st.column_config.NumberColumn("Entry", format="%.5f"),
            "close_price": st.column_config.NumberColumn("Close", format="%.5f"),
            "sl_pips": st.column_config.NumberColumn("SL pips", format="%.1f"),
            "tp1_pips": st.column_config.NumberColumn("TP1 pips", format="%.1f"),
            "rr_tp1": st.column_config.NumberColumn("R:R", format="%.2f"),
            "lot_size": st.column_config.NumberColumn("Lots", format="%.2f"),
            "risk_amount": st.column_config.NumberColumn("Risk $", format="%.2f"),
            "checks_passed": st.column_config.NumberColumn("✅", width="small"),
            "is_open": st.column_config.CheckboxColumn("Open", width="small"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "notes": st.column_config.TextColumn("Notes", width="large"),
        },
    )

    # Export
    csv = df_view.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Export filtered trades as CSV",
        data=csv,
        file_name=f"trade_journal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# Footer
st.markdown("""
<div style="text-align:center;color:#555555;font-size:11px;margin-top:32px;padding-top:16px;border-top:1px solid #2a2a2a;">
  📓 Trade Journal · Equity Curve · Performance Analytics · 66% Win Rate System
</div>
""", unsafe_allow_html=True)