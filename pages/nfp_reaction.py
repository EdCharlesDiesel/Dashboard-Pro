"""
nfp_reaction.py — Event Reaction Map (standalone dashboard page,
auto-registered from pages/ — see src/pages_lib/navigation.py's EVNT entry).
==========================================================================

The UI over ``src/core/nfp_reaction.py``. That module owns every number on
this page — the surprise z-scores, the regime weights, the exposure betas, the
transmission chain and the session phases, all per event — so the maths is
unit-tested without a Streamlit runtime and the page stays presentation.

Four scheduled US releases: **NFP, CPI, PPI and FOMC**. Pick the event, type
what printed against what was expected, and the page scores the surprise,
pushes it through that event's own transmission chain, and ranks which
instruments are actually exposed — with a conviction score that collapses when
the rate channel and the growth channel disagree.

The betas are PRIORS — starting values, not findings. They are made falsifiable
rather than calibrated in place: the read is journalled to ``tool_usage_log``
and the high-conviction directional calls go to ``trade_setups`` under **one
source tag per event** (``nfp_reaction`` / ``cpi_reaction`` / ``ppi_reaction``
/ ``fomc_reaction``), so the Trade Journal's Source Scorecard grades each
event's betas separately. CPI betas being any good says nothing about NFP's.

The filename is a holdover from when this page only did payrolls. Renaming it
would churn the URL, the sweep registry and the nav for no functional gain.

Entry point: render()
Standalone:  streamlit run pages/nfp_reaction.py (also auto-registers as a
multipage page — see src/pages_lib/navigation.py)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.core.event_calendar import next_release
from src.core.nfp_reaction import (
    EVENTS,
    NY,
    REGIMES,
    board_to_signals,
    chain_leaves,
    compute_surprise,
    release_datetime_sast,
    score_instruments,
    timing_frame,
)
from src.pages_lib.navigation import render_sidebar_nav
from src.services.alert_service import NotifyCache
from src.services.signal_store import persist_signals
from src.services.tool_log import log_tool_usage
from src.ui.charts import ChartKit
from src.ui.theme import BloombergTheme as T


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
# Both are hand-rolled `go.Figure`: the chain is a node-and-arrow diagram and
# the board is a horizontal bar chart, neither of which ChartKit's price-panel
# primitives cover (`ChartKit.bars` is vertical only). They are page-specific
# composites in the same sense as the volume profile and the radar — but they
# take their colours from BloombergTheme and render with the shared config, so
# they still read as part of one terminal.

_BY_LABEL = {spec.label: spec for spec in EVENTS.values()}


def _arrow(up: bool) -> str:
    return "▲" if up else "▼"


def _house_layout(fig: go.Figure, height: int, title: str) -> go.Figure:
    """The tokens `ChartKit.finish` applies, for figures it can't compose."""
    fig.update_layout(
        height=height,
        paper_bgcolor=T.BG, plot_bgcolor=T.BG_PANEL,
        font=dict(color=T.WHITE, family=T.FONT_MONO, size=10),
        showlegend=False,
        margin=dict(l=10, r=10, t=34, b=10),
        title=dict(text=title, font=dict(size=11, color=T.AMBER,
                                         family=T.FONT_MONO)),
    )
    return fig


def chain_figure(spec, surprise, board) -> go.Figure:
    """The transmission chain, with each node stamped by the actual surprise."""
    faded = abs(surprise.composite) < 0.35
    chain, leaves = chain_leaves(spec, surprise, board)

    fig = go.Figure()
    _house_layout(fig, 330, "TRANSMISSION CHAIN — {0}".format(spec.label.upper()))
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False, range=[0, 100]),
    )

    # Trunk. Node widths follow the label, since "consumer prices" and "jobs"
    # are not the same size and a fixed box clips one of them.
    xs = [4, 28, 52, 76]
    for x, (name, up) in zip(xs, chain):
        col = T.GREY if faded else (T.GREEN if up else T.RED)
        fig.add_shape(type="rect", x0=x, x1=x + 20, y0=72, y1=88,
                      line=dict(color=col, width=1.4), fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=x + 10, y=80, text=f"<b>{name}</b>  {_arrow(up)}",
                           showarrow=False,
                           font=dict(color=col, size=12, family=T.FONT_MONO))
        if x != xs[-1]:
            fig.add_annotation(x=x + 25, y=80, ax=x + 21, ay=80, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=2,
                               arrowsize=1, arrowwidth=1.2, arrowcolor=T.BORDER, text="")

    # Last trunk node -> the three leaves.
    fig.add_shape(type="line", x0=86, x1=86, y0=72, y1=52, line=dict(color=T.BORDER, width=1))
    fig.add_shape(type="line", x0=14, x1=86, y0=52, y1=52, line=dict(color=T.BORDER, width=1))

    lx = [4, 38, 72]
    for x, (name, up) in zip(lx, leaves):
        col = T.GREY if faded else (T.GREEN if up else T.RED)
        fig.add_shape(type="line", x0=x + 10, x1=x + 10, y0=52, y1=38,
                      line=dict(color=T.BORDER, width=1))
        fig.add_shape(type="rect", x0=x, x1=x + 20, y0=20, y1=38,
                      line=dict(color=col, width=1.4), fillcolor="rgba(0,0,0,0)")
        fig.add_annotation(x=x + 10, y=29, text=f"<b>{name}</b>  {_arrow(up)}",
                           showarrow=False,
                           font=dict(color=col, size=12, family=T.FONT_MONO))

    fig.add_annotation(
        x=50, y=6,
        text=f"composite z = {surprise.composite:+.2f} "
             f"({surprise.label}, {surprise.direction})",
        showarrow=False, font=dict(color=T.AMBER, size=12, family=T.FONT_MONO),
    )
    return fig


def board_figure(board) -> go.Figure:
    d = board.head(10).iloc[::-1]
    colors = [T.GREEN if s > 0 else T.RED for s in d["score"]]
    opacity = [0.35 + 0.65 * c for c in d["conviction"]]

    fig = go.Figure(
        go.Bar(
            x=d["score"],
            y=d["symbol"],
            orientation="h",
            marker=dict(color=colors, opacity=opacity),
            customdata=np.stack([d["expected_move"], d["conviction"], d["unit"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>score %{x:+.2f}"
                          "<br>typical 30m move %{customdata[0]:.2f} %{customdata[2]}"
                          "<br>conviction %{customdata[1]:.0%}<extra></extra>",
        )
    )
    _house_layout(fig, 380, "DIRECTIONAL EXPOSURE — BAR OPACITY = CONVICTION")
    fig.update_layout(
        hoverlabel=dict(bgcolor=T.BG, bordercolor=T.AMBER,
                        font=dict(color=T.AMBER, family=T.FONT_MONO, size=11)),
        xaxis=dict(title="signed score", zerolinecolor=T.GREY, gridcolor=T.BORDER,
                   tickfont=dict(color=T.GREY, size=9)),
        yaxis=dict(gridcolor=T.BORDER, tickfont=dict(color=T.GREY, size=9)),
    )
    return fig


# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------

def _collect_inputs(spec) -> Dict[str, float]:
    """Render this event's declared components and return what was entered.

    Widget keys are namespaced per event so switching events cannot carry a
    stale value across — CPI's `core_mm` and PPI's `core_mm` are different
    numbers on very different scales.
    """
    values: Dict[str, float] = {}
    cols = st.columns(4)
    slot = 0

    for c in spec.components:
        with cols[slot % 4]:
            if c.key == "tone":
                # A dial, not a measurement — a slider says that better than a
                # number box does.
                values[c.key] = st.slider(
                    c.label, -2.0, 2.0, float(c.default), float(c.step),
                    key=f"{spec.key}_{c.key}", help=c.help)
            elif c.delta_only:
                values[c.key] = st.number_input(
                    c.label, value=float(c.default), step=float(c.step),
                    format=c.fmt, key=f"{spec.key}_{c.key}", help=c.help)
            else:
                values[c.key] = st.number_input(
                    f"{c.label} — actual", value=float(c.default),
                    step=float(c.step), format=c.fmt,
                    key=f"{spec.key}_{c.key}", help=c.help)
                values[c.key + "_c"] = st.number_input(
                    f"{c.label} — consensus", value=float(c.default),
                    step=float(c.step), format=c.fmt,
                    key=f"{spec.key}_{c.key}_c")
        slot += 1

    return values


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def _persist_for(spec, signals: List[Dict[str, Any]]) -> int:
    """Four literal calls, not one computed tag.

    ``tests/test_signal_sweep.py`` scans the source with a regex that only
    matches a source tag written as a **string literal** in the call. A
    computed tag would be invisible to it, and this page would silently drop
    out of the sweep registry — the exact class of drift that test exists to
    catch. Four branches is the price of keeping the guard, and the guard is
    worth more than the branches.

    (Writing the regex's own pattern into this docstring is itself enough to
    trip it, which is a fair demonstration that the check is doing real work.)
    """
    if spec.key == "NFP":
        return persist_signals("nfp_reaction", signals)
    if spec.key == "CPI":
        return persist_signals("cpi_reaction", signals)
    if spec.key == "PPI":
        return persist_signals("ppi_reaction", signals)
    return persist_signals("fomc_reaction", signals)


def _persist(spec, rel_date, regime: str, inputs: Dict[str, float],
             surprise, board) -> List[Dict[str, Any]]:
    """Journal the read, then persist the directional calls it is confident in.

    Two destinations, deliberately:

    - ``tool_usage_log`` records *every* read, including the in-line prints and
      the low-conviction rows the board tells you to skip — that is the audit
      trail, and a rejected read is still worth having journalled. It stays a
      single ``nfp_reaction`` stream with an ``event`` field, because it is
      queried by tool.
    - ``trade_setups`` gets only what clears the gates in
      :func:`board_to_signals`, under this event's own tag, because the Source
      Scorecard ranks a source by expectancy over its stored rows: filling it
      with coin flips the page itself flags as coin flips would corrupt its own
      score.

    The log is deduped on the shape of the read. Streamlit reruns the whole
    script on every widget touch, so without this a single keystroke in a
    number input would append a row.
    """
    key = "{0}|{1}|{2}|{3}".format(
        spec.key, rel_date.isoformat(), regime,
        "|".join("{0}:{1}".format(k, round(float(v), 3))
                 for k, v in sorted(inputs.items())))
    if NotifyCache("nfp_reaction_log").filter_new([key]):
        log_tool_usage("nfp_reaction", {
            "event": spec.key,
            "release_date": rel_date.isoformat(),
            "regime": regime,
            "composite": round(float(surprise.composite), 4),
            "label": surprise.label,
            "direction": surprise.direction,
            "z": {k: round(float(v), 4) for k, v in surprise.z.items()},
            "inputs": {k: float(v) for k, v in inputs.items()},
            "board": [
                {"symbol": r["symbol"], "score": round(float(r["score"]), 4),
                 "conviction": round(float(r["conviction"]), 4)}
                for _, r in board.iterrows()
            ],
        })

    signals = board_to_signals(spec, board, rel_date, regime, surprise.composite)
    if signals:
        _persist_for(spec, signals)
    return signals


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------

def render() -> None:
    st.title("Event Reaction Map")
    st.caption("Release surprise → regime-aware transmission chain → instrument exposure")

    with st.sidebar:
        st.markdown("### 🧾 EVENT REACTION MAP")
        st.caption("NFP · CPI · PPI · FOMC")
        # The options are the labels themselves rather than the keys behind a
        # `format_func`: one less indirection, and it keeps the widget's value
        # equal to what is displayed, which is what the sweep's PREPARE hook
        # and any AppTest have to select on.
        ev_label = st.selectbox("Event", [e.label for e in EVENTS.values()],
                                index=0, key="evt_event")
        spec = _BY_LABEL[ev_label]

        scheduled = next_release(spec.calendar_key)
        rel_date = st.date_input("Release date", value=scheduled or date.today(),
                                 key="evt_date")
        if scheduled is None:
            st.caption("⚠️ No scheduled date on file for this event — the seed "
                       "list has run out. Enter it manually.")
        st.divider()
        render_sidebar_nav()

    t0 = release_datetime_sast(spec, rel_date)
    st.caption(f"**{spec.label}** prints {t0:%H:%M} SAST · "
               f"{t0.astimezone(NY):%H:%M} New York")
    st.caption(spec.note)

    values = _collect_inputs(spec)

    regime = st.selectbox("Reaction regime", list(REGIMES.keys()), index=1,
                          key="evt_regime")
    st.caption(REGIMES[regime]["note"])

    s = compute_surprise(spec, values)
    board = score_instruments(spec, s.composite, regime)

    # One metric per scored component, composite first. The count varies by
    # event (2 for PPI, 4 for NFP/CPI), so the row is built, not hard-coded.
    by_key = {c.key: c for c in spec.components}
    cells = st.columns(1 + len(s.z))
    cells[0].metric("Composite z", f"{s.composite:+.2f}", s.label)
    for cell, (key, z) in zip(cells[1:], s.z.items()):
        comp = by_key[key]
        cell.metric(comp.label, f"{z:+.2f}",
                    "inverted" if comp.invert else None)

    if abs(s.composite) < 0.35:
        st.info("In line with what was priced. Expect a liquidity spike with no "
                "durable direction — the board below is noise at this magnitude.")

    st.plotly_chart(chain_figure(spec, s, board), use_container_width=True,
                    config=ChartKit.PLOTLY_CONFIG)
    st.plotly_chart(board_figure(board), use_container_width=True,
                    config=ChartKit.PLOTLY_CONFIG)

    disp = board.copy()
    disp["expected 30m move"] = [
        f"{r['expected_move']:.{int(r['decimals'])}f} {r['unit']}" for _, r in disp.iterrows()
    ]
    disp["bias"] = disp["direction"].map({"up": "long", "down": "short", "flat": "—"})
    disp["conviction"] = (disp["conviction"] * 100).round(0).astype(int).astype(str) + "%"
    st.dataframe(
        disp[["symbol", "bias", "expected 30m move", "conviction", "score"]]
        .style.format({"score": "{:+.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    low = board[board["conviction"] < 0.35]["symbol"].tolist()
    if low:
        st.caption(
            "Rate and growth channels are fighting in " + ", ".join(low) +
            ". These are the ones that whipsaw both ways in the first fifteen minutes."
        )

    with st.expander("Session map (SAST)", expanded=False):
        st.dataframe(timing_frame(spec, rel_date), use_container_width=True,
                     hide_index=True)

    st.markdown("---")
    saved = _persist(spec, rel_date, regime, values, s, board)

    if saved:
        pairs = ", ".join(sorted({sig["pair"] for sig in saved}))
        st.success(f"📌 {len(saved)} directional call(s) saved under "
                   f"`{spec.source_tag}`: {pairs}")
    else:
        st.caption(
            "Nothing persisted. Either the print is in line with what was priced, "
            "or every registry-tradable row is below the conviction floor — the "
            "page declining to forecast, not a failure."
        )
    st.caption(
        f"The betas on this page are priors, not findings. Whether **{spec.label}**'s "
        f"are any good is settled in the Trade Journal's 🏆 Source Scorecard, which "
        f"resolves every row stored here under source `{spec.source_tag}` against what "
        f"price actually did — separately from the other three events. DXY, US500, "
        f"NAS100, US10Y and BTCUSD have no tradable registry pair, so they are read "
        f"here but never scored."
    )
    try:
        st.page_link("pages/trade-journal.py", label="Open the Source Scorecard", icon="🏆")
    except Exception:
        # st.page_link needs the multipage registry, which the AppTest harness
        # has no equivalent of (see CLAUDE.md). Harmless under `streamlit run`.
        pass


# ===========================================================================
# PAGE ENTRY
# ===========================================================================
# Streamlit executes a page module top-to-bottom on every run (the same way the
# other legacy pages self-initialise), so call the entry unconditionally.

def _page() -> None:
    st.set_page_config(
        page_title="EVNT · Event Reaction Map",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    T.apply()
    render()


_page()
