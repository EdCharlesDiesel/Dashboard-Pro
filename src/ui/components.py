"""Object-oriented Bloomberg-terminal UI primitives.

Each class renders a single self-contained widget via st.markdown() with
unsafe_allow_html=True. CSS lives in BloombergTheme — never inline a colour
here that's not sourced from BloombergTheme.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from src.ui.theme import BloombergTheme as T


class _HTMLWidget:
    """Mixin: render() returns a string; show() pushes to Streamlit."""

    def render(self) -> str:  # pragma: no cover - subclasses override
        raise NotImplementedError

    def show(self) -> None:
        st.markdown(self.render(), unsafe_allow_html=True)


# ── Panel ──────────────────────────────────────────────────────────────────

@dataclass
class Panel(_HTMLWidget):
    """Bordered window with a labelled header strip.

    Use as a context manager for arbitrary Streamlit content::

        with Panel("Trade Verdict", tag="LIVE").context():
            st.markdown("...")
    """
    title: str
    tag: str = ""
    body_html: str = ""

    def render(self) -> str:
        tag = f'<span class="bb-tag">{self.tag}</span>' if self.tag else ""
        return (
            f'<div class="bb-panel">'
            f'<div class="bb-panel-header"><span>{self.title}</span>{tag}</div>'
            f'<div class="bb-panel-body">{self.body_html}</div>'
            f'</div>'
        )

    @contextmanager
    def context(self):
        tag = f'<span class="bb-tag">{self.tag}</span>' if self.tag else ""
        st.markdown(
            f'<div class="bb-panel"><div class="bb-panel-header">'
            f'<span>{self.title}</span>{tag}</div><div class="bb-panel-body">',
            unsafe_allow_html=True,
        )
        try:
            yield
        finally:
            st.markdown("</div></div>", unsafe_allow_html=True)


# ── MetricCell ─────────────────────────────────────────────────────────────

@dataclass
class MetricCell(_HTMLWidget):
    label: str
    value: str
    color: str = "amber"   # amber | green | red | cyan | yellow | white
    delta: str = ""

    def render(self) -> str:
        delta = f'<div class="bb-metric-delta">{self.delta}</div>' if self.delta else ""
        return (
            f'<div class="bb-metric">'
            f'<div class="bb-metric-label">{self.label}</div>'
            f'<div class="bb-metric-value bb-{self.color}">{self.value}</div>'
            f'{delta}</div>'
        )


# ── Chip ───────────────────────────────────────────────────────────────────

@dataclass
class Chip(_HTMLWidget):
    text: str
    kind: str = "info"   # go | wait | no | info | amber

    def render(self) -> str:
        return f'<span class="bb-chip bb-chip-{self.kind}">{self.text}</span>'


# ── ProgressBar ────────────────────────────────────────────────────────────

@dataclass
class ProgressBar(_HTMLWidget):
    pct: float
    color: str = T.AMBER

    def render(self) -> str:
        pct = max(0, min(100, int(self.pct)))
        return (
            f'<div class="bb-progress">'
            f'<div class="bb-progress-fill" style="width:{pct}%;background:{self.color};"></div>'
            f'</div>'
        )


# ── Data Row ───────────────────────────────────────────────────────────────

@dataclass
class DataRow(_HTMLWidget):
    key: str
    value: str
    color: str = "white"   # white | green | red | amber | cyan

    def render(self) -> str:
        return (
            f'<div class="bb-data-row">'
            f'<span class="bb-data-key">{self.key}</span>'
            f'<span class="bb-data-val bb-{self.color}">{self.value}</span>'
            f'</div>'
        )


# ── Ticker Strip ───────────────────────────────────────────────────────────

@dataclass
class TickerStrip(_HTMLWidget):
    """Top-of-page ticker bar — symbols with optional delta colouring."""
    items: Sequence[Tuple[str, str, Optional[float]]]
    # (symbol, value_text, delta_pct or None)

    def render(self) -> str:
        pills = []
        for sym, val, delta in self.items:
            if delta is None:
                arrow = ""
                cls = ""
            elif delta >= 0:
                arrow, cls = "▲", "bb-up"
            else:
                arrow, cls = "▼", "bb-down"
            delta_html = (
                f' <span class="{cls}">{arrow} {abs(delta):.2f}%</span>'
                if delta is not None else ""
            )
            pills.append(
                f'<span class="bb-ticker-pill">'
                f'<span class="bb-sym">{sym}</span> {val}{delta_html}</span>'
            )
        return f'<div class="bb-ticker-strip">{"".join(pills)}</div>'


# ── Command Bar ────────────────────────────────────────────────────────────

@dataclass
class CommandBar(_HTMLWidget):
    """Bloomberg-style command strip: <LABEL> | cell | cell | cell."""
    label: str
    cells: Sequence[Tuple[str, str]]   # (text, color_class) — color: '', 'cyan', 'green', 'red'

    def render(self) -> str:
        cell_html = "".join(
            f'<div class="bb-cmd-cell {("bb-" + c) if c else ""}">{txt}</div>'
            for txt, c in self.cells
        )
        return (
            f'<div class="bb-cmd-bar">'
            f'<div class="bb-cmd-label">{self.label}</div>'
            f'{cell_html}</div>'
        )


# ── Function Code Grid ─────────────────────────────────────────────────────

@dataclass
class FunctionCodeGrid(_HTMLWidget):
    """Grid of Bloomberg-style function shortcuts.

    Each entry: (code, label, url). Render as clickable tiles.
    """
    entries: Sequence[Tuple[str, str, str]]

    def render(self) -> str:
        tiles = []
        for code, label, url in self.entries:
            tiles.append(
                f'<div class="bb-fn-tile">'
                f'<a href="{url}" target="_self">'
                f'<div class="bb-fn-code">{code}</div>'
                f'<div class="bb-fn-label">{label}</div>'
                f'</a></div>'
            )
        return f'<div class="bb-fn-grid">{"".join(tiles)}</div>'


# ── Status Bar (sticky bottom) ─────────────────────────────────────────────

@dataclass
class StatusBar(_HTMLWidget):
    cells: Sequence[str]

    def render(self) -> str:
        sep = '<span class="bb-sep">│</span>'
        body = sep.join(f"<span>{c}</span>" for c in self.cells)
        return f'<div class="bb-status-bar">{body}</div>'


# ── Hero header ────────────────────────────────────────────────────────────

@dataclass
class HeroHeader(_HTMLWidget):
    title: str
    subtitle: str
    symbol: str
    ticker: str
    side: str   # LONG | SHORT | ""

    def render(self) -> str:
        side_cls = "bb-long" if self.side == "LONG" else "bb-short" if self.side == "SHORT" else ""
        arrow = "▲" if self.side == "LONG" else ("▼" if self.side == "SHORT" else "")
        side_html = (
            f'<div class="bb-hero-side {side_cls}">{arrow} {self.side}</div>'
            if self.side else ""
        )
        return (
            f'<div class="bb-hero">'
            f'<div>'
            f'<div class="bb-hero-title">{self.title}</div>'
            f'<div class="bb-hero-sub">{self.subtitle}</div>'
            f'</div>'
            f'<div class="bb-hero-right">'
            f'<div class="bb-hero-symbol">{self.symbol}</div>'
            f'<div class="bb-hero-ticker">{self.ticker}</div>'
            f'{side_html}'
            f'</div>'
            f'</div>'
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def render_metric_row(cells: Iterable[MetricCell]) -> None:
    """Renders a flex row of metric cells in a responsive grid."""
    html = '<div class="bb-mosaic bb-mosaic-4">' + "".join(c.render() for c in cells) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_data_rows(rows: Iterable[DataRow]) -> str:
    """Returns the HTML for a stack of DataRows (useful inside a Panel)."""
    return "".join(r.render() for r in rows)
