"""Base class for every Bloomberg-styled Streamlit page.

Subclass `BloombergPage`, override `configure()`, `sidebar()`, and `body()`,
then call `MyPage().run()` at module bottom.

The base class handles:
- `st.set_page_config()`
- Theme CSS injection
- Common sidebar header (logo + nav function grid)
- Status bar at the bottom
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import streamlit as st

from src.pages_lib.navigation import render_sidebar_nav
from src.ui.components import StatusBar
from src.ui.theme import BloombergTheme


@dataclass
class PageContext:
    """Carries cross-cutting flags between hook methods."""
    code: str            # Function code displayed in the header (e.g. "CHCK")
    title: str           # Page title shown in tab + hero
    icon: str = "📊"
    show_status_bar: bool = True


class BloombergPage:
    """Template-method pattern: subclasses override the protected hooks."""

    # ── public lifecycle ────────────────────────────────────────────────────
    def run(self) -> None:
        ctx = self.configure()
        self._page_config(ctx)
        BloombergTheme.apply()
        with st.sidebar:
            self._sidebar_header(ctx)
            # Page controls (symbol pickers etc.) first, navigation below —
            # the trader's working inputs always stay within reach.
            self.sidebar(ctx)
            st.markdown("---")
            render_sidebar_nav()
        self.body(ctx)
        if ctx.show_status_bar:
            self._status_bar(ctx)

    # ── hooks (override these) ──────────────────────────────────────────────
    def configure(self) -> PageContext:
        raise NotImplementedError

    def sidebar(self, ctx: PageContext) -> None:
        """Override to render additional sidebar content."""
        pass

    def body(self, ctx: PageContext) -> None:
        raise NotImplementedError

    # ── shared rendering ────────────────────────────────────────────────────
    @staticmethod
    def _page_config(ctx: PageContext) -> None:
        st.set_page_config(
            page_title=f"{ctx.code} · {ctx.title}",
            page_icon=ctx.icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @staticmethod
    def _sidebar_header(ctx: PageContext) -> None:
        st.markdown(
            f"""
<div style="padding:6px 0 10px 0;border-bottom:1px solid #2a2a2a;margin-bottom:8px;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;
              color:#00ff41;letter-spacing:0.18em;">DASHBOARD&nbsp;PRO</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#9a9a9a;
              letter-spacing:0.12em;margin-top:2px;">TERMINAL · {ctx.code}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    @staticmethod
    def _status_bar(ctx: PageContext) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        StatusBar([
            f"FN {ctx.code}",
            f"PG {ctx.title.upper()}",
            f"LOCAL {now}",
            "DASHBOARD PRO v3.0",
        ]).show()
