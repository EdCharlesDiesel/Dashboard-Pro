"""Daily Trading page composition — wires the checklist sidebar and controller."""
from __future__ import annotations

from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.daily_trading.checklist import ChecklistController
from src.pages_lib.daily_trading.sidebar import ChecklistSidebar
from src.pages_lib.daily_trading.state import SessionStateBootstrap


class DailyTradingPage(BloombergPage):
    """The top-level Bloomberg-style daily-trading page — the 18-point checklist.

    The multi-symbol trend scanner that used to share this page via a MODE
    toggle now lives on its own page (``pages/trend-signals.py`` →
    ``TrendSignalsPage``).
    """

    def configure(self) -> PageContext:
        SessionStateBootstrap.init()
        return PageContext(
            code="CHCK",
            title="Daily Trading System",
            icon="📈",
        )

    # ── sidebar ────────────────────────────────────────────────────────────
    def sidebar(self, ctx: PageContext) -> None:
        ChecklistSidebar().render()

    # ── body ───────────────────────────────────────────────────────────────
    def body(self, ctx: PageContext) -> None:
        ChecklistController().render()
