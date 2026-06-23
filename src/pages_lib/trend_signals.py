"""Trend Signals — standalone Bloomberg-terminal page.

The multi-symbol trend-following scanner (50/200 EMA · RSI(14) · MACD · ADX)
that used to be a MODE toggle inside the Daily Trading System page. The body
and sidebar controllers are reused verbatim from
``src.pages_lib.daily_trading`` — only this page wrapper is new, so the scanner
logic is unchanged.
"""
from __future__ import annotations

from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.daily_trading.sidebar import TrendSidebar
from src.pages_lib.daily_trading.state import SessionStateBootstrap
from src.pages_lib.daily_trading.trend import TrendController


class TrendSignalsPage(BloombergPage):
    """Multi-symbol trend-following signal scanner."""

    def configure(self) -> PageContext:
        SessionStateBootstrap.init()
        return PageContext(
            code="TSIG",
            title="Trend Signals",
            icon="📡",
        )

    def sidebar(self, ctx: PageContext) -> None:
        TrendSidebar().render()

    def body(self, ctx: PageContext) -> None:
        TrendController().render()
