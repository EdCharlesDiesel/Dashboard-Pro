"""Streamlit-facing wrapper over the canonical bias engine.

Fetches the three canonical timeframes through the data spine
(src/services/market_data.py), computes the pure HouseView
(src/core/bias.py), and caches the result so all pages rendered in the same
5-minute window show the *identical* house view for a pair.

Page integration is one line::

    from src.services.bias_service import show_house_view
    show_house_view("EUR/USD", page_read="Bullish", page_label="Weekly EMA")

``page_read`` is the page's own headline direction in its own vocabulary
(Long/Short, BUY/SELL, Bullish/Bearish…); when it opposes the house view the
strip flags the conflict instead of letting two pages silently contradict
each other.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from src.core.bias import HouseView, house_view
from src.instruments.registry import INSTRUMENTS
from src.services.market_data import daily_ohlc, h4_ohlc, weekly_ohlc


# How stale the worker's precomputed board may be before this funnel ignores
# it and computes live. The daemon cycles every 5 min; 15 min (three missed
# cycles) means it's down or the market's closed — live compute is then honest.
_BOARD_MAX_AGE_S = 900


@st.cache_data(ttl=300, show_spinner=False)
def _house_view_cached(pair: str, ticker: str) -> HouseView:
    return house_view(
        pair,
        weekly=weekly_ohlc(ticker),
        daily=daily_ohlc(ticker),
        h4=h4_ohlc(ticker),
    )


@st.cache_data(ttl=60, show_spinner=False)
def _board_cached():
    """The worker's precomputed board, read once per minute for the whole
    process (one DB round-trip serves every pair on a universe-wide page).
    ``None`` when the worker hasn't run / the DB is down — caller goes live."""
    from src.services.precomputed import read_board
    return read_board()


def get_house_view(pair: str) -> Optional[HouseView]:
    """The canonical composite read for a registry pair, or ``None`` for a
    symbol the registry doesn't know (free-text tickers stay lens-only).

    Prefers the background daemon's precomputed board — a single cached DB read
    for the entire universe instead of three timeframe fetches + a recompute
    per pair on the request path. Falls through to a live compute whenever the
    board is missing, stale, or doesn't have this pair, so behavior is
    identical when the worker isn't running.
    """
    inst = INSTRUMENTS.get(pair)
    if inst is None:
        return None
    try:
        from src.services.precomputed import board_is_fresh, house_view_from_board
        board = _board_cached()
        if board_is_fresh(board, _BOARD_MAX_AGE_S):
            view = house_view_from_board(pair, board)
            if view is not None:
                return view
    except Exception:
        pass  # any board-read problem → fall through to live compute
    try:
        return _house_view_cached(pair, inst.ticker)
    except Exception:
        return None


def show_house_view(
    pair: str,
    page_read: Optional[str] = None,
    page_label: str = "This page",
) -> Optional[HouseView]:
    """Render the shared house-view strip for ``pair`` and return the view.

    Safe on every page: unknown pair or fetch failure renders nothing and
    returns ``None``.
    """
    view = get_house_view(pair)
    if view is None:
        return None
    from src.ui.components import HouseViewStrip
    HouseViewStrip(view=view, page_read=page_read, page_label=page_label).show()
    return view
