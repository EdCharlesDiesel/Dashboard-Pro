from dataclasses import dataclass, field
from typing import Dict

from src.instruments.registry import INSTRUMENTS


def _assets_from_registry() -> Dict[str, str]:
    """Symbol → Yahoo ticker map, derived from the instrument registry so the
    whole system (Checklist, Setup Ranker, Market Overview, signals…) scans the
    identical universe with identical tickers — one source of truth."""
    return {name: inst.ticker for name, inst in INSTRUMENTS.items()}


@dataclass
class AppConfig:
    """Application configuration — all tuneable parameters in one place."""
    version: str = "1.0.1"
    last_updated: str = "2026-06-23"

    # Mirrors src/instruments/registry.py — never hand-maintain a separate list.
    assets: Dict[str, str] = field(default_factory=_assets_from_registry)

    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Weekly": {"interval": "1wk", "period": "2y"},   # ~104 bars — enough for EMA50, BB, ADX etc.
        "Daily": {"interval": "1d", "period": "3mo"},
        "4 Hour": {"interval": "4h", "period": "1mo"},
        "Hourly": {"interval": "1h", "period": "1mo"},
        "15 Minute": {"interval": "15m", "period": "5d"},
    })

    risk_per_trade: float = 0.02
    atr_sl_mult: float = 1.5
    tp1_atr_mult: float = 3.0
    tp2_atr_mult: float = 5.0
    min_rr: float = 2.0
    adx_trend_min: float = 20.0
    rsi_os: float = 40.0
    rsi_ob: float = 60.0
    stoch_os: float = 25.0
    stoch_ob: float = 75.0

    # ── Canonical lens parameters ────────────────────────────────────────────
    # The cross-page sync contract: page sliders DEFAULT to these values, and a
    # deviation is labelled a "custom lens" (src/ui/components.py lens_caption).
    # src/core/bias.py derives its EMA pair from here — edit these, and the
    # house view, score_setup's trend criteria, and every page's defaults move
    # together instead of drifting page by page.
    ema_fast: int = 20      # canonical trend pair (bias engine, daily/weekly trend)
    ema_slow: int = 50
    ema_long: int = 200     # long-trend lens (TrendSignalEvaluator, 4H zone pages)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    swing_strength: int = 3  # pivot bars each side (market-structure, swing pages)

    # Keyed to every instrument in the registry (registry order). Existing
    # values are unchanged; the additional pairs/metals follow the same scale.
    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 1.5, "GBP/USD": 1.8, "AUD/USD": 1.5, "NZD/USD": 1.6,
        "USD/JPY": 1.5, "USD/CHF": 1.5, "USD/CAD": 1.5,
        "EUR/GBP": 1.5, "EUR/JPY": 1.5, "GBP/JPY": 1.8, "AUD/JPY": 1.6,
        "EUR/AUD": 1.6, "GBP/AUD": 1.8, "EUR/CAD": 1.6, "GBP/CAD": 1.8,
        "USD/ZAR": 2.5, "EUR/ZAR": 2.5, "GBP/ZAR": 2.5,
        "XAU/USD": 2.0, "XAG/USD": 2.0, "XPT/USD": 2.0, "WTI/USD": 2.0,
    })

    # Minimum stop distance in price terms — scaled to each instrument's pip
    # size (0.0001 majors, 0.01 JPY crosses, 0.10 gold/platinum, 0.01 silver).
    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0010, "GBP/USD": 0.0015, "AUD/USD": 0.0010, "NZD/USD": 0.0010,
        "USD/JPY": 0.10, "USD/CHF": 0.0010, "USD/CAD": 0.0010,
        "EUR/GBP": 0.0010, "EUR/JPY": 0.10, "GBP/JPY": 0.15, "AUD/JPY": 0.10,
        "EUR/AUD": 0.0015, "GBP/AUD": 0.0015, "EUR/CAD": 0.0015, "GBP/CAD": 0.0015,
        "USD/ZAR": 0.05, "EUR/ZAR": 0.05, "GBP/ZAR": 0.05,
        "XAU/USD": 2.00, "XAG/USD": 0.05, "XPT/USD": 2.00, "WTI/USD": 0.20,
    })

    dxy_symbol: str = "DX-Y.NYB"
    quantconnect_url: str = "http://localhost:5000/data"
    cache_ttl: int = 300
    auto_refresh_interval: int = 300


default_config = AppConfig()

# ─────────────────────────────────────────────────────────────────────────────
# Shared chart style constants
# Import these in every file that builds a Plotly chart so all charts look
# identical across the application.
# ─────────────────────────────────────────────────────────────────────────────

# Candlestick colours — the single source of truth for every candle chart in the
# app. Bright terminal green/red wicks & borders over dark bodies so candles read
# clearly against the near-black plot background. (Also aliased as VISIBLE_CANDLE
# in src/pages_lib/market_overview_lib.py.)
CANDLE_STYLE = dict(
    increasing_fillcolor="#0d2f1a",   # dark green body
    increasing_line_color="#00ff66",  # bright green wick/border
    decreasing_fillcolor="#2f0d0d",   # dark red body
    decreasing_line_color="#ff3344",  # bright red wick/border
    line_width=1,
)

# Dark chart background applied via fig.update_layout(**CHART_LAYOUT).
# Legend + hover use green text on a solid black box so EMA/series labels read.
CHART_LAYOUT = dict(
    plot_bgcolor="#0f0f0f",
    paper_bgcolor="#000000",
    font=dict(color="#e6e6e6", size=11),
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    legend=dict(bgcolor="#000000", bordercolor="#00ff41", borderwidth=1,
                font=dict(color="#00ff41", size=11)),
    hoverlabel=dict(bgcolor="#000000", bordercolor="#00ff41",
                    font=dict(color="#00ff41", size=11)),
)

# EMA overlay colours
EMA_COLORS = {
    "EMA_20": "#ff9800",   # amber
    "EMA_50": "#ab47bc",   # purple
}

# RSI line style
RSI_LINE   = dict(color="#7986cb", width=1.5)
RSI_OB     = dict(dash="dot", color="#ef5350", width=1)   # overbought (70)
RSI_OS     = dict(dash="dot", color="#26a69a", width=1)   # oversold  (30)