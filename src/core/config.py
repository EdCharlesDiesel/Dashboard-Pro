from dataclasses import dataclass, field
from typing import Dict
@dataclass
class AppConfig:
    """Application configuration — all tuneable parameters in one place."""
    version: str = "2.0.1"
    last_updated: str = "2026-09-05"

    assets: Dict[str, str] = field(default_factory=lambda: {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/ZAR": "ZAR=X",
        "AUD/USD": "AUDUSD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CAD": "CAD=X",
        "USD/CHF": "CHF=X",
        "XAU/USD": "GC=F",
        "BTC/USD": "BTC-USD",
    })

    timeframes: Dict[str, Dict] = field(default_factory=lambda: {
        "Weekly": {"interval": "1wk", "period": "3mo"},
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

    pair_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 1.5, "GBP/USD": 1.8, "USD/JPY": 1.5, "USD/ZAR": 2.5,
        "AUD/USD": 1.5, "NZD/USD": 1.6, "USD/CAD": 1.5, "USD/CHF": 1.5,
        "XAU/USD": 2.0, "BTC/USD": 2.0,
    })

    pair_min_stop: Dict[str, float] = field(default_factory=lambda: {
        "EUR/USD": 0.0010, "GBP/USD": 0.0015, "USD/JPY": 0.10, "USD/ZAR": 0.05,
        "AUD/USD": 0.0010, "NZD/USD": 0.0010, "USD/CAD": 0.0010, "USD/CHF": 0.0010,
        "XAU/USD": 2.00, "BTC/USD": 500.0,
    })

    dxy_symbol: str = "DX-Y.NYB"
    quantconnect_url: str = "http://localhost:5000/data"
    cache_ttl: int = 300
    auto_refresh_interval: int = 300


default_config = AppConfig()
