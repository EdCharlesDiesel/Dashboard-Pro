"""Daily Trading System — Bloomberg-terminal entry point.

The implementation lives in `src/pages_lib/daily_trading/`. This file is the
thin Streamlit entrypoint; it just instantiates `DailyTradingPage` and calls
`run()`. All logic and presentation are in the OOP framework.
"""
from src.pages_lib.daily_trading import DailyTradingPage

DailyTradingPage().run()
