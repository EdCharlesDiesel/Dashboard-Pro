"""Local debug mode — one flag, no hostname guessing.

Set DASHBOARD_ENV=dev in your local launch script only. Unset (or any value
other than "dev") means production: the debug panel renders nothing and
cached_ohlc's source/asof params default to live data with no truncation.
"""
import os


def is_dev() -> bool:
    return os.getenv("DASHBOARD_ENV", "prod").lower() == "dev"
