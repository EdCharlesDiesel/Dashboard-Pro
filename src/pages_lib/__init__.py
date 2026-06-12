"""Base classes and the central navigation registry for every page."""
from src.pages_lib.base import BloombergPage, PageContext
from src.pages_lib.navigation import NAV_ENTRIES, NavEntry

__all__ = ["BloombergPage", "PageContext", "NAV_ENTRIES", "NavEntry"]
