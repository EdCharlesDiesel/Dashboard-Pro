"""Local debug mode — env-gated data-source switching and historical replay."""
from src.debug.env import is_dev
from src.debug.panel import debug_panel

__all__ = ["is_dev", "debug_panel"]
