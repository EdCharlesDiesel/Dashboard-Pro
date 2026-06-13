"""Optional Claude-powered narrative summaries for the Reports page.

The Reports page always works on rule-based analytics. When an Anthropic API
key is configured (``[api] ANTHROPIC_API_KEY`` or the ``ANTHROPIC_API_KEY`` env
var) *and* the ``anthropic`` SDK is installed, this module turns a report's
structured data into a short, trader-facing narrative via the Claude API.

Everything degrades gracefully: if the SDK is missing or no key is present,
:func:`is_available` returns ``False`` and the page hides the AI controls.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.core import observability, secrets

# Default model — Anthropic's most capable Opus-tier model.
_MODEL = "claude-opus-4-8"

_SYSTEM = (
    "You are a senior forex/metals day-trading desk analyst. You read "
    "structured report data from a trading dashboard and write a tight, "
    "plain-language briefing for the trader who runs the desk.\n\n"
    "Rules:\n"
    "- Lead with the single most important takeaway in one sentence.\n"
    "- Then 3-6 short bullets of concrete, data-grounded observations.\n"
    "- Only state things the data supports; never invent numbers.\n"
    "- Flag risks explicitly (loss streaks, stale data, correlation stacking, "
    "instruments stuck in manipulation/distribution).\n"
    "- Be specific with instruments, phases, win rate, R-multiples.\n"
    "- No preamble, no disclaimers, no markdown headers. Bullets and short "
    "sentences only. This is internal desk colour, not financial advice."
)


def is_available() -> bool:
    """True when an AI summary can actually be produced."""
    if not secrets.anthropic_api_key():
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def unavailable_reason() -> str:
    """Human-readable reason the AI summary is disabled (for the UI)."""
    if not secrets.anthropic_api_key():
        return ("No Anthropic API key. Add `ANTHROPIC_API_KEY` under `[api]` in "
                "`.streamlit/secrets.toml` to enable AI summaries.")
    try:
        import anthropic  # noqa: F401
    except Exception:
        return "The `anthropic` package isn't installed. Run `pip install anthropic`."
    return ""


def summarize_report(title: str, data: Dict[str, Any],
                     model: str = _MODEL) -> str:
    """Return a narrative summary of ``data`` (a report dict), or raise.

    Caller should gate on :func:`is_available` first. Recorded as an
    ``ai_summary`` observability event (without the response body).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=secrets.anthropic_api_key())
    payload = json.dumps(data, default=str, indent=2)
    user = (
        f"Report: {title}\n\n"
        f"Here is the structured report data as JSON:\n```json\n{payload}\n```\n\n"
        "Write the desk briefing for this report."
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        observability.log_event(
            "ai_summary", report=title, model=model, ok=True,
            output_tokens=getattr(resp.usage, "output_tokens", None),
        )
        return text or "(The model returned no text.)"
    except Exception as exc:  # surfaced to the page; also recorded
        observability.log_event("ai_summary", level="ERROR", report=title,
                                model=model, ok=False, error=str(exc)[:200])
        raise
