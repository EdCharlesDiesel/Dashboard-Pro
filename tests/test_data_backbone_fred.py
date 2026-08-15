"""Guard: the FRED fetch must not spoof a browser User-Agent.

Measured 2026-08-15 inside the running worker container, same URL, same
timeout, only the header differing::

    worker headers -> FAILED ReadTimeout after 20.1s
    no custom UA   -> 200  0.3s  423307 bytes

`fred.stlouisfed.org` hangs for a spoofed `Mozilla/5.0` agent and answers
immediately for `requests`' own. That single header is why `fred_series` sat
empty while the worker "ran" daily: every series raised ReadTimeout, and
`worker.refresh_fred` logs that at WARNING, so nothing ever alerted.

Pure inspection — no network.
"""
from __future__ import annotations

import inspect

from src.data_backbone import data_access


def test_fetch_fred_does_not_spoof_a_browser_user_agent():
    ua = (data_access.HEADERS or {}).get("User-Agent", "")
    assert "Mozilla" not in ua, (
        "A spoofed browser UA makes fred.stlouisfed.org hang until the read "
        "timeout — measured 20.1s vs 0.3s without it. Got: " + ua
    )


def test_fetch_fred_still_sends_a_timeout():
    """Whatever the headers, the call must not be able to hang forever."""
    src = inspect.getsource(data_access.fetch_fred)
    assert "timeout=" in src, "fetch_fred must pass a timeout to requests.get"
