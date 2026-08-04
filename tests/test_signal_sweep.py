"""The sweep registry must never drift from the pages' actual persist_signals calls.

A source tag that exists in a page but not in `PAGES` is a page that silently
never gets swept — the exact failure this module was written to fix, and one
that is invisible until someone notices the Source Scorecard is thin. So the
registry is asserted against the call sites themselves rather than against a
hand-maintained list (CLAUDE.md already carries four tags that no longer exist
in code, which is how this class of drift usually shows up).

Pure filesystem work: no network, no DB, no Streamlit runtime.
"""
import re
from pathlib import Path

import pytest

from src.services.signal_sweep import PAGES

REPO = Path(__file__).resolve().parents[1]
CALL = re.compile(r'persist_signals\(\s*["\']([a-z_0-9]+)["\']')
AUDIT_CALL = re.compile(r'log_tool_usage\(\s*["\']([a-z_0-9]+)["\']')


def _tags_in_code(pattern=CALL):
    """Every source tag literally passed to persist_signals() across the app."""
    found = {}
    for sub in ("src", "pages"):
        for path in (REPO / sub).rglob("*.py"):
            if "__pycache__" in str(path) or "archive" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for tag in pattern.findall(text):
                found.setdefault(tag, []).append(path.relative_to(REPO).as_posix())
    # The module docstring in signal_store.py carries an illustrative example.
    found.pop("setup_ranker_example", None)
    return found


class TestRegistryMatchesCallSites:
    def test_every_persisting_page_is_registered(self):
        registered = {source for source, _ in PAGES}
        missing = set(_tags_in_code()) - registered
        assert not missing, (
            "these source tags persist signals but are not in signal_sweep.PAGES, "
            "so they will never be swept: {0}".format(sorted(missing)))

    def test_no_registered_tag_is_dead(self):
        from src.services.signal_sweep import AUDIT_ONLY
        # An audit-only page is swept but writes to tool_usage_log, so it is
        # allowed to have no persist_signals call — it must have the other one.
        in_code = set(_tags_in_code()) | (set(_tags_in_code(AUDIT_CALL)) & AUDIT_ONLY)
        dead = {source for source, _ in PAGES} - in_code
        assert not dead, (
            "registered in signal_sweep.PAGES but nothing calls persist_signals "
            "(or log_tool_usage, for AUDIT_ONLY) with this tag: {0}".format(sorted(dead)))

    def test_audit_only_pages_are_registered_and_really_audit_only(self):
        from src.services.signal_sweep import AUDIT_ONLY
        registered = {source for source, _ in PAGES}
        assert AUDIT_ONLY <= registered, "AUDIT_ONLY tag missing from PAGES"
        # The point of the exemption is that these no longer reach trade_setups;
        # if one starts persisting again it must leave the exemption list.
        still_persisting = AUDIT_ONLY & set(_tags_in_code())
        assert not still_persisting, (
            "AUDIT_ONLY pages must not call persist_signals — the Source Scorecard "
            "cannot resolve their non-directional reads: {0}".format(
                sorted(still_persisting)))

    def test_source_tags_are_unique(self):
        tags = [source for source, _ in PAGES]
        assert len(tags) == len(set(tags)), "duplicate source tag in PAGES"


class TestRegistryIsWellFormed:
    @pytest.mark.parametrize("source,path", PAGES)
    def test_page_file_exists(self, source, path):
        assert (REPO / path).is_file(), "{0} -> missing page {1}".format(source, path)

    @pytest.mark.parametrize("source,path", PAGES)
    def test_tag_fits_the_column(self, source, path):
        # trade_setups.source is VARCHAR(20); a longer tag fails on write, which
        # is why cot_composite is not called cot_composite_trade_signal.
        assert len(source) <= 20, "source tag {0!r} exceeds VARCHAR(20)".format(source)

    @pytest.mark.parametrize("source,path", PAGES)
    def test_paths_are_repo_relative_posix(self, source, path):
        assert not path.startswith("/") and ":" not in path, \
            "page path must be repo-relative: {0}".format(path)


class TestPrepareHooks:
    """Widget presets exist only for pages whose default state can't signal."""

    def test_prepare_keys_are_registered_sources(self):
        from src.services.signal_sweep import PREPARE
        registered = {source for source, _ in PAGES}
        unknown = set(PREPARE) - registered
        assert not unknown, "PREPARE hook for unregistered source: {0}".format(
            sorted(unknown))

    def test_prepare_values_are_callable(self):
        from src.services.signal_sweep import PREPARE
        assert all(callable(fn) for fn in PREPARE.values())


class TestBrokerSync:
    """The pre-run sync must never be able to abort the sweep.

    It reads a live terminal that is Windows-only, may not be running, and may
    not be logged in. A stale balance is bad; a scheduled job that dies before
    running any page because the terminal was closed is worse.
    """

    def test_returns_a_result_when_mt5_link_is_missing(self, monkeypatch):
        # Stands in for the Linux deploy container, where MetaTrader5 has no
        # wheel. Both halves are needed and the test is order-dependent without
        # them: `from src.services import mt5_link` resolves through the *package
        # attribute* once anything else in the suite has imported the submodule,
        # so blocking sys.modules alone passes in isolation and fails in a full
        # run. Removing the attribute forces the import machinery back to
        # sys.modules, where the None entry raises ImportError.
        import sys
        import src.services as services_pkg
        from src.services import signal_sweep
        monkeypatch.delattr(services_pkg, "mt5_link", raising=False)
        monkeypatch.setitem(sys.modules, "src.services.mt5_link", None)
        res = signal_sweep.sync_broker_state()
        assert res["ok"] is False and "unavailable" in res["message"]

    def test_a_raising_sync_is_caught(self, monkeypatch):
        from src.services import mt5_link, signal_sweep

        def boom():
            raise RuntimeError("terminal not reachable")

        monkeypatch.setattr(mt5_link, "sync", boom)
        res = signal_sweep.sync_broker_state()
        assert res["ok"] is False and "terminal not reachable" in res["message"]

    def test_a_successful_sync_reports_balance_and_count(self, monkeypatch):
        from src.services import mt5_link, signal_sweep
        monkeypatch.setattr(mt5_link, "sync", lambda: {
            "ok": True, "message": "3 position(s) from account 1 @ X",
            "saved": 3, "account": {"balance": 1989.0}})
        res = signal_sweep.sync_broker_state()
        assert res == {"ok": True, "message": "3 position(s) from account 1 @ X",
                       "balance": 1989.0, "positions": 3}

    def test_sweep_can_opt_out_of_syncing(self, monkeypatch):
        from src.services import signal_sweep
        called = []
        monkeypatch.setattr(signal_sweep, "sync_broker_state",
                            lambda: called.append(1) or {"ok": True, "message": "",
                                                         "balance": 0, "positions": 0})
        monkeypatch.setattr(signal_sweep, "_counts_by_source", lambda: {})
        signal_sweep.sweep_once(only=["__none__"], sync=False)
        assert called == []
