"""Unit tests for the MT5 MCP server's pure helpers and its trade guardrails.

The module is a live-terminal entrypoint (omitted from the coverage scope), but
the parts that decide *whether an order is legal and correctly sized* are pure
and are exactly the parts worth pinning down — a wrong volume step or filling
mode is a rejected order, and a leaky guardrail is a real trade.

Skipped wholesale off Windows: `MetaTrader5` has no wheel there, matching the
`sys_platform == "win32"` marker in requirements.txt.
"""
import pytest

pytest.importorskip("MetaTrader5", reason="Windows-only broker package")
pytest.importorskip("mcp", reason="MCP SDK is a Windows-only extra here")

import MetaTrader5 as mt5  # noqa: E402
from src.services import mt5_mcp  # noqa: E402


class TestRoundVolume:
    """Brokers reject an off-step lot size outright, so snap before sending."""

    def test_snaps_to_step(self):
        assert mt5_mcp.round_volume(0.037, 0.01, 0.01, 200.0) == pytest.approx(0.04)

    def test_clamps_to_minimum(self):
        assert mt5_mcp.round_volume(0.001, 0.01, 0.01, 200.0) == pytest.approx(0.01)

    def test_clamps_to_maximum(self):
        assert mt5_mcp.round_volume(500.0, 0.01, 0.01, 200.0) == pytest.approx(200.0)

    def test_zero_step_falls_back_rather_than_dividing_by_zero(self):
        assert mt5_mcp.round_volume(0.05, 0.0, 0.01, 200.0) == pytest.approx(0.05)

    def test_coarse_step_broker(self):
        # Some brokers only allow whole lots.
        assert mt5_mcp.round_volume(2.4, 1.0, 1.0, 100.0) == pytest.approx(2.0)


class TestPickFilling:
    """An unsupported filling policy is a silent order rejection."""

    def test_prefers_fok_when_supported(self):
        assert mt5_mcp.pick_filling(1) == mt5.ORDER_FILLING_FOK

    def test_ioc_when_only_ioc(self):
        assert mt5_mcp.pick_filling(2) == mt5.ORDER_FILLING_IOC

    def test_fok_wins_when_both_supported(self):
        assert mt5_mcp.pick_filling(3) == mt5.ORDER_FILLING_FOK

    def test_falls_back_to_return_when_neither(self):
        assert mt5_mcp.pick_filling(0) == mt5.ORDER_FILLING_RETURN


class TestInstrumentMapping:
    """Delegates to broker_symbols — the one mapping the whole app shares."""

    def test_plain_symbol(self):
        assert mt5_mcp.instrument_for("EURUSD") == "EUR/USD"

    def test_broker_suffix(self):
        assert mt5_mcp.instrument_for("EURUSDm") == "EUR/USD"

    def test_metal_alias(self):
        assert mt5_mcp.instrument_for("GOLD") == "XAU/USD"

    def test_unmapped_symbol_is_none(self):
        assert mt5_mcp.instrument_for("NAS100") is None


class TestTimestamps:
    def test_iso_is_utc(self):
        assert mt5_mcp.iso(0) == "1970-01-01T00:00:00+00:00"

    def test_stamp_adds_iso_siblings(self):
        row = mt5_mcp.stamp({"time": 1700000000, "volume": 0.1})
        assert row["time_iso"] == "2023-11-14T22:13:20+00:00"
        assert row["time"] == 1700000000  # the raw epoch survives alongside it
        assert row["volume"] == 0.1

    def test_stamp_ignores_zero_and_missing_times(self):
        # MT5 reports an unset expiry as 0; that is "none", not the epoch.
        assert "time_expiration_iso" not in mt5_mcp.stamp({"time_expiration": 0})

    def test_stamp_ignores_booleans(self):
        assert "time_iso" not in mt5_mcp.stamp({"time": True})


class TestGuardrails:
    """Four gates stand between a tool call and a real order."""

    def test_unconfirmed_call_is_refused(self):
        with pytest.raises(mt5_mcp.MT5Error, match="confirm=true"):
            mt5_mcp._confirmed(False)

    def test_confirmed_call_passes(self):
        assert mt5_mcp._confirmed(True) is None

    def test_trading_disabled_blocks_everything(self, monkeypatch):
        monkeypatch.setattr(mt5_mcp, "ALLOW_TRADING", False)
        with pytest.raises(mt5_mcp.MT5Error, match="trading is disabled"):
            mt5_mcp._check_tradable("EURUSD", 0.01)

    def test_volume_cap_is_enforced(self, monkeypatch):
        monkeypatch.setattr(mt5_mcp, "ALLOW_TRADING", True)
        monkeypatch.setattr(mt5_mcp, "MAX_VOLUME", 0.5)
        with pytest.raises(mt5_mcp.MT5Error, match="exceeds the MT5_MAX_VOLUME cap"):
            mt5_mcp._check_tradable("EURUSD", 0.75)

    def test_symbol_whitelist_is_enforced(self, monkeypatch):
        monkeypatch.setattr(mt5_mcp, "ALLOW_TRADING", True)
        monkeypatch.setattr(mt5_mcp, "ALLOWED_SYMBOLS", {"EURUSD"})
        with pytest.raises(mt5_mcp.MT5Error, match="not in MT5_ALLOWED_SYMBOLS"):
            mt5_mcp._check_tradable("XAUUSD", 0.01)

    def test_whitelist_allows_listed_symbol(self, monkeypatch):
        monkeypatch.setattr(mt5_mcp, "ALLOW_TRADING", True)
        monkeypatch.setattr(mt5_mcp, "ALLOWED_SYMBOLS", {"EURUSD"})
        monkeypatch.setattr(mt5_mcp, "MAX_VOLUME", 1.0)
        # No terminal running under pytest, so terminal_info() is None and the
        # Algo Trading check is skipped -- the point here is that nothing raised.
        assert mt5_mcp._check_tradable("EURUSD", 0.01) is None


class TestReadOnlySeparationFromMt5Link:
    """`mt5_link` must stay read-only; trading lives only in this module."""

    def test_mt5_link_exposes_no_order_functions(self):
        from src.services import mt5_link
        assert not [n for n in dir(mt5_link)
                    if "order" in n.lower() or "send" in n.lower()]
