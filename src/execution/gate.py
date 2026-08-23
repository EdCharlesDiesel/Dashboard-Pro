"""
Pre-trade gate and position sizing.

Pure Python — no MT5, no DB, no I/O. Everything here is a function of its
arguments so it can be unit-tested exhaustively without a terminal running.
That matters: this module is the only thing standing between a malformed
signal and your account.

Lives at: src/execution/gate.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import time as dtime

__all__ = [
    "GateConfig",
    "Signal",
    "MarketSnapshot",
    "AccountState",
    "GateResult",
    "run_gate",
    "SizingResult",
    "compute_lots",
    "classify_order_type",
    "looks_synthetic",
]


# ---------------------------------------------------------------------------
# value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    signal_id: str
    symbol: str
    direction: str          # 'BUY' | 'SELL'
    entry: float
    stop: float
    tp1: float | None = None
    tp2: float | None = None
    risk_pct: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def stop_distance(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def is_buy(self) -> bool:
        return self.direction.upper() == "BUY"


@dataclass(frozen=True)
class MarketSnapshot:
    """What the terminal says about the symbol right now."""
    symbol: str
    bid: float
    ask: float
    point: float
    digits: int
    tick_value: float           # account currency per tick per 1.0 lot
    tick_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    trade_allowed: bool = True
    stops_level_points: float = 0.0     # broker minimum stop distance
    margin_per_lot: float = 0.0

    @property
    def spread_points(self) -> float:
        return (self.ask - self.bid) / self.point if self.point else float("inf")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class AccountState:
    balance: float
    equity: float
    free_margin: float
    open_positions: int
    open_symbols: tuple[str, ...] = ()
    daily_loss_r: float = 0.0
    enabled: bool = False
    dry_run: bool = True


@dataclass
class GateConfig:
    # --- what may be traded at all -----------------------------------------
    symbol_whitelist: tuple[str, ...] = ("XAUUSD", "XAGUSD")

    # --- sanity on the levels themselves -----------------------------------
    # The 1.10000 / 1.09500 / 1.11000 stub would fail on both of these.
    max_entry_deviation_points: float = 300.0   # entry vs current market
    reject_suspicious_round: bool = True

    # --- risk shape ---------------------------------------------------------
    min_stop_points: float = 50.0
    max_stop_points: float = 5000.0
    min_rr_tp1: float = 1.0
    default_risk_pct: float = 0.5
    max_risk_pct: float = 2.0

    # --- execution conditions ----------------------------------------------
    max_spread_points: float = 40.0

    # --- exposure limits ----------------------------------------------------
    max_concurrent_positions: int = 3
    one_position_per_symbol: bool = True
    max_daily_loss_r: float = 3.0

    # --- session window (broker/server clock) ------------------------------
    session_start: dtime | None = None
    session_end: dtime | None = None


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)


# ---------------------------------------------------------------------------
# the stub detector
# ---------------------------------------------------------------------------

def looks_synthetic(sig: Signal, snap: MarketSnapshot) -> bool:
    """Heuristic for placeholder/default levels leaking into a live signal.

    Motivating case: entry 1.10000, stop 1.09500, tp1 1.11000, tp2 1.12000 —
    a round 50-pip stop with exactly 100 and 200 pips to target, i.e. exactly
    2.00R. Real fib legs do not land on grid points like that.

    Flags when the entry AND stop are both round to a suspiciously coarse grid
    relative to the instrument's precision. Deliberately conservative: it is
    far cheaper to hand-check a false positive than to auto-execute a stub.
    """
    if snap.point <= 0:
        return False

    # A "coarse grid" = 1000 points. For a 5-digit FX pair that is 100 pips;
    # for gold at 0.01 point it is $10.
    grid = snap.point * 1000.0

    def on_grid(x: float, g: float) -> bool:
        return abs(x / g - round(x / g)) < 1e-6

    half_grid = grid / 2.0
    entry_round = on_grid(sig.entry, half_grid)
    stop_round = on_grid(sig.stop, half_grid)
    if not (entry_round and stop_round):
        return False

    # Both round AND the R-multiple to tp1 is a clean integer -> almost
    # certainly generated, not measured.
    if sig.tp1 is not None and sig.stop_distance > 0:
        rr = abs(sig.tp1 - sig.entry) / sig.stop_distance
        if abs(rr - round(rr)) < 1e-6:
            return True

    return True


def classify_order_type(sig: Signal, snap: MarketSnapshot,
                        market_tolerance_points: float = 20.0) -> str:
    """Decide MARKET / LIMIT / STOP from where entry sits vs the current book.

    Blindly sending a market order for an entry that is 80 points away is how
    you turn a 2R setup into a 1.2R one.
    """
    ref = snap.ask if sig.is_buy else snap.bid
    dist_points = abs(sig.entry - ref) / snap.point if snap.point else float("inf")

    if dist_points <= market_tolerance_points:
        return "MARKET"
    if sig.is_buy:
        return "BUY_LIMIT" if sig.entry < ref else "BUY_STOP"
    return "SELL_LIMIT" if sig.entry > ref else "SELL_STOP"


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------

def run_gate(sig: Signal, snap: MarketSnapshot, acct: AccountState,
             cfg: GateConfig, now: dtime | None = None) -> GateResult:
    """Every reason a signal must not reach order_send, in one place."""
    res = GateResult(ok=True)

    # --- global switches ----------------------------------------------------
    if not acct.enabled:
        res.block("executor disabled (kill switch)")
    if not snap.trade_allowed:
        res.block(f"trading not allowed on {sig.symbol} (market closed or disabled)")

    # --- instrument ---------------------------------------------------------
    if sig.symbol not in cfg.symbol_whitelist:
        res.block(f"symbol {sig.symbol} not in whitelist {cfg.symbol_whitelist}")

    # --- direction / geometry ----------------------------------------------
    if sig.direction.upper() not in ("BUY", "SELL"):
        res.block(f"bad direction {sig.direction!r}")
    if sig.stop_distance <= 0:
        res.block("stop equals entry")
    if sig.is_buy and sig.stop >= sig.entry:
        res.block("BUY with stop at or above entry")
    if not sig.is_buy and sig.stop <= sig.entry:
        res.block("SELL with stop at or below entry")

    if sig.tp1 is not None:
        if sig.is_buy and sig.tp1 <= sig.entry:
            res.block("BUY with tp1 at or below entry")
        if not sig.is_buy and sig.tp1 >= sig.entry:
            res.block("SELL with tp1 at or above entry")
        if sig.stop_distance > 0:
            rr = abs(sig.tp1 - sig.entry) / sig.stop_distance
            if rr < cfg.min_rr_tp1:
                res.block(f"tp1 R:R {rr:.2f} below minimum {cfg.min_rr_tp1}")

    # --- stop distance vs instrument ---------------------------------------
    if snap.point > 0:
        stop_pts = sig.stop_distance / snap.point
        if stop_pts < cfg.min_stop_points:
            res.block(f"stop {stop_pts:.0f}pts below minimum {cfg.min_stop_points:.0f}")
        if stop_pts > cfg.max_stop_points:
            res.block(f"stop {stop_pts:.0f}pts above maximum {cfg.max_stop_points:.0f}")
        if snap.stops_level_points and stop_pts < snap.stops_level_points:
            res.block(f"stop {stop_pts:.0f}pts inside broker stops level "
                      f"{snap.stops_level_points:.0f}")

    # --- is this signal anywhere near the actual market? --------------------
    if snap.point > 0:
        ref = snap.ask if sig.is_buy else snap.bid
        dev = abs(sig.entry - ref) / snap.point
        if dev > cfg.max_entry_deviation_points:
            res.block(f"entry {dev:.0f}pts from market "
                      f"(max {cfg.max_entry_deviation_points:.0f}) — stale or stub")

    if cfg.reject_suspicious_round and looks_synthetic(sig, snap):
        res.block("levels look synthetic (round grid + integer R) — "
                  "suspected placeholder, not a measured fib leg")

    # --- execution conditions ----------------------------------------------
    if snap.spread_points > cfg.max_spread_points:
        res.block(f"spread {snap.spread_points:.0f}pts above "
                  f"{cfg.max_spread_points:.0f}")

    # --- exposure -----------------------------------------------------------
    if acct.open_positions >= cfg.max_concurrent_positions:
        res.block(f"{acct.open_positions} positions open, "
                  f"max {cfg.max_concurrent_positions}")
    if cfg.one_position_per_symbol and sig.symbol in acct.open_symbols:
        res.block(f"already holding {sig.symbol}")
    if acct.daily_loss_r >= cfg.max_daily_loss_r:
        res.block(f"daily loss {acct.daily_loss_r:.2f}R at limit "
                  f"{cfg.max_daily_loss_r:.2f}R")

    # --- risk sizing sanity -------------------------------------------------
    risk = sig.risk_pct if sig.risk_pct is not None else cfg.default_risk_pct
    if risk <= 0:
        res.block(f"non-positive risk {risk}")
    if risk > cfg.max_risk_pct:
        res.block(f"risk {risk}% above cap {cfg.max_risk_pct}%")

    # --- session ------------------------------------------------------------
    if cfg.session_start and cfg.session_end and now:
        in_session = (cfg.session_start <= now <= cfg.session_end
                      if cfg.session_start <= cfg.session_end
                      else now >= cfg.session_start or now <= cfg.session_end)
        if not in_session:
            res.block(f"outside session {cfg.session_start}-{cfg.session_end}")

    # --- non-blocking observations -----------------------------------------
    if acct.equity < acct.balance * 0.95:
        res.warn(f"equity {acct.equity:,.2f} is 5%+ below balance "
                 f"{acct.balance:,.2f} — open drawdown")

    return res


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------

@dataclass
class SizingResult:
    lots: float
    ok: bool
    risk_amount: float
    risk_per_lot: float
    raw_lots: float
    reason: str = ""


def compute_lots(sig: Signal, snap: MarketSnapshot, acct: AccountState,
                 risk_pct: float) -> SizingResult:
    """Risk-based sizing, rounded DOWN to the broker's volume step.

    Always down. Rounding up to "get closer" to the target risk is how a 0.5%
    risk quietly becomes 0.8% on a small account with a coarse step.

    risk_per_lot = stop_distance / tick_size * tick_value, i.e. what one full
    lot loses if the stop is hit. Works across FX, metals and CFDs without
    special-casing pip values.
    """
    risk_amount = acct.balance * (risk_pct / 100.0)

    if snap.tick_size <= 0 or snap.tick_value <= 0:
        return SizingResult(0.0, False, risk_amount, 0.0, 0.0,
                            "invalid tick_size/tick_value from terminal")

    ticks = sig.stop_distance / snap.tick_size
    risk_per_lot = ticks * snap.tick_value
    if risk_per_lot <= 0:
        return SizingResult(0.0, False, risk_amount, risk_per_lot, 0.0,
                            "non-positive risk per lot")

    raw = risk_amount / risk_per_lot

    step = snap.volume_step if snap.volume_step > 0 else 0.01
    lots = math.floor(raw / step) * step
    lots = round(lots, 8)

    if lots < snap.volume_min:
        return SizingResult(
            0.0, False, risk_amount, risk_per_lot, raw,
            f"required {raw:.4f} lots below broker minimum {snap.volume_min} — "
            f"stop too wide for {risk_pct}% on a {acct.balance:,.2f} balance")

    if lots > snap.volume_max:
        lots = snap.volume_max

    # Margin headroom. Leave 20% buffer so a normal adverse move does not
    # trip a margin call on an otherwise valid trade.
    if snap.margin_per_lot > 0:
        needed = lots * snap.margin_per_lot
        if needed > acct.free_margin * 0.8:
            affordable = math.floor(
                (acct.free_margin * 0.8 / snap.margin_per_lot) / step) * step
            if affordable < snap.volume_min:
                return SizingResult(
                    0.0, False, risk_amount, risk_per_lot, raw,
                    f"insufficient free margin ({acct.free_margin:,.2f}) "
                    f"for minimum size")
            lots = round(affordable, 8)

    actual_risk = lots * risk_per_lot
    return SizingResult(
        lots, True, risk_amount, risk_per_lot, raw,
        f"{lots} lots -> {actual_risk:,.2f} risk "
        f"({actual_risk / acct.balance * 100:.2f}% of balance)")
