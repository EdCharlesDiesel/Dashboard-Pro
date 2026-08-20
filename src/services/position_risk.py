"""What the open book is actually risking, priced by the stochastic engine.

The desk question this answers: *every position has a stop and a target — so is
each one worth holding?* A stop makes a position survivable, which is not the
same as making it worth having. Three numbers decide it, and the first two are
routinely mistaken for the third:

1. **P(target before stop)** — the scale function from `src/core/stochastic.py`.
2. **Reward:risk** — how much the target pays relative to what the stop costs.
3. **Edge** — P(target) minus the win rate that reward:risk *requires*. Only
   this one can be negative while the other two look healthy.

Measured on the live book on 2026-08-10: four positions, every one with a
positive probability edge (+3.2pp to +6.8pp), and a book expected value of
**zero** — between -$0.45 and +$1.36 on $2,049 of equity, against $446 of
downside. Stops sat wider than targets on every ZAR position, so a 75.3% hit
rate on GBP/ZAR paying 0.39:1 still lost money. Probability edge alone is not
an edge; it has to survive the payoff ratio.

**Shorts.** `barrier_probabilities` is long-convention (target above spot, stop
below). A short is run on the reciprocal: Y = 1/S is GBM with drift
``-mu + sigma^2``, whose log-drift is exactly ``-nu`` — so the short's barriers
map to ``1/target`` and ``1/stop`` on Y, and the same closed form applies with no
second implementation to keep in step.

Pure functions take calibrated :class:`GBMParams` and a rate map; only
:func:`load_book` and :func:`main` touch I/O, so the maths is testable without a
broker, a database or a network.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.stochastic import (GBMParams, barrier_probabilities,
                                 calibrate_gbm, simulate_gbm)
from src.core.todays_trades import (CONTRACT_SIZES, DEFAULT_CONTRACT, _side,
                                    LONG, pos_price, pos_volume, usd_rates)
from src.instruments.registry import INSTRUMENTS

DEFAULT_HORIZON_DAYS = 21          # ~one trading month
PERIODS_PER_YEAR = 252
DEFAULT_QUANTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class Position:
    """One open position, normalised. ``stop``/``target`` are None when unset."""

    pair: str
    side: str                      # "Long" / "Short"
    lots: float
    entry: float
    spot: float
    stop: Optional[float] = None
    target: Optional[float] = None

    @property
    def bracketed(self) -> bool:
        """A barrier problem needs two barriers."""
        return bool(self.stop) and bool(self.target)


@dataclass(frozen=True)
class PositionRisk:
    """The verdict on one position. ``edge_pp`` is the number that decides."""

    pair: str
    side: str
    p_target: float
    p_stop: float
    reward_risk: float
    breakeven: float
    edge_pp: float
    win_usd: float
    loss_usd: float
    ev_usd: float
    # Share of paths that touch neither barrier inside the horizon. High values
    # mean the verdict is about a trade that mostly will not have happened yet.
    p_unresolved: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def contract_units(pair: str, lots: float) -> float:
    """Base units for ``lots``. Metals and oil are not 100k contracts."""
    return float(lots) * CONTRACT_SIZES.get(pair, DEFAULT_CONTRACT)


def usd_pnl(pair: str, side: Any, lots: float, entry: float, exit_price: float,
            rates: Optional[Dict[str, float]] = None) -> float:
    """P&L in USD for a move from ``entry`` to ``exit_price``.

    P&L accrues in the *quote* currency, so a ZAR-quoted pair earns rand that
    have to be converted. Skipping that conversion overstates every ZAR result
    by roughly 16x — the pair's own price is not the dollar rate.
    """
    parts = str(pair or "").split("/")
    if len(parts) != 2:
        return 0.0
    normalised = _side(side)
    if normalised is None or not lots or not entry or not exit_price:
        return 0.0
    sign = 1.0 if normalised == LONG else -1.0
    quote_pnl = sign * contract_units(pair, lots) * (float(exit_price) - float(entry))
    quote = parts[1].strip().upper()
    if quote == "USD":
        return quote_pnl
    rate = (rates or {}).get(quote)
    if not rate:
        return quote_pnl          # no rate known — report unconverted, never zero
    return quote_pnl * rate


def book_projection(rows: Sequence[Dict[str, Any]],
                    rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """What the open book is worth now, at every target, and at every stop.

    Returns ``{floating, at_target, at_stop, converted, unconverted}``. All
    three figures are in account currency when ``converted`` is True.

    **Why this does not just call `usd_pnl`.** P&L accrues in the *quote*
    currency, so a ZAR cross has to be converted or it overstates by about
    16x. `usd_pnl` needs a rate table for that, and the Streamlit app runs in
    a Linux container where MetaTrader cannot be reached and no quote can be
    fetched. But the broker already did the conversion: every position carries
    its floating P/L in account currency, measured at a price we also store.
    Since P&L is linear in price, one division recovers the exact factor the
    broker used:

        factor   = profit_now / (price_now - entry)      [account ccy per price unit]
        at_target= factor * (target - entry)

    No contract sizes, no pip values, no quote lookup - and it agrees with the
    contract arithmetic to within a fraction of a percent, because it *is* the
    broker's own arithmetic run backwards.

    Falls back to `usd_pnl` when the broker figures are absent or the position
    is exactly at entry (where the factor is 0/0). If that fallback cannot
    convert either, the pair is named in ``unconverted`` and ``converted`` goes
    False - the caller must then not print a currency symbol on the number.
    """
    floating = at_target = at_stop = 0.0
    unconverted: List[str] = []

    for row in rows or []:
        pair = str(row.get("pair") or "")
        entry = row.get("entry_price")
        target = row.get("take_profit")
        stop = row.get("stop_loss")
        profit = row.get("profit")
        price_now = row.get("price_current")

        if profit is not None:
            floating += float(profit)

        if entry is None:
            continue
        entry = float(entry)

        span = (float(price_now) - entry) if price_now is not None else 0.0
        if profit is not None and span:
            factor = float(profit) / span
            if target is not None:
                at_target += factor * (float(target) - entry)
            if stop is not None:
                at_stop += factor * (float(stop) - entry)
            continue

        # Fallback: value it directly. usd_pnl reports unconverted rather than
        # zero when it has no rate, so check whether a rate was actually known.
        side = row.get("direction")
        lots = row.get("lot_size")
        quote = pair.split("/")[-1].strip().upper() if "/" in pair else ""
        if quote and quote != "USD" and not (rates or {}).get(quote):
            unconverted.append(pair)
        if target is not None:
            at_target += usd_pnl(pair, side, lots, entry, float(target), rates)
        if stop is not None:
            at_stop += usd_pnl(pair, side, lots, entry, float(stop), rates)

    return {
        "floating": floating,
        "at_target": at_target,
        "at_stop": at_stop,
        "converted": not unconverted,
        "unconverted": unconverted,
    }


def barrier_view(side: Any, spot: float, stop: float, target: float,
                 params: GBMParams, horizon_years: Optional[float] = None,
                 n_sims: int = 50_000) -> Dict[str, float]:
    """``barrier_probabilities`` for either direction.

    Longs pass straight through. Shorts go via the reciprocal (see module
    docstring), which keeps one tested implementation of the scale function
    rather than a mirrored copy that can drift out of step with it.
    """
    if _side(side) == LONG:
        return barrier_probabilities(spot, target, stop, params.sigma, params.mu,
                                     horizon_years=horizon_years, n_sims=n_sims)
    return barrier_probabilities(1.0 / spot, 1.0 / target, 1.0 / stop,
                                 params.sigma,
                                 -params.mu + params.sigma ** 2,
                                 horizon_years=horizon_years, n_sims=n_sims)


def assess(position: Position, params: GBMParams,
           rates: Optional[Dict[str, float]] = None,
           horizon_days: int = DEFAULT_HORIZON_DAYS,
           n_sims: int = 50_000,
           use_monte_carlo: bool = True) -> Optional[PositionRisk]:
    """Price one position, or ``None`` when it has no two barriers to price."""
    if not position.bracketed:
        return None
    horizon_years = horizon_days / float(PERIODS_PER_YEAR)
    view = barrier_view(position.side, position.spot, position.stop,
                        position.target, params,
                        horizon_years=horizon_years if use_monte_carlo else None,
                        n_sims=n_sims)
    # The closed form is primary, and neither Monte Carlo figure may replace it.
    # `breakeven_win_rate` is 1/(1+R:R): the share of trades held **to a
    # barrier** that must win. The scale function answers exactly that question
    # — P(target before stop, unlimited horizon) — and does it exactly.
    #
    # Both MC figures answer something else and are wrong here in opposite
    # directions. The unconditional `mc_p_take_profit` counts every path that
    # touched neither barrier inside the horizon as a loss, understating edge by
    # the unresolved fraction. The conditional `mc_p_tp_given_resolved` asks
    # "given one barrier was hit within the horizon, which?" — and on wide
    # barriers that is nearly 1.0, because a short horizon can reach a near
    # target but never a far stop. Measured on +4%/-10% barriers at 9% vol:
    # closed form 67.7%, conditional MC 99.96%. Substituting either turns a
    # losing trade into a spectacular one.
    #
    # The Monte Carlo earns its keep as a *diagnostic* instead: p_unresolved
    # says whether this verdict is about a trade that will mostly not have
    # happened yet.
    p_target = float(view["p_take_profit"])
    win = usd_pnl(position.pair, position.side, position.lots,
                  position.entry, position.target, rates)
    loss = usd_pnl(position.pair, position.side, position.lots,
                   position.entry, position.stop, rates)
    return PositionRisk(
        pair=position.pair, side=position.side,
        p_target=p_target, p_stop=1.0 - p_target,
        reward_risk=float(view["reward_risk"]),
        breakeven=float(view["breakeven_win_rate"]),
        edge_pp=(p_target - float(view["breakeven_win_rate"])) * 100.0,
        win_usd=win, loss_usd=loss,
        ev_usd=p_target * win + (1.0 - p_target) * loss,
        p_unresolved=float(view.get("mc_p_unresolved", 0.0)),
    )


def cone(position: Position, params: GBMParams,
         rates: Optional[Dict[str, float]] = None,
         quantiles: Sequence[float] = DEFAULT_QUANTILES,
         horizon_days: int = DEFAULT_HORIZON_DAYS,
         n_paths: int = 20_000, seed: Optional[int] = 7) -> Dict[str, Any]:
    """Where GBM puts the price at the horizon, and what each quantile is worth.

    Deliberately ignores the stop: this is the unmanaged distribution, which is
    what tells you whether a position is too large *before* asking whether its
    stop would save you.
    """
    ends = simulate_gbm(position.spot, params.mu, params.sigma,
                        horizon_days / float(PERIODS_PER_YEAR),
                        n_paths=n_paths, periods_per_year=PERIODS_PER_YEAR,
                        seed=seed)[:, -1]
    prices = [float(v) for v in np.percentile(ends, list(quantiles))]
    pnl = [usd_pnl(position.pair, position.side, position.lots,
                   position.entry, p, rates) for p in prices]
    # A short profits as price falls, so its worst P&L sits at the *highest*
    # price quantile. Report P&L ascending so the columns line up with the
    # price columns as "bad -> good" in both directions.
    if _side(position.side) != LONG:
        pnl = pnl[::-1]
    return {"pair": position.pair, "quantiles": list(quantiles),
            "prices": prices, "pnl_usd": pnl, "sigma": params.sigma}


def positions_from_store(rows: Sequence[Dict[str, Any]]) -> List[Position]:
    """Normalise `open_positions.load()` rows into :class:`Position`.

    Spot falls back to entry: the durable store keeps entry_price but no live
    quote, and a position priced at its own entry is a defensible default that
    never silently reports zero.
    """
    out: List[Position] = []
    for row in rows or []:
        pair = str(row.get("pair") or "").strip()
        side = _side(row.get("direction"))
        entry = pos_price(row)
        if not pair or not side or not entry:
            continue
        out.append(Position(
            pair=pair, side=side, lots=pos_volume(row), entry=entry,
            spot=float(row.get("price_current") or entry),
            stop=float(row.get("stop_loss") or row.get("sl") or 0) or None,
            target=float(row.get("take_profit") or row.get("tp") or 0) or None,
        ))
    return out


def verdict(risk: PositionRisk, horizon_days: int = DEFAULT_HORIZON_DAYS
            ) -> Dict[str, str]:
    """A take/skip call on one candidate, with the arithmetic that produced it.

    The reason always names both halves, because either alone misleads. "Hits
    the target 34.8% of the time" sounds like a bad trade and is a good one at
    2:1; "wins 78.8% of the time" sounds like a certainty and loses money at
    0.32:1. Only the pair of numbers is an argument.
    """
    needs, gets = risk.breakeven * 100.0, risk.p_target * 100.0
    take = risk.edge_pp > 0
    reason = ("{0:.2f}:1 pays for a {1:.1f}% hit rate; the model gives {2:.1f}%"
              .format(risk.reward_risk, needs, gets))
    if take:
        reason = "TAKE — " + reason
    else:
        reason = "SKIP — " + reason.replace("gives", "gives only")
    if risk.p_target < 0.5:
        # Worth saying plainly: on a 2:1 target the stop is twice as near, so
        # being stopped more often than not is the design, not a warning.
        reason += ". Stop is hit first {0:.0f}% of the time — expected at this "\
                  "reward:risk, and not a reason to skip on its own".format(
                      risk.p_stop * 100)
    if risk.p_unresolved > 0.25:
        reason += ". Only {0:.0f}% resolve inside {1}d".format(
            (1 - risk.p_unresolved) * 100, horizon_days)
    return {"verdict": "TAKE" if take else "SKIP", "reason": reason}


def position_from_idea(idea: Any, spot: Optional[float] = None) -> Optional[Position]:
    """A sized :class:`Idea` from ``todays_trades`` as a priceable Position.

    Closes the loop the desk actually wants: consensus decides *what*, the
    conflict guards decide *whether*, ``size_idea`` decides *how much* — and only
    then can the scale function say whether the target is reachable before the
    stop. Ranking candidates before that step ranks them on agreement alone,
    which is how a 78.8%-hit-rate trade paying 0.32:1 gets to look like the best
    idea on the board.
    """
    pair = str(getattr(idea, "pair", "") or "").strip()
    side = _side(getattr(idea, "direction", None))
    entry = getattr(idea, "entry", None)
    if not pair or not side or not entry:
        return None
    return Position(
        pair=pair, side=side, lots=float(getattr(idea, "lots", 0) or 0),
        entry=float(entry), spot=float(spot or entry),
        stop=float(getattr(idea, "stop", 0) or 0) or None,
        target=float(getattr(idea, "target", 0) or 0) or None,
    )


def screen(positions: Sequence[Position], params_by_pair: Dict[str, GBMParams],
           rates: Optional[Dict[str, float]] = None,
           horizon_days: int = DEFAULT_HORIZON_DAYS,
           min_p_target: float = 0.5, require_positive_edge: bool = True,
           n_sims: int = 50_000) -> List[PositionRisk]:
    """Candidates whose target is reached first often enough to be worth taking.

    Two filters, and they are not the same test:

    * ``min_p_target`` — "target before stop", the question as usually asked.
    * ``require_positive_edge`` — whether that probability beats the win rate the
      payoff demands. A 0.32 reward:risk needs 75.9%; clearing 50% means nothing
      there. Leaving this on is why the screen is worth running at all.

    Ranked by edge, because that is what compounds — not by probability, which
    is what looks impressive.
    """
    out: List[PositionRisk] = []
    for position in positions:
        params = params_by_pair.get(position.pair)
        if params is None:
            continue
        risk = assess(position, params, rates, horizon_days=horizon_days,
                      n_sims=n_sims)
        if risk is None:
            continue
        if risk.p_target < min_p_target:
            continue
        if require_positive_edge and risk.edge_pp <= 0:
            continue
        out.append(risk)
    return sorted(out, key=lambda r: r.edge_pp, reverse=True)


def load_book() -> List[Position]:  # pragma: no cover - I/O seam
    """The stored book. Empty list when nothing is stored or the DB is down."""
    try:
        from src.services.open_positions import load
        return positions_from_store(load() or [])
    except Exception:
        return []


def calibrate(pair: str, periods_per_year: int = PERIODS_PER_YEAR
              ) -> Optional[GBMParams]:  # pragma: no cover - hits the spine
    """Fit GBM to the pair's daily closes, or ``None`` if it cannot be priced."""
    inst = INSTRUMENTS.get(pair)
    if inst is None:
        return None
    try:
        from src.services.market_data import daily_ohlc
        frame = daily_ohlc(inst.ticker)
        if frame is None or frame.empty:
            return None
        return calibrate_gbm(frame["Close"].to_numpy(), periods_per_year)
    except Exception:
        return None


def load_candidates(min_sources: int = 2, risk_pct: float = 1.0
                    ) -> List[Position]:  # pragma: no cover - I/O seam
    """Today's board, conflict-screened against the open book and sized.

    The same chain the Today's Trades page runs, so a candidate reaching the
    scale function has already survived consensus, the correlation guard and the
    currency-exposure guard.
    """
    from src.core.todays_trades import (consensus, exposure_conflict,
                                        find_conflict, size_idea)
    from src.db.market_cache import _resolve_cfg
    from src.db.cache import pooled_repository
    from src.indicators.technical import TechnicalIndicators as TI
    from src.services.account_state import get_balance
    from src.services.market_data import daily_ohlc
    from contextlib import closing

    cfg = _resolve_cfg()
    if cfg is None:
        return []
    repo = pooled_repository(cfg)
    with closing(repo._connect()) as conn, conn, conn.cursor() as cur:  # noqa: SLF001
        # 45 days, not 24 hours: `consensus` holds each signal for its own
        # horizon, so the query only has to be wide enough to contain the
        # longest one. Selecting the timestamp and detail is what makes that
        # possible — without them every row looks equally fresh.
        cur.execute("SELECT instrument, direction, source, logged_at, checks_detail "
                    "FROM trade_setups "
                    "WHERE logged_at > now() - interval '45 days'")
        rows = [{"instrument": r[0], "direction": r[1], "source": r[2],
                 "logged_at": r[3], "checks_detail": r[4]}
                for r in cur.fetchall()]

    stored = load_book()
    book = [{"pair": p.pair, "direction": p.side, "volume": p.lots,
             "price": p.spot} for p in stored]
    balance = float(get_balance() or 0.0)

    out: List[Position] = []
    for idea in consensus(rows):
        if len(idea.agree) < min_sources:
            continue
        if find_conflict(idea.pair, idea.direction, book):
            continue
        if exposure_conflict(idea.pair, idea.direction, book, balance):
            continue
        inst = INSTRUMENTS.get(idea.pair)
        if inst is None:
            continue
        try:
            frame = daily_ohlc(inst.ticker)
            atr = float(TI.atr(frame["High"], frame["Low"], frame["Close"], 14).iloc[-1])
            size_idea(idea, float(frame["Close"].iloc[-1]), atr, balance,
                      risk_pct=risk_pct)
        except Exception:
            continue
        position = position_from_idea(idea)
        if position is not None:
            out.append(position)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    """Price the open book, or today's candidates.

    ``python -m src.services.position_risk [--days N]``
    ``python -m src.services.position_risk --candidates [--any-edge]``
    """
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    days = DEFAULT_HORIZON_DAYS
    if "--days" in argv:
        i = argv.index("--days")
        days = int(argv[i + 1]) if len(argv) > i + 1 else DEFAULT_HORIZON_DAYS

    if "--candidates" in argv:
        candidates = load_candidates()
        if not candidates:
            print("No candidates today (or the DB is unreachable).")
            return 1
        params = {p.pair: calibrate(p.pair) for p in candidates}
        params = {k: v for k, v in params.items() if v is not None}
        rates = usd_rates([{"pair": p.pair, "direction": p.side, "price": p.spot}
                           for p in candidates] +
                          [{"pair": p.pair, "direction": p.side, "price": p.spot}
                           for p in load_book()])
        passed = screen(candidates, params, rates, horizon_days=days,
                        require_positive_edge="--any-edge" not in argv)
        print("%d candidates screened, %d have P(target first) > 50%% %s\n" % (
            len(candidates), len(passed),
            "" if "--any-edge" in argv else "AND positive edge"))
        print("%-9s %-6s %8s %8s %8s %9s %9s %9s %10s %8s" % (
            "pair", "side", "P(TP)", "P(SL)", "R:R", "breakeven", "edge",
            "win", "loss", "EV"))
        for risk in passed:
            print("%-9s %-6s %7.1f%% %7.1f%% %8.2f %8.1f%% %+8.1fpp %+9.2f %+10.2f %+8.2f" % (
                risk.pair, risk.side, risk.p_target * 100, risk.p_stop * 100,
                risk.reward_risk, risk.breakeven * 100, risk.edge_pp,
                risk.win_usd, risk.loss_usd, risk.ev_usd))
        rejected = [p.pair for p in candidates
                    if p.pair not in {r.pair for r in passed}]
        if rejected:
            print("\nfiltered out: %s" % ", ".join(sorted(set(rejected))))
        return 0

    book = load_book()
    if not book:
        print("No stored positions. Run the MT5 sync (deploy/mt5_sync.py).")
        return 1
    rates = usd_rates([{"pair": p.pair, "direction": p.side, "price": p.spot}
                       for p in book])

    print("%-9s %-6s %8s %8s %7s %9s %10s %10s %11s" % (
        "pair", "side", "P(TP)", "R:R", "edge", "breakeven", "win", "loss", "EV"))
    total_ev = total_risk = 0.0
    for position in book:
        params = calibrate(position.pair)
        if params is None:
            print("%-9s %-6s  cannot calibrate" % (position.pair, position.side))
            continue
        risk = assess(position, params, rates, horizon_days=days)
        if risk is None:
            print("%-9s %-6s  unbracketed (stop=%s target=%s) - not priceable" % (
                position.pair, position.side, position.stop, position.target))
            continue
        total_ev += risk.ev_usd
        total_risk += risk.loss_usd
        print("%-9s %-6s %7.1f%% %8.2f %+6.1fpp %8.1f%% %+10.2f %+10.2f %+11.2f" % (
            risk.pair, risk.side, risk.p_target * 100, risk.reward_risk,
            risk.edge_pp, risk.breakeven * 100, risk.win_usd, risk.loss_usd,
            risk.ev_usd))
    print("-" * 88)
    print("%-9s %-6s %8s %8s %7s %9s %10s %+10.2f %+11.2f" % (
        "BOOK", "", "", "", "", "", "", total_risk, total_ev))
    if abs(total_ev) < abs(total_risk) * 0.05:
        print("\nBook EV is ~zero against %.2f of downside: every position can carry a"
              % abs(total_risk))
        print("positive probability edge and still net nothing once reward:risk is paid.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
