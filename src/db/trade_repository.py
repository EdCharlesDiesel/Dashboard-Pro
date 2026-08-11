"""Trade persistence via PostgreSQL.

Wraps psycopg2 connections in a class so callers stop passing around raw cfg
dicts. SQL strings are byte-equivalent to the procedural code they replace.
"""
from __future__ import annotations

import re
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import psycopg2
import psycopg2.extras


@dataclass(frozen=True)
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "dashboardprov1"
    user: str = "postgres"
    password: str = "$ta99Ath0"

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "DBConfig":
        return cls(
            host=str(m.get("host", "localhost")),
            port=int(m.get("port", 5432)),
            dbname=str(m.get("dbname") or m.get("name") or "trading"),
            user=str(m.get("user", "postgres")),
            password=str(m.get("password", "")),
        )

    def as_kwargs(self) -> Dict[str, Any]:
        return {
            "host": self.host, "port": self.port, "dbname": self.dbname,
            "user": self.user, "password": self.password,
        }


# ``trade_setups`` does double duty: it holds trades the user actually took AND
# every auto-saved page SIGNAL (src/services/signal_store.py — ~25 pages plus
# the background worker write those continuously). Anything analysing *trading
# performance* must filter to these sources, or the signal firehose drowns the
# real record. A NULL source is the schema default 'checklist', i.e. a trade.
# Single source of truth: the Trade Journal and the Martingale page both use it.
EXECUTED_SOURCES = ("checklist", "mt4_import", "mt5_sync")


class TradeRepository:
    """All trade_setups CRUD + analytics behind a single class."""

    OUTCOME_COLUMNS = (
        "entry_price FLOAT",
        "outcome     VARCHAR(10)",
        "close_price FLOAT",
        "pips_gained FLOAT",
        "r_multiple  FLOAT",
        "is_open     BOOLEAN DEFAULT TRUE",
        "source      VARCHAR(20) DEFAULT 'checklist'",
        "profit      FLOAT",
        "invalidated_at     TIMESTAMP",
        "invalidation_price FLOAT",
    )

    # Small key/value store for app-level state that must survive restarts and
    # be shared across devices — e.g. the live account balance the Trade Journal
    # imports and the Setup Ranker sizes from. JSONB value keeps it flexible.
    APP_STATE_SQL = """
        CREATE TABLE IF NOT EXISTS app_state (
            key        VARCHAR(64) PRIMARY KEY,
            value      JSONB,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        )
    """

    # Lightweight audit trail for the interactive tool pages (R:R calculator,
    # Account Risk, Correlations, News Filter, Stop Structure) that don't
    # produce a directional trade_setups row. One row per meaningful
    # computation; JSONB payload keeps each tool's shape flexible.
    TOOL_USAGE_SQL = """
        CREATE TABLE IF NOT EXISTS tool_usage_log (
            id        SERIAL PRIMARY KEY,
            logged_at TIMESTAMP NOT NULL DEFAULT NOW(),
            tool      VARCHAR(30) NOT NULL,
            payload   JSONB
        )
    """

    # One row per symbol+timeframe holding the ABR Toolkit's latest read
    # (pages/abr_toolkit_tab.py) — upserted on every rerun rather than
    # appended, since only the current signal per instrument/timeframe matters.
    ABR_SIGNALS_SQL = """
        CREATE TABLE IF NOT EXISTS abr_signals (
            symbol      TEXT NOT NULL,
            timeframe   TEXT NOT NULL,
            ts          TIMESTAMPTZ NOT NULL,
            signal      TEXT,
            entry       DOUBLE PRECISION,
            sl          DOUBLE PRECISION,
            tp3         DOUBLE PRECISION,
            quality     INT,
            grade       TEXT,
            htf_aligned INT,
            PRIMARY KEY (symbol, timeframe)
        )
    """

    # Every journaled GARCH-cone forecast (pages/forecast_tab.py) — appended,
    # not upserted (one row per symbol/horizon/day), so each forecast's
    # eventual realized outcome can be scored independently.
    ABR_FORECASTS_SQL = """
        CREATE TABLE IF NOT EXISTS abr_forecasts (
            id           SERIAL PRIMARY KEY,
            symbol       TEXT NOT NULL,
            made_at      TIMESTAMPTZ NOT NULL,
            horizon_days INT NOT NULL,
            spot         DOUBLE PRECISION,
            score        INT,
            label        TEXT,
            drivers      TEXT,
            lo68 DOUBLE PRECISION, hi68 DOUBLE PRECISION,
            lo95 DOUBLE PRECISION, hi95 DOUBLE PRECISION,
            realized     DOUBLE PRECISION,
            UNIQUE (symbol, horizon_days, made_at)
        )
    """

    # One row per instrument+week holding the Swing Playbook's hand-typed
    # weekly thesis (pages/swing_playbook_tab.py) — upserted so only the
    # current week's thesis per instrument is kept.
    SWING_THESES_SQL = """
        CREATE TABLE IF NOT EXISTS swing_theses (
            id           SERIAL PRIMARY KEY,
            instrument   VARCHAR(20) NOT NULL,
            week_start   DATE NOT NULL,
            bias         VARCHAR(10),
            invalidation TEXT,
            created_at   TIMESTAMP NOT NULL DEFAULT NOW(),
            UNIQUE (instrument, week_start)
        )
    """

    CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS trade_setups (
            id            SERIAL PRIMARY KEY,
            logged_at     TIMESTAMP NOT NULL DEFAULT NOW(),
            instrument    VARCHAR(30),
            ticker        VARCHAR(20),
            direction     VARCHAR(10),
            session       VARCHAR(20),
            score         VARCHAR(20),
            verdict       VARCHAR(20),
            atr14         FLOAT,
            atr20         FLOAT,
            sl_pips       FLOAT,
            tp1_pips      FLOAT,
            tp2_pips      FLOAT,
            lot_size      FLOAT,
            risk_amount   FLOAT,
            rr_tp1        FLOAT,
            rr_tp2        FLOAT,
            account_bal   FLOAT,
            risk_pct      FLOAT,
            checks_passed INT,
            checks_total  INT,
            checks_detail JSONB,
            notes         TEXT
        )
    """

    def __init__(
        self,
        cfg: DBConfig,
        connect_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        """``connect_factory`` overrides how a connection is obtained.

        Defaults to a fresh ``psycopg2.connect`` per call. Inject a factory to
        (a) borrow from a pooled/cached connection (see ``src/db/cache.py``) or
        (b) hand the repository a fake connection in unit tests. The returned
        object must support the psycopg2 connection contract used here:
        ``cursor()``, transaction context (``__enter__``/``__exit__``), and
        ``close()`` (which the pool wrapper repurposes as "return to pool").
        """
        self.cfg = cfg
        self._connect_factory = connect_factory

    # ── connection helper ───────────────────────────────────────────────────
    def _connect(self):
        if self._connect_factory is not None:
            return self._connect_factory()
        return psycopg2.connect(**self.cfg.as_kwargs())

    # Secondary indexes (idempotent, same version-controlled pattern as the
    # ADD COLUMN IF NOT EXISTS migrations). Deliberately minimal — a live
    # pg_stat_user_tables audit showed the only large table (market_bars,
    # ~153k rows) is already served entirely by its composite PK, and every
    # other table is small enough that the planner correctly prefers a seq
    # scan. `trade_setups` is the one table that grows without bound (every
    # trade + every persisted signal from 25+ auto-save sources), and its
    # dominant read pattern across load_setups / load_open / performance_stats
    # / the journal is `ORDER BY logged_at DESC LIMIT n`. A descending index on
    # logged_at serves that ordering directly (no sort) once the table grows
    # past a few thousand rows; at current size the planner may still seq-scan,
    # and that's fine — the index is free insurance, not a fix for a problem
    # that exists today. Do NOT add speculative indexes on the tiny/static
    # tables: an unused index is pure write-time and storage cost.
    INDEX_DDL = (
        "CREATE INDEX IF NOT EXISTS idx_trade_setups_logged_at "
        "ON trade_setups (logged_at DESC)",
    )

    # ── schema ──────────────────────────────────────────────────────────────
    def init_schema(self) -> Tuple[bool, str]:
        """Create the table + outcome columns + indexes. Returns (ok, message)."""
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(self.CREATE_SQL)
                for col_def in self.OUTCOME_COLUMNS:
                    cur.execute(
                        f"ALTER TABLE trade_setups ADD COLUMN IF NOT EXISTS {col_def}"
                    )
                cur.execute(self.APP_STATE_SQL)
                cur.execute(self.TOOL_USAGE_SQL)
                cur.execute(self.ABR_SIGNALS_SQL)
                cur.execute(self.ABR_FORECASTS_SQL)
                cur.execute(self.SWING_THESES_SQL)
                for ddl in self.INDEX_DDL:
                    cur.execute(ddl)
                conn.commit()
            return True, "Connected"
        except Exception as exc:
            return False, str(exc)

    # ── app_state key/value store ───────────────────────────────────────────
    def get_state(self, key: str) -> Optional[Any]:
        """Return the JSON value stored under ``key`` (psycopg2 decodes JSONB to a
        Python object), or ``None`` if the key is absent."""
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM app_state WHERE key = %s", (key,))
            row = cur.fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: Any) -> None:
        """Upsert a JSON ``value`` under ``key`` (last write wins)."""
        sql = """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = NOW()
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, (key, psycopg2.extras.Json(value)))
            conn.commit()

    # ── tool usage log ─────────────────────────────────────────────────────
    def log_tool_usage(self, tool: str, payload: Dict[str, Any]) -> None:
        """Insert one tool-interaction row. ``tool`` names the page (e.g.
        ``'rr_calculator'``); ``payload`` is whatever inputs/outputs that tool
        wants recorded, stored as-is in JSONB."""
        sql = "INSERT INTO tool_usage_log (tool, payload) VALUES (%s, %s)"
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, (tool, psycopg2.extras.Json(payload)))
            conn.commit()

    # ── ABR Toolkit signals ─────────────────────────────────────────────────
    def save_abr_signal(
        self,
        symbol: str,
        timeframe: str,
        signal: str,
        entry: Optional[float],
        sl: Optional[float],
        tp3: Optional[float],
        quality: int,
        grade: str,
        htf_aligned: int,
    ) -> None:
        """Upsert the ABR Toolkit's latest read for one symbol+timeframe."""
        sql = """
            INSERT INTO abr_signals
                (symbol, timeframe, ts, signal, entry, sl, tp3, quality, grade, htf_aligned)
            VALUES
                (%(symbol)s, %(timeframe)s, NOW(), %(signal)s, %(entry)s, %(sl)s, %(tp3)s,
                 %(quality)s, %(grade)s, %(htf_aligned)s)
            ON CONFLICT (symbol, timeframe) DO UPDATE SET
                ts = EXCLUDED.ts, signal = EXCLUDED.signal, entry = EXCLUDED.entry,
                sl = EXCLUDED.sl, tp3 = EXCLUDED.tp3, quality = EXCLUDED.quality,
                grade = EXCLUDED.grade, htf_aligned = EXCLUDED.htf_aligned
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, {
                "symbol": symbol, "timeframe": timeframe, "signal": signal,
                "entry": entry, "sl": sl, "tp3": tp3,
                "quality": quality, "grade": grade, "htf_aligned": htf_aligned,
            })
            conn.commit()

    def latest_abr_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Most recent ABR Toolkit read for ``symbol`` (any timeframe), or
        ``None`` (used by pages/forecast_tab.py as a driver input)."""
        sql = """
            SELECT signal, grade, timeframe FROM abr_signals
            WHERE symbol = %s ORDER BY ts DESC LIMIT 1
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, (symbol,))
            row = cur.fetchone()
        return dict(row) if row else None

    # ── ABR Toolkit forecasts (pages/forecast_tab.py) ───────────────────────
    def save_forecast(
        self,
        symbol: str,
        horizon_days: int,
        spot: float,
        score: int,
        label: str,
        drivers_json: str,
        lo68: float,
        hi68: float,
        lo95: float,
        hi95: float,
    ) -> bool:
        """Insert one forecast row, at most once per symbol/horizon/day.
        Returns ``True`` if inserted, ``False`` if today's forecast already
        exists for this symbol+horizon."""
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM abr_forecasts WHERE symbol = %s AND horizon_days = %s "
                "AND made_at::date = CURRENT_DATE",
                (symbol, horizon_days),
            )
            if cur.fetchone():
                return False
            cur.execute("""
                INSERT INTO abr_forecasts
                    (symbol, made_at, horizon_days, spot, score, label, drivers,
                     lo68, hi68, lo95, hi95)
                VALUES
                    (%(symbol)s, NOW(), %(horizon_days)s, %(spot)s, %(score)s,
                     %(label)s, %(drivers)s, %(lo68)s, %(hi68)s, %(lo95)s, %(hi95)s)
            """, {
                "symbol": symbol, "horizon_days": horizon_days, "spot": spot,
                "score": score, "label": label, "drivers": drivers_json,
                "lo68": lo68, "hi68": hi68, "lo95": lo95, "hi95": hi95,
            })
            conn.commit()
        return True

    def load_forecasts(self, symbol: str, limit: int = 40) -> List[Dict[str, Any]]:
        """Recent journaled forecasts for ``symbol``, newest first."""
        sql = """
            SELECT id, made_at, horizon_days, spot, score, label,
                   lo68, hi68, lo95, hi95, realized
            FROM abr_forecasts WHERE symbol = %s ORDER BY made_at DESC LIMIT %s
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, (symbol, limit))
            return [dict(r) for r in cur.fetchall()]

    def update_forecast_realized(self, forecast_id: int, realized: float) -> None:
        """Fill in a matured forecast's realized price (see evaluate_forecasts
        in pages/forecast_tab.py, which computes it from later price history)."""
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE abr_forecasts SET realized = %s WHERE id = %s",
                (realized, forecast_id),
            )
            conn.commit()

    # ── Swing Playbook weekly theses ────────────────────────────────────────
    def save_swing_thesis(
        self, instrument: str, week_start: Any, bias: str, invalidation: str
    ) -> None:
        """Upsert the weekly thesis for ``instrument`` — at most one per
        instrument+week (see ``pages/swing_playbook_tab.py``)."""
        sql = """
            INSERT INTO swing_theses (instrument, week_start, bias, invalidation)
            VALUES (%(instrument)s, %(week_start)s, %(bias)s, %(invalidation)s)
            ON CONFLICT (instrument, week_start) DO UPDATE SET
                bias = EXCLUDED.bias, invalidation = EXCLUDED.invalidation
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, {
                "instrument": instrument, "week_start": week_start,
                "bias": bias, "invalidation": invalidation,
            })
            conn.commit()

    def load_swing_thesis(self, instrument: str, week_start: Any) -> Optional[Dict[str, Any]]:
        """This week's thesis for ``instrument``, or ``None`` if not yet set."""
        sql = """
            SELECT bias, invalidation FROM swing_theses
            WHERE instrument = %s AND week_start = %s
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, (instrument, week_start))
            row = cur.fetchone()
        return dict(row) if row else None

    # ── writes ──────────────────────────────────────────────────────────────
    def save_setup(self, row: Dict[str, Any], source: Optional[str] = None) -> None:
        """Insert one setup row. ``source`` tags where the row came from (e.g.
        ``"market_overview"`` for auto-saved scanner signals); when omitted the
        ``source`` column keeps its ``'checklist'`` default."""
        sql = """
            INSERT INTO trade_setups (
                logged_at, instrument, ticker, direction, session,
                score, verdict, atr14, atr20, entry_price, sl_pips, tp1_pips, tp2_pips,
                lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct,
                checks_passed, checks_total, checks_detail, notes
            ) VALUES (
                %(logged_at)s, %(instrument)s, %(ticker)s, %(direction)s, %(session)s,
                %(score)s, %(verdict)s, %(atr14)s, %(atr20)s, %(entry_price)s, %(sl_pips)s, %(tp1_pips)s, %(tp2_pips)s,
                %(lot_size)s, %(risk_amount)s, %(rr_tp1)s, %(rr_tp2)s, %(account_bal)s, %(risk_pct)s,
                %(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s
            )
        """
        # entry_price is NOT optional decoration: `source_scorecard.evaluate_row`
        # needs the price a signal was proposed at to measure subsequent bars
        # against, and without it every row is permanently unresolvable.
        # `signal_to_setup_row` had been producing the value all along; this
        # INSERT silently dropped it, so 724 of 724 stored signals carried NULL
        # and only the 12 whose idea leaked `entry` into checks_detail could ever
        # be scored. A row builder test cannot catch that — see
        # tests/test_trade_repository_columns.py, which asserts the INSERT and
        # the builder agree.
        params: Dict[str, Any] = row
        if source is not None:
            # Add the source column without mutating the caller's row dict.
            sql = sql.replace(
                "checks_passed, checks_total, checks_detail, notes\n",
                "checks_passed, checks_total, checks_detail, notes, source\n",
            ).replace(
                "%(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s\n",
                "%(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s, %(source)s\n",
            )
            params = {**row, "source": source}
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()

    def imported_tickets(self) -> set:
        """MT4 tickets already imported (parsed from the notes tag), for dedupe."""
        sql = "SELECT notes FROM trade_setups WHERE source = 'mt4_import' AND notes IS NOT NULL"
        out = set()
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(sql)
                for (notes,) in cur.fetchall():
                    m = re.search(r"MT4 #(\d+)", notes or "")
                    if m:
                        out.add(int(m.group(1)))
        except Exception:
            pass
        return out

    def import_mt4_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk-insert mapped MT4 rows as closed, source='mt4_import'. Returns
        the number inserted. Caller is responsible for dedupe."""
        if not rows:
            return 0
        sql = """
            INSERT INTO trade_setups (
                logged_at, instrument, ticker, direction, session, lot_size,
                entry_price, close_price, outcome, pips_gained, r_multiple,
                sl_pips, is_open, source, notes, profit
            ) VALUES (
                %(logged_at)s, %(instrument)s, %(ticker)s, %(direction)s, %(session)s, %(lot_size)s,
                %(entry_price)s, %(close_price)s, %(outcome)s, %(pips_gained)s, %(r_multiple)s,
                %(sl_pips)s, FALSE, 'mt4_import', %(notes)s, %(profit)s
            )
        """
        payload = [{k: r.get(k) for k in (
            "logged_at", "instrument", "ticker", "direction", "session", "lot_size",
            "entry_price", "close_price", "outcome", "pips_gained", "r_multiple",
            "sl_pips", "notes", "profit")} for r in rows]
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, payload)
            conn.commit()
        return len(payload)

    def delete_setup(self, trade_id: int) -> None:
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute("DELETE FROM trade_setups WHERE id = %s", (trade_id,))
            conn.commit()

    def close_trade(
        self,
        trade_id: int,
        entry_price: float,
        close_price: float,
        pips_gained: float,
        r_multiple: float,
        outcome: str,
    ) -> None:
        sql = """
            UPDATE trade_setups
            SET entry_price = %s, close_price = %s, pips_gained = %s,
                r_multiple = %s, outcome = %s, is_open = FALSE
            WHERE id = %s
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (entry_price, close_price, pips_gained, r_multiple, outcome, trade_id),
            )
            conn.commit()

    def mark_invalidated(self, trade_id: int, price: float) -> None:
        """Flag a signal as price-invalidated (stop level breached before or
        instead of being taken). A visibility badge only — never touches
        ``outcome``/``close_price``/``is_open``, which stay driven by a real
        close (MT4 import or the Checklist's close-trade form). Idempotent:
        a row already marked keeps its original ``invalidated_at``."""
        sql = """
            UPDATE trade_setups
            SET invalidated_at = NOW(), invalidation_price = %s
            WHERE id = %s AND invalidated_at IS NULL
        """
        with closing(self._connect()) as conn, conn, conn.cursor() as cur:
            cur.execute(sql, (price, trade_id))
            conn.commit()

    # ── reads ───────────────────────────────────────────────────────────────
    def load_setups(self, limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, logged_at, instrument, ticker,
                   direction, session, score, verdict, atr14, atr20,
                   sl_pips, tp1_pips, tp2_pips, lot_size, risk_amount,
                   rr_tp1, rr_tp2, account_bal, risk_pct, checks_passed,
                   checks_total, checks_detail, notes,
                   entry_price, close_price, outcome, pips_gained,
                   r_multiple, is_open, source, profit
            FROM trade_setups
            ORDER BY logged_at DESC
            LIMIT %s
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def load_open(self) -> List[Dict[str, Any]]:
        sql = """
            SELECT id, logged_at, instrument, direction, sl_pips, tp1_pips,
                   tp2_pips, lot_size
            FROM trade_setups
            WHERE is_open IS TRUE OR is_open IS NULL
            ORDER BY logged_at DESC LIMIT 20
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    def daily_losses(self, max_losses: int = 2) -> Dict[str, Any]:
        sql = """
            SELECT COUNT(*) AS cnt
            FROM trade_setups
            WHERE outcome = 'LOSS' AND is_open = FALSE
              AND DATE(logged_at) = CURRENT_DATE
        """
        try:
            with closing(self._connect()) as conn, conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(sql)
                row = cur.fetchone()
            count = int(row["cnt"]) if row else 0
        except Exception:
            count = 0
        return {
            "losses_today": count,
            "limit": max_losses,
            "blocked": count >= max_losses,
        }

    def performance_stats(self, n: int = 20) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT outcome, r_multiple, pips_gained
            FROM trade_setups
            WHERE is_open = FALSE AND outcome IS NOT NULL
            ORDER BY logged_at DESC LIMIT %s
        """
        with closing(self._connect()) as conn, conn, conn.cursor(
            cursor_factory=psycopg2.extras.RealDictCursor
        ) as cur:
            cur.execute(sql, (n,))
            rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return None
        wins = [r for r in rows if r["outcome"] == "WIN"]
        losses = [r for r in rows if r["outcome"] == "LOSS"]
        be_trades = [r for r in rows if r["outcome"] == "BE"]
        total = len(rows)
        win_rate = len(wins) / total * 100 if total else 0
        win_rs = [r["r_multiple"] for r in wins if r["r_multiple"]]
        loss_rs = [abs(r["r_multiple"]) for r in losses if r["r_multiple"]]
        avg_win_r = sum(win_rs) / len(win_rs) if win_rs else 0
        avg_loss_r = sum(loss_rs) / len(loss_rs) if loss_rs else 1.0
        loss_rate = len(losses) / total if total else 0
        expectancy = (win_rate / 100 * avg_win_r) - (loss_rate * avg_loss_r)
        pf = (
            (len(wins) * avg_win_r) / (len(losses) * avg_loss_r)
            if losses and avg_loss_r else 0.0
        )
        return {
            "total": total, "wins": len(wins), "losses": len(losses),
            "be": len(be_trades), "win_rate": win_rate,
            "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
            "expectancy": expectancy, "profit_factor": pf,
        }

    def realized_pnl(self, limit: int = 100_000) -> Dict[str, Any]:
        """Realised account-currency P/L summed across all closed trades.

        Per-trade P/L uses the broker's stored "profit`` when available (most
        exact — it includes swap/commission); otherwise it derives the figure
        from ``pips_gained × pip_value × lot_size``, which matches the app's risk
        model (``risk_amount = lot × sl_pips × pip_value``). Trades missing the
        data needed for either path are skipped and counted separately.

        Returns ``{pnl, trades, counted, skipped}``.
        """
        from src.instruments.registry import INSTRUMENTS

        rows = self.load_setups(limit=limit)
        pnl = 0.0
        counted = skipped = 0
        for r in rows:
            if r.get("is_open", True) is True or r.get("outcome") is None:
                continue  # only realised, closed trades
            profit = r.get("profit")
            if profit is not None:
                pnl += float(profit)
                counted += 1
                continue
            pips = r.get("pips_gained")
            lot = r.get("lot_size")
            inst = INSTRUMENTS.get(r.get("instrument") or "")
            if pips is None or lot is None or inst is None:
                skipped += 1
                continue
            pnl += float(pips) * float(inst.pip) * float(lot)
            counted += 1
        return {"pnl": round(pnl, 2), "trades": counted + skipped,
                "counted": counted, "skipped": skipped}
