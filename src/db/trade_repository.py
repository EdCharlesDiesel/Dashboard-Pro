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

    # ── schema ──────────────────────────────────────────────────────────────
    def init_schema(self) -> Tuple[bool, str]:
        """Create the table + outcome columns. Returns (ok, message)."""
        try:
            with closing(self._connect()) as conn, conn, conn.cursor() as cur:
                cur.execute(self.CREATE_SQL)
                for col_def in self.OUTCOME_COLUMNS:
                    cur.execute(
                        f"ALTER TABLE trade_setups ADD COLUMN IF NOT EXISTS {col_def}"
                    )
                cur.execute(self.APP_STATE_SQL)
                cur.execute(self.TOOL_USAGE_SQL)
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

    # ── writes ──────────────────────────────────────────────────────────────
    def save_setup(self, row: Dict[str, Any], source: Optional[str] = None) -> None:
        """Insert one setup row. ``source`` tags where the row came from (e.g.
        ``"market_overview"`` for auto-saved scanner signals); when omitted the
        ``source`` column keeps its ``'checklist'`` default."""
        sql = """
            INSERT INTO trade_setups (
                logged_at, instrument, ticker, direction, session,
                score, verdict, atr14, atr20, sl_pips, tp1_pips, tp2_pips,
                lot_size, risk_amount, rr_tp1, rr_tp2, account_bal, risk_pct,
                checks_passed, checks_total, checks_detail, notes
            ) VALUES (
                %(logged_at)s, %(instrument)s, %(ticker)s, %(direction)s, %(session)s,
                %(score)s, %(verdict)s, %(atr14)s, %(atr20)s, %(sl_pips)s, %(tp1_pips)s, %(tp2_pips)s,
                %(lot_size)s, %(risk_amount)s, %(rr_tp1)s, %(rr_tp2)s, %(account_bal)s, %(risk_pct)s,
                %(checks_passed)s, %(checks_total)s, %(checks_detail)s, %(notes)s
            )
        """
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
