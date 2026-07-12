"""
evening_sentry.py  (v2)
=======================
"The dashboard runs the evening shift."

Watches your watchlist during the 17:00-20:00 SAST study window, runs
your scoring logic, journals everything to PostgreSQL, and pings
Telegram ONLY on state changes. No auto-execution.

v2 additions (the "what's missing" fixes):
  [1] Heartbeat        — pings HEALTHCHECK_URL after every successful
                         scan; an external monitor (healthchecks.io)
                         alerts YOU if the sentry itself dies.
  [2] Staleness guard  — data older than MAX_BAR_AGE_DAYS scores as
                         STALE_DATA; excluded states never trigger
                         alerts, so a feed hiccup can't fake a signal.
  [3] News tagging     — populate sentry_news each morning; alerts
                         within +/-60 min of a high-impact event for a
                         related currency are tagged "NEWS WINDOW".
  [4] Decay tracking   — signal price is journaled; the digest reports
                         how far price moved between alert and review,
                         so you can measure what the class-time delay
                         actually costs you.
  [5] Gap awareness    — every scan run is logged; the digest reports
                         runs vs expected, so missed scans are visible
                         instead of silently absent.

Schedule (Africa/Johannesburg, weekdays):
  16:45  pre-shift briefing        20:15  end-of-shift digest
  17:00-19:30 scan every 30 min (6 expected runs)

Env vars:
  DATABASE_URL         postgresql+psycopg2://... (sqlite fallback)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
  HEALTHCHECK_URL      e.g. https://hc-ping.com/<uuid>   (optional)

Usage:
  python evening_sentry.py --once | --brief | --digest | --run
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

import requests
import sqlalchemy as sa

log = logging.getLogger("evening_sentry")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

TZ_NAME = "Africa/Johannesburg"
WATCHLIST = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDZAR=X", "GC=F"]
EXPECTED_SCANS_PER_DAY = 6          # 17:00,17:30,...,19:30
MAX_BAR_AGE_DAYS = 4                # daily bars older than this = stale
NEWS_WINDOW_MIN = 60                # tag alerts within +/- this many min
NON_ALERTING_STATES = {"NO_DATA", "STALE_DATA"}

SYMBOL_CURRENCIES = {
    "EURUSD=X": {"EUR", "USD"}, "GBPUSD=X": {"GBP", "USD"},
    "USDJPY=X": {"USD", "JPY"}, "USDZAR=X": {"USD", "ZAR"},
    "GC=F": {"USD"},
}


# ----------------------------------------------------------------------
# Result model
# ----------------------------------------------------------------------

@dataclass
class SignalResult:
    symbol: str
    state: str                  # LONG_SETUP / SHORT_SETUP / NO_TRADE / ...
    score: float
    detail: str
    price: float | None = None  # price at evaluation time (for decay calc)


class SignalProvider(Protocol):
    def evaluate(self, symbol: str) -> SignalResult: ...


# ----------------------------------------------------------------------
# Demo provider — swap for confluence_engine
# ----------------------------------------------------------------------

class DemoBreakoutProvider:
    """20-day breakout with staleness validation. Replace evaluate()
    internals with your confluence_engine call; keep the staleness
    check pattern."""

    max_age_days = MAX_BAR_AGE_DAYS

    def evaluate(self, symbol: str) -> SignalResult:
        import pandas as pd
        import yfinance as yf
        df = yf.download(symbol, period="60d", interval="1d",
                         auto_adjust=False, progress=False)
        if df.empty or len(df) < 25:
            return SignalResult(symbol, "NO_DATA", 0.0, "insufficient data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # [2] staleness guard
        last_ts = pd.Timestamp(df.index[-1]).tz_localize(None)
        age = pd.Timestamp.utcnow().tz_localize(None) - last_ts
        if age > pd.Timedelta(days=self.max_age_days):
            return SignalResult(symbol, "STALE_DATA", 0.0,
                                f"last bar {last_ts.date()} is {age.days}d old")

        close = float(df["Close"].iloc[-1])
        hi20 = float(df["High"].iloc[-21:-1].max())
        lo20 = float(df["Low"].iloc[-21:-1].min())
        if close > hi20:
            return SignalResult(symbol, "LONG_SETUP", 1.0,
                                f"close {close:.5g} > 20d high {hi20:.5g}", close)
        if close < lo20:
            return SignalResult(symbol, "SHORT_SETUP", 1.0,
                                f"close {close:.5g} < 20d low {lo20:.5g}", close)
        return SignalResult(symbol, "NO_TRADE", 0.0,
                            f"inside [{lo20:.5g}, {hi20:.5g}]", close)


# ----------------------------------------------------------------------
# Storage
# ----------------------------------------------------------------------

class Store:
    def __init__(self, url: str | None = None):
        self.engine = sa.create_engine(
            url or os.environ.get("DATABASE_URL", "sqlite:///sentry.db"))
        md = sa.MetaData()
        self.journal = sa.Table(
            "sentry_journal", md,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("ts_utc", sa.DateTime, nullable=False),
            sa.Column("symbol", sa.String(20), nullable=False),
            sa.Column("state", sa.String(30), nullable=False),
            sa.Column("score", sa.Float),
            sa.Column("detail", sa.Text),
            sa.Column("price", sa.Float),
            sa.Column("alerted", sa.Boolean, default=False),
            sa.Column("news_tagged", sa.Boolean, default=False),
        )
        self.runs = sa.Table(                       # [5]
            "sentry_runs", md,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("ts_utc", sa.DateTime, nullable=False),
            sa.Column("job", sa.String(20), nullable=False),
            sa.Column("symbols_ok", sa.Integer),
            sa.Column("errors", sa.Integer),
        )
        self.news = sa.Table(                       # [3]
            "sentry_news", md,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("ts_utc", sa.DateTime, nullable=False),
            sa.Column("currency", sa.String(6), nullable=False),
            sa.Column("impact", sa.String(10), default="high"),
            sa.Column("title", sa.Text),
        )
        md.create_all(self.engine)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def last_state(self, symbol: str) -> str | None:
        q = (sa.select(self.journal.c.state)
             .where(self.journal.c.symbol == symbol)
             .order_by(self.journal.c.id.desc()).limit(1))
        with self.engine.connect() as cx:
            row = cx.execute(q).fetchone()
        return row[0] if row else None

    def record(self, r: SignalResult, alerted: bool,
               news_tagged: bool = False) -> None:
        with self.engine.begin() as cx:
            cx.execute(self.journal.insert().values(
                ts_utc=self._utcnow(), symbol=r.symbol, state=r.state,
                score=r.score, detail=r.detail, price=r.price,
                alerted=alerted, news_tagged=news_tagged))

    def record_run(self, job: str, symbols_ok: int, errors: int) -> None:
        with self.engine.begin() as cx:
            cx.execute(self.runs.insert().values(
                ts_utc=self._utcnow(), job=job,
                symbols_ok=symbols_ok, errors=errors))

    def runs_today(self, job: str = "scan") -> int:
        today = self._utcnow().date()
        q = (sa.select(sa.func.count())
             .select_from(self.runs)
             .where(self.runs.c.job == job)
             .where(sa.func.date(self.runs.c.ts_utc) == str(today)))
        with self.engine.connect() as cx:
            return int(cx.execute(q).scalar() or 0)

    def today_rows(self) -> list:
        today = self._utcnow().date()
        q = (sa.select(self.journal)
             .where(sa.func.date(self.journal.c.ts_utc) == str(today))
             .order_by(self.journal.c.id))
        with self.engine.connect() as cx:
            return list(cx.execute(q).fetchall())

    def news_near(self, currencies: set[str], when: datetime,
                  window_min: int = NEWS_WINDOW_MIN) -> list:
        lo = when - timedelta(minutes=window_min)
        hi = when + timedelta(minutes=window_min)
        q = (sa.select(self.news)
             .where(self.news.c.ts_utc.between(lo, hi))
             .where(self.news.c.currency.in_(list(currencies))))
        with self.engine.connect() as cx:
            return list(cx.execute(q).fetchall())


# ----------------------------------------------------------------------
# Notifiers
# ----------------------------------------------------------------------

class Telegram:
    def __init__(self):
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat = os.environ.get("TELEGRAM_CHAT_ID")

    def send(self, text: str) -> None:
        if not (self.token and self.chat):
            log.info("[telegram disabled] %s", text.replace("\n", " | "))
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat, "text": text}, timeout=10,
            ).raise_for_status()
        except Exception as e:
            log.error("telegram send failed: %s", e)


class Heartbeat:                                    # [1]
    def __init__(self):
        self.url = os.environ.get("HEALTHCHECK_URL")

    def ping(self, ok: bool = True) -> None:
        if not self.url:
            return
        try:
            url = self.url if ok else self.url.rstrip("/") + "/fail"
            requests.get(url, timeout=10)
        except Exception as e:
            log.error("heartbeat ping failed: %s", e)


# ----------------------------------------------------------------------
# Sentry
# ----------------------------------------------------------------------

class EveningSentry:
    def __init__(self, provider: SignalProvider | None = None,
                 store: Store | None = None,
                 notifier: Telegram | None = None,
                 heartbeat: Heartbeat | None = None,
                 watchlist: list[str] | None = None):
        self.provider = provider or DemoBreakoutProvider()
        self.store = store or Store()
        self.tg = notifier or Telegram()
        self.hb = heartbeat or Heartbeat()
        self.watchlist = watchlist or WATCHLIST

    # ---- helpers ----
    def _news_tag(self, symbol: str) -> bool:
        ccys = SYMBOL_CURRENCIES.get(symbol, {"USD"})
        return bool(self.store.news_near(ccys, Store._utcnow()))

    # ---- jobs ----
    def scan(self) -> list[SignalResult]:
        results, changed, errors = [], [], 0
        for sym in self.watchlist:
            try:
                r = self.provider.evaluate(sym)
            except Exception as e:
                errors += 1
                log.error("evaluate(%s) failed: %s", sym, e)
                continue
            prev = self.store.last_state(sym)
            is_change = (
                prev is not None
                and prev != r.state
                and r.state not in NON_ALERTING_STATES
                and prev not in NON_ALERTING_STATES      # data recovery != signal
            )
            tagged = self._news_tag(sym) if is_change else False
            self.store.record(r, alerted=is_change, news_tagged=tagged)
            results.append(r)
            if is_change:
                changed.append((prev, r, tagged))

        if changed:
            lines = []
            for p, r, tagged in changed:
                warn = "  \u26A0 NEWS WINDOW" if tagged else ""
                lines.append(f"\u26A1 {r.symbol}: {p} \u2192 {r.state}{warn}\n"
                             f"   {r.detail}")
            self.tg.send("STATE CHANGES\n" + "\n".join(lines))

        self.store.record_run("scan", symbols_ok=len(results), errors=errors)
        self.hb.ping(ok=(errors == 0))               # [1]
        log.info("scan done: %d ok, %d errors, %d changes",
                 len(results), errors, len(changed))
        return results

    def briefing(self) -> None:
        lines = []
        for sym in self.watchlist:
            try:
                r = self.provider.evaluate(sym)
                lines.append(f"{r.symbol}: {r.state} ({r.detail})")
            except Exception as e:
                lines.append(f"{sym}: EVAL ERROR ({e})")
        self.store.record_run("brief", symbols_ok=len(lines), errors=0)
        self.tg.send("\U0001F4CB PRE-SHIFT BRIEFING 16:45\n" + "\n".join(lines)
                     + "\n\nSentry has the evening. Go study.")

    def digest(self) -> None:
        rows = self.store.today_rows()
        alerts = [r for r in rows if r.alerted]
        runs = self.store.runs_today("scan")

        last_state: dict[str, str] = {}
        for r in rows:
            last_state[r.symbol] = r.state
        lines = [f"{s}: {st}" for s, st in sorted(last_state.items())]

        # [5] gap awareness
        gap_note = ""
        if runs < EXPECTED_SCANS_PER_DAY:
            gap_note = (f"\n\u26A0 Only {runs}/{EXPECTED_SCANS_PER_DAY} scans "
                        f"ran today — check the sentry host.")

        # [4] decay: how far did price move since each alert fired?
        decay_lines = []
        for a in alerts:
            if a.price is None:
                continue
            try:
                now_r = self.provider.evaluate(a.symbol)
                if now_r.price:
                    move = (now_r.price / a.price - 1) * 1e4
                    decay_lines.append(
                        f"- {a.symbol} {a.state} @ {a.price:.5g} \u2192 "
                        f"now {now_r.price:.5g} ({move:+.1f} bps since alert)")
            except Exception as e:
                log.error("decay check(%s): %s", a.symbol, e)

        msg = (f"\U0001F319 END-OF-SHIFT DIGEST 20:15\n"
               f"Scans: {runs}/{EXPECTED_SCANS_PER_DAY} | "
               f"Journaled: {len(rows)} | Changes: {len(alerts)}{gap_note}\n"
               + "\n".join(lines))
        if alerts:
            msg += "\n\nChanges today:\n" + "\n".join(
                f"- {r.symbol} \u2192 {r.state}"
                + (" \u26A0news" if r.news_tagged else "")
                + f" ({r.detail})" for r in alerts)
        if decay_lines:
            msg += "\n\nMove since alert (your review lag):\n" + "\n".join(decay_lines)

        self.store.record_run("digest", symbols_ok=0, errors=0)
        self.tg.send(msg)
        self.hb.ping(ok=True)

    # ---- scheduler ----
    def run(self) -> None:
        from apscheduler.schedulers.blocking import BlockingScheduler
        sched = BlockingScheduler(timezone=TZ_NAME)
        sched.add_job(self.briefing, "cron", hour=16, minute=45,
                      day_of_week="mon-fri", id="briefing",
                      misfire_grace_time=300)
        sched.add_job(self.scan, "cron", hour="17-19", minute="0,30",
                      day_of_week="mon-fri", id="scan",
                      misfire_grace_time=300)
        sched.add_job(self.digest, "cron", hour=20, minute=15,
                      day_of_week="mon-fri", id="digest",
                      misfire_grace_time=600)
        log.info("Evening Sentry scheduled (%s). Ctrl+C to stop.", TZ_NAME)
        sched.start()


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="single scan now")
    g.add_argument("--brief", action="store_true", help="send briefing now")
    g.add_argument("--digest", action="store_true", help="send digest now")
    g.add_argument("--run", action="store_true", help="start scheduler")
    args = ap.parse_args()

    sentry = EveningSentry()
    if args.once:
        for r in sentry.scan():
            print(f"{r.symbol:12s} {r.state:12s} {r.detail}")
    elif args.brief:
        sentry.briefing()
    elif args.digest:
        sentry.digest()
    else:
        sentry.run()


if __name__ == "__main__":
    main()