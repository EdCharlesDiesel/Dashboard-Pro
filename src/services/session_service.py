"""Trading-session detection — Kill Zone classifier."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class SessionStatus:
    window: str
    prime: bool
    color: str
    icon: str
    desc: str
    time: str

    def as_dict(self) -> dict:
        return {
            "window": self.window, "prime": self.prime, "color": self.color,
            "icon": self.icon, "desc": self.desc, "time": self.time,
        }


class SessionService:
    """Classifies the current UTC hour into a market window.

    Hours and windows match the original `get_session_status()` in
    daily-trading-checklist.py exactly.
    """

    LONDON_KZ = (7.0, 9.0)
    NY_KZ = (12.0, 14.0)
    LONDON_CLOSE = (15.0, 17.0)
    TOKYO = (0.0, 3.0)
    NEXT_PRIME = ((7.0, "London KZ"), (12.0, "NY KZ"),
                  (15.0, "London Close"), (24.0, "London KZ"))

    @classmethod
    def current(cls, now: "datetime | None" = None) -> dict:
        now = now or datetime.now(timezone.utc)
        h = now.hour + now.minute / 60
        t = now.strftime("%H:%M UTC")
        if cls.LONDON_KZ[0] <= h < cls.LONDON_KZ[1]:
            return SessionStatus(
                "London Kill Zone", True, "#3fb950", "🟢",
                "07:00–09:00 UTC — highest order flow", t,
            ).as_dict()
        if cls.NY_KZ[0] <= h < cls.NY_KZ[1]:
            return SessionStatus(
                "NY Kill Zone", True, "#3fb950", "🟢",
                "12:00–14:00 UTC — NY open sweep", t,
            ).as_dict()
        if cls.LONDON_CLOSE[0] <= h < cls.LONDON_CLOSE[1]:
            return SessionStatus(
                "London Close", False, "#e3b341", "🟡",
                "15:00–17:00 UTC — trend continuation", t,
            ).as_dict()
        if cls.TOKYO[0] <= h < cls.TOKYO[1]:
            return SessionStatus(
                "Tokyo Session", False, "#f85149", "🔴",
                "00:00–03:00 UTC — low probability, avoid", t,
            ).as_dict()
        next_name, wait_mins = "London KZ", 0
        for start, name in cls.NEXT_PRIME:
            if h < start:
                wait_mins = int((start - h) * 60)
                next_name = name
                break
        return SessionStatus(
            "Dead Zone", False, "#484f58", "⚫",
            f"Between sessions — {wait_mins}m until {next_name}", t,
        ).as_dict()
