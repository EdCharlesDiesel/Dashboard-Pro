"""Shared email-alert plumbing for the scanner pages.

Both the Market Overview and Setup Ranker want the same thing: send an HTML
email when a fresh high-conviction signal appears, and never send the same
signal twice. That logic lives here so each page only has to describe *what* a
signal is, not *how* to mail it or remember it.

- Credentials come from `src.core.secrets.email_config()` (Gmail SMTP via an app
  password), which itself falls back to `GMAIL_*` env vars — so it works in the
  Docker image with no secrets file mounted.
- `NotifyCache` persists the set of already-alerted keys to a small JSON file so
  dedupe survives Streamlit reruns and process restarts. Each page passes its own
  namespace so the two scanners never clash.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import tempfile
import threading
from email.message import EmailMessage
from typing import Iterable, List, Optional, Tuple

from src.core import secrets

logger = logging.getLogger("ForexDashboard")


# ══════════════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════════════

def email_configured() -> bool:
    """True when enough Gmail config is present to actually send."""
    c = secrets.email_config()
    return bool(c.get("user") and c.get("password") and c.get("recipient"))


def email_recipient() -> str:
    return secrets.email_config().get("recipient", "")


def send_email(subject: str, html_body: str,
               plain_body: Optional[str] = None) -> Tuple[bool, str]:
    """Send one multipart (plain + HTML) email. Returns (ok, status_message) so
    the caller can surface the result in the UI instead of only logging it."""
    c = secrets.email_config()
    if not (c.get("user") and c.get("password") and c.get("recipient")):
        return False, "Email not configured — set [gmail] in secrets.toml or GMAIL_* env vars."

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = c.get("sender") or c["user"]
        msg["To"] = c["recipient"]
        msg.set_content(plain_body or "This alert is best viewed as HTML.")
        msg.add_alternative(html_body, subtype="html")

        ctx = ssl.create_default_context()
        with smtplib.SMTP(c["smtp_host"], int(c["smtp_port"]), timeout=20) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(c["user"], c["password"])
            server.send_message(msg)

        logger.info("Alert email sent to %s", c["recipient"])
        return True, f"Sent to {c['recipient']}"
    except smtplib.SMTPAuthenticationError:
        return False, "Auth failed — check the Gmail address / 16-char app password in secrets.toml."
    except Exception as exc:  # noqa: BLE001 — report any SMTP/network failure to the UI
        logger.error("Email send failed: %s", exc)
        return False, f"Send failed: {exc}"


# ══════════════════════════════════════════════════════════════════
# DEDUPE
# ══════════════════════════════════════════════════════════════════

# One lock per cache file, shared across every NotifyCache instance that points
# at it. Streamlit serves each browser session on its own thread inside a single
# process, so two sessions (or the alert + auto-save paths) can hit the same
# namespace concurrently. The lock serializes each load→mutate→write so updates
# merge instead of clobbering one another.
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(path)
        if lock is None:
            lock = _LOCKS[path] = threading.Lock()
        return lock


class NotifyCache:
    """File-backed set of already-alerted keys, namespaced per page.

    Concurrency-safe: every read-modify-write is serialized by a per-file lock,
    and the file itself is written atomically (temp file + ``os.replace``) so a
    concurrent reader never sees a truncated/half-written JSON.
    """

    def __init__(self, namespace: str) -> None:
        self.path = os.path.join(os.getcwd(), f"{namespace}_notify_cache.json")
        self._lock = _lock_for(self.path)

    def load(self) -> set:
        try:
            if os.path.exists(self.path):
                with open(self.path) as fh:
                    return set(json.load(fh).get("keys", []))
        except Exception:
            pass
        return set()

    def _save(self, keys: set) -> None:
        """Atomically overwrite the file. Writes to a sibling temp file then
        ``os.replace``s it into place, so a concurrent reader sees either the old
        or the new file — never a partial one — and a crash mid-write can't
        corrupt the ledger."""
        try:
            dir_ = os.path.dirname(self.path) or "."
            fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".notify_", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump({"keys": sorted(keys)}, fh)
                os.replace(tmp, self.path)  # atomic on the same filesystem
            except Exception:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        except Exception:
            pass

    def save(self, keys: Iterable[str]) -> None:
        """Overwrite the persisted set (for callers that manage the set in
        session state and just need it durable)."""
        with self._lock:
            self._save(set(keys))

    def add(self, keys: Iterable[str]) -> set:
        """Atomically merge ``keys`` into the persisted set; return the result.

        Unlike :meth:`save`, this unions with whatever is already on disk under
        the lock, so concurrent callers accumulate keys instead of overwriting
        each other with a stale in-memory snapshot."""
        with self._lock:
            merged = self.load() | set(keys)
            self._save(merged)
            return merged

    def filter_new(self, keys: Iterable[str]) -> List[str]:
        """Return only keys not seen before, and record them as seen."""
        with self._lock:
            seen = self.load()
            new = [k for k in dict.fromkeys(keys) if k not in seen]  # de-dup + preserve order
            if new:
                seen.update(new)
                self._save(seen)
            return new

    def reset(self) -> None:
        with self._lock:
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
            except Exception:
                pass
