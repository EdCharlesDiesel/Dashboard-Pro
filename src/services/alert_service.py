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

class NotifyCache:
    """File-backed set of already-alerted keys, namespaced per page."""

    def __init__(self, namespace: str) -> None:
        self.path = os.path.join(os.getcwd(), f"{namespace}_notify_cache.json")

    def load(self) -> set:
        try:
            if os.path.exists(self.path):
                with open(self.path) as fh:
                    return set(json.load(fh).get("keys", []))
        except Exception:
            pass
        return set()

    def _save(self, keys: set) -> None:
        try:
            with open(self.path, "w") as fh:
                json.dump({"keys": sorted(keys)}, fh)
        except Exception:
            pass

    def save(self, keys: Iterable[str]) -> None:
        """Overwrite the persisted set (for callers that manage the set in
        session state and just need it durable)."""
        self._save(set(keys))

    def filter_new(self, keys: Iterable[str]) -> List[str]:
        """Return only keys not seen before, and record them as seen."""
        seen = self.load()
        new = [k for k in dict.fromkeys(keys) if k not in seen]  # de-dup + preserve order
        if new:
            seen.update(new)
            self._save(seen)
        return new

    def reset(self) -> None:
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass
