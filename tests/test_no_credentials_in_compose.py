"""No credential may be written as a literal in docker-compose.yml.

That file is tracked. On 2026-08-22 it carried **16 literal credentials** —
both Postgres passwords, the FRED/FMP/Finnhub keys, the Gmail app password and
the Telegram pair — already committed and pushed to `origin/Production`.

A literal is also silently wrong, not just unsafe: **a literal beats `${VAR}`
interpolation**, so the Telegram one meant compose never read `.env` at all.
The scanner ran for hours with a 35-character fragment of a token — the secret
half without the `<bot-id>:` prefix — and every send returned `404 Not Found`
while the host worked fine.
"""
from __future__ import annotations

import os
import re
import subprocess

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPOSE = os.path.join(_REPO, "docker-compose.yml")

# Environment keys whose values are credentials by name.
_SECRET_KEY = re.compile(r"^[A-Z][A-Z0-9_]*(TOKEN|PASSWORD|SECRET|_KEY|CHAT_ID)$")

# `      KEY: value` — a service's environment entry, at compose's indentation.
_ENTRY = re.compile(r"^\s+([A-Z][A-Z0-9_]*):\s*(.*)$", re.M)


def _credential_entries() -> list:
    """``(key, value)`` for every credential-shaped environment entry.

    Read as text rather than parsed as YAML on purpose: the point is what the
    *file* contains, and a YAML loader would resolve nothing while a compose
    interpolation is not valid YAML semantics anyway.
    """
    with open(_COMPOSE, encoding="utf-8") as fh:
        text = fh.read()
    return [(m.group(1), m.group(2).strip())
            for m in _ENTRY.finditer(text) if _SECRET_KEY.match(m.group(1))]


def test_no_credential_is_a_literal():
    bad = sorted({k for k, v in _credential_entries()
                  if not v.strip().strip('"').startswith("${")})
    assert not bad, (
        f"literal credentials in docker-compose.yml: {bad} - move each value "
        f"into .env and use ${{KEY}} interpolation. This file is tracked, so a "
        f"literal is committed the moment it is written")


def test_the_database_credentials_fail_loudly_when_unset():
    """`${DB_PASSWORD:-}` would start Postgres with an empty password.

    Every page would then fall back silently — the exact failure mode that hid
    a wrong database target for weeks. `:?` stops compose with a message
    instead.
    """
    for key, value in _credential_entries():
        if key in ("DB_PASSWORD", "POSTGRES_PASSWORD"):
            assert ":?" in value, (
                f"{key} must use ${{{key}:?...}} so a missing value stops the "
                f"stack, not ${{{key}:-}} which starts it with an empty password")


def test_env_is_gitignored():
    # The whole scheme rests on .env never being committable.
    rc = subprocess.run(["git", "check-ignore", "-q", ".env"],
                        cwd=_REPO, capture_output=True, timeout=60).returncode
    assert rc == 0, ".env is not gitignored - moving secrets there would be worse"


def test_interpolation_names_match_the_env_file():
    """Every `${KEY}` in compose must exist in .env, or the value resolves empty.

    A typo here is invisible: compose substitutes nothing and the container
    starts with an empty credential.
    """
    env_path = os.path.join(_REPO, ".env")
    if not os.path.exists(env_path):
        return                      # a fresh clone; nothing to cross-check yet
    with open(env_path, encoding="utf-8") as fh:
        defined = {l.partition("=")[0].strip() for l in fh
                   if "=" in l and not l.lstrip().startswith("#")}

    missing = sorted({k for k, v in _credential_entries()
                      if v.strip().strip('"').startswith("${") and k not in defined})
    assert not missing, f"interpolated in compose but absent from .env: {missing}"
