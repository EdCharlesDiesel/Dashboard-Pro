"""Propagate the VERSION file into the places Docker reads.

``VERSION`` is the single source of truth for the app version. Two consumers
cannot read it directly:

* **Compose's image tag.** ``image:`` interpolates ``${APP_VERSION}`` from the
  environment or a ``.env`` file; it cannot read an arbitrary file. So ``.env``
  is generated from ``VERSION`` rather than hand-edited.
* **The status bar** reads ``VERSION`` itself at runtime (see
  ``src/core/config.py``), which is why a container reports the version it was
  *built* from rather than whatever tag someone typed into compose.

Why this matters: the footer read ``v1.0.1`` for weeks after 1.0.1 shipped, then
``v1.1.0`` while the running image was 1.4.0. A version nobody updates is worse
than no version, because it is believed.

Usage::

    python deploy/sync_version.py            # write .env from VERSION
    python deploy/sync_version.py --check    # exit 1 if they disagree
    python deploy/sync_version.py 1.5.0      # bump VERSION, then write .env
"""
from __future__ import annotations

import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION_FILE = os.path.join(_REPO, "VERSION")
ENV_FILE = os.path.join(_REPO, ".env")

_HEADER = ("# Generated from VERSION -- the single source of truth for the app "
           "version.\n# Regenerate with: python deploy/sync_version.py\n")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")


def read_version(path: str | None = None) -> str:
    # Resolved at call time, never as a default argument: a default binds the
    # module global once at import, so reassigning VERSION_FILE (tests do) is
    # silently ignored and the *real* repo file gets read and written instead.
    # That is not hypothetical -- it rewrote this repo's VERSION during a test.
    path = path or VERSION_FILE
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def write_version(version: str, path: str | None = None) -> None:
    path = path or VERSION_FILE
    if not _SEMVER.match(version):
        raise ValueError("not a semantic version: {0!r}".format(version))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(version + "\n")


def env_version(path: str | None = None) -> str | None:
    """``APP_VERSION`` currently in ``.env``, or None if absent/unreadable."""
    path = path or ENV_FILE
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("APP_VERSION="):
                    return line.split("=", 1)[1].strip()
    except Exception:
        return None
    return None


def write_env(version: str, path: str | None = None) -> None:
    path = path or ENV_FILE
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_HEADER + "APP_VERSION={0}\n".format(version))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]

    if argv:
        write_version(argv[0])

    version = read_version()
    if check:
        current = env_version()
        if current == version:
            print("ok: VERSION and .env both {0}".format(version))
            return 0
        print("DRIFT: VERSION={0} but .env APP_VERSION={1}. "
              "Run: python deploy/sync_version.py".format(version, current))
        return 1

    write_env(version)
    print("VERSION={0} -> .env, compose will tag dashboard-pro:{0}".format(version))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
