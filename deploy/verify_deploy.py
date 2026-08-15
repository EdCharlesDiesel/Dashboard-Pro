"""Does the running container actually contain the code in this working tree?

The existing guard, ``sync_version.py --check``, verifies that VERSION, ``.env``
and the compose tag all agree. That is a check on *labels*. It passes happily
while the running image is a day behind the source, because rebuilding without
bumping VERSION makes one tag mean two different codebases — and then every
downstream signal (the footer, ``docker ps``, the compose file) reports a
version that is true about the label and false about the code.

That is not hypothetical. On 2026-08-14 the desk ran ``dashboard-pro:1.7.1``
for 28 hours after the `sl_pips` fix landed in ``src/core/signals.py``. Every
check passed. The sweeper inside that container kept writing rows with the bug,
because a fix in git is not a fix in production.

So this compares **contents**: it hashes the Python sources on disk and the
same paths inside each running container, using one function for both sides so
the two can never disagree about the algorithm.

    python deploy/verify_deploy.py            # compare; exit 1 on drift
    python deploy/verify_deploy.py --quiet    # exit code only

``deploy/`` is excluded by ``.dockerignore``, so this file is never in the
image and cannot run there. The container's files are streamed out with ``tar``
and hashed here instead.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
from typing import Dict, List, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The trees that make up the application. Kept narrow on purpose: comparing
# everything would drag in runtime state (caches, logs) that legitimately
# differs between host and container and would make the check cry wolf.
TREES = ("src", "pages")
FILES = ("app.py",)

_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _is_source(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    if any(p in _SKIP_DIRS for p in parts):
        return False
    return rel.endswith(".py")


def fingerprint(files: Dict[str, bytes]) -> str:
    """SHA-256 over ``{relative path: bytes}``, order-independent.

    Paths are sorted and length-prefixed so that a rename cannot collide with a
    content change, and the digest is identical whichever side computed it.
    """
    digest = hashlib.sha256()
    for rel in sorted(files):
        norm = rel.replace("\\", "/").encode("utf-8")
        digest.update(str(len(norm)).encode("ascii") + b":" + norm)
        body = files[rel]
        digest.update(str(len(body)).encode("ascii") + b":")
        digest.update(body)
    return digest.hexdigest()


def read_host_tree(root: str = _REPO) -> Dict[str, bytes]:
    out: Dict[str, bytes] = {}
    for tree in TREES:
        base = os.path.join(root, tree)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root)
                if _is_source(rel):
                    with open(full, "rb") as handle:
                        out[rel.replace("\\", "/")] = handle.read()
    for name in FILES:
        full = os.path.join(root, name)
        if os.path.exists(full):
            with open(full, "rb") as handle:
                out[name] = handle.read()
    return out


def read_container_tree(container: str) -> Dict[str, bytes]:
    """Stream /app sources out of a running container via tar."""
    paths = list(TREES) + list(FILES)
    proc = subprocess.run(
        ["docker", "exec", container, "tar", "-cf", "-", "-C", "/app"] + paths,
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()
                           or "tar failed in {0}".format(container))
    out: Dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not _is_source(member.name):
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                out[member.name.replace("\\", "/")] = handle.read()
    return out


def parse_ps(stdout: str) -> List[str]:
    """Container names whose image is a dashboard-pro tag, from `docker ps`."""
    names = []
    for line in stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].startswith("dashboard-pro:"):
            names.append(parts[0])
    return sorted(names)


def app_containers() -> List[str]:
    """Running containers built from the app image (the db is not one).

    Filtering happens here rather than via ``--filter ancestor=dashboard-pro``:
    that filter resolves the name to a single image id, so it silently matches
    nothing whenever the tag has moved — which is precisely the situation this
    tool exists to detect.
    """
    proc = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "docker ps failed")
    return parse_ps(proc.stdout)


IN_SYNC = 0
DRIFT = 1
UNVERIFIABLE = 2


def compare() -> Tuple[int, List[str]]:
    """Returns ``(status, lines)`` where status is IN_SYNC / DRIFT / UNVERIFIABLE.

    "Could not check" is deliberately a different status from "checked and the
    code is stale". Collapsing them means a Docker daemon that is simply down
    reads as a code problem, and — worse — a real drift reads as a glitch worth
    retrying. Only DRIFT is a statement about the code.
    """
    host = read_host_tree()
    host_digest = fingerprint(host)
    lines = ["host  {0}  ({1} files)".format(host_digest[:12], len(host))]

    try:
        containers = app_containers()
    except Exception as exc:  # noqa: BLE001 — report, never crash the check
        lines.append("cannot list containers: {0}".format(exc))
        return UNVERIFIABLE, lines
    if not containers:
        lines.append("no running dashboard-pro containers to compare against")
        return UNVERIFIABLE, lines

    status = IN_SYNC
    for name in containers:
        try:
            theirs = read_container_tree(name)
        except Exception as exc:  # noqa: BLE001 — report, never crash the check
            status = max(status, UNVERIFIABLE)
            lines.append("{0:<28} UNREADABLE: {1}".format(name, exc))
            continue
        digest = fingerprint(theirs)
        match = digest == host_digest
        if not match:
            status = max(status, DRIFT)
        lines.append("{0:<28} {1}  ({2} files) {3}".format(
            name, digest[:12], len(theirs), "OK" if match else "DRIFT"))
        if not match:
            missing = sorted(set(host) - set(theirs))
            extra = sorted(set(theirs) - set(host))
            changed = sorted(p for p in (set(host) & set(theirs))
                             if host[p] != theirs[p])
            for label, items in (("missing from container", missing),
                                 ("only in container", extra),
                                 ("differs", changed)):
                if items:
                    lines.append("    {0}: {1}{2}".format(
                        label, ", ".join(items[:5]),
                        " (+{0} more)".format(len(items) - 5)
                        if len(items) > 5 else ""))
    return status, lines


_VERDICT = {
    IN_SYNC: "in sync",
    DRIFT: ("OUT OF SYNC - rebuild: python deploy/sync_version.py <next>; "
            "docker compose build; docker compose up -d"),
    UNVERIFIABLE: "COULD NOT VERIFY - this says nothing about the code",
}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress output on success. Failures still explain "
                         "themselves.")
    args = ap.parse_args(argv)

    status, lines = compare()
    # --quiet silences the *success* chatter only. A guard that exits non-zero
    # without saying why cannot be acted on, and an unexplained failure is one
    # people learn to ignore.
    if not args.quiet or status != IN_SYNC:
        for line in lines:
            print(line)
        print(_VERDICT[status])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
