"""Validate `.foglamp/scan.json` and regenerate `.foglamp/scan.html` from it.

The scan is the architecture map of this repo: entry points, scheduled jobs,
AI surfaces, the internal services the product is built from, and the stores
and third parties they touch. `scan.json` is the only thing anyone edits;
`scan.html` is a generated artifact (gitignored) and `template.html` is the
renderer it is spliced into.

    python .foglamp/render.py            # validate + regenerate scan.html
    python .foglamp/render.py --check    # validate only, write nothing

**Keep the scan current.** When a feature lands that adds an entry point, a
worker, an AI call, a service module, a table or a third-party dependency, add
the node and its edges to `scan.json` and re-run this. The rebuild numbering in
the map is *derived from the graph at render time*, so a new node renumbers
everything automatically — there is no ordering to maintain by hand.

The validator enforces foglamp.dev's published contract, so a scan that passes
here is also safe to upload (see the Publish section of the scan brief). The
caps are theirs, not ours; do not relax them to make an edit fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCAN = HERE / "scan.json"
CODE = HERE / "code.json"
TEMPLATE = HERE / "template.html"
OUT = HERE / "scan.html"

KINDS = {"entry", "cron", "agent", "model", "tool", "service", "store", "external"}
EDGE_KINDS = {"calls", "reads", "writes", "triggers"}

# Contract caps: (json path, limit). Lists are capped by length, strings by chars.
LIST_CAPS = [
    ("topModels", 3),
    ("topTools", 10),
    ("topIntegrations", 10),
    ("graph.nodes", 60),
    ("graph.edges", 120),
]
STR_CAPS = {
    "project.name": 48,
    "project.slug": 48,
    "project.tagline": 80,
    "node.label": 28,
    "node.sub": 40,
    "node.detail": 200,
    "node.sourceRef": 120,
    "node.group": 24,
    "edge.label": 24,
}


def _dig(obj, path):
    for part in path.split("."):
        obj = obj[part]
    return obj


def validate(scan: dict) -> list[str]:
    """Return a list of contract violations; empty means the scan is valid."""
    bad: list[str] = []

    def cap(kind: str, value, where: str) -> None:
        limit = STR_CAPS[kind]
        if value is not None and len(value) > limit:
            bad.append(f"{where}: {kind} is {len(value)} chars, max {limit} -> {value!r}")

    for path, limit in LIST_CAPS:
        try:
            n = len(_dig(scan, path))
        except (KeyError, TypeError):
            bad.append(f"missing required list: {path}")
            continue
        if n > limit:
            bad.append(f"{path} has {n} entries, max {limit}")

    for key in ("name", "slug", "tagline"):
        val = scan.get("project", {}).get(key)
        if key != "tagline" and not val:
            bad.append(f"project.{key} is required")
        cap(f"project.{key}", val, "project")

    slug = scan.get("project", {}).get("slug", "")
    if slug and (slug != slug.lower() or " " in slug):
        bad.append(f"project.slug must be lowercase-dashed -> {slug!r}")

    nodes = scan.get("graph", {}).get("nodes", [])
    edges = scan.get("graph", {}).get("edges", [])

    seen: set[str] = set()
    for n in nodes:
        nid = n.get("id", "<no id>")
        if not n.get("id"):
            bad.append("a node is missing its id")
        elif nid in seen:
            bad.append(f"duplicate node id: {nid}")
        seen.add(nid)

        if n.get("kind") not in KINDS:
            bad.append(f"{nid}: kind {n.get('kind')!r} not one of {sorted(KINDS)}")
        if not n.get("label"):
            bad.append(f"{nid}: label is required")
        for key in ("label", "sub", "detail", "sourceRef", "group"):
            cap(f"node.{key}", n.get(key), nid)
        if (n.get("domain") or "").startswith(("http://", "https://")):
            bad.append(f"{nid}: domain must be a bare favicon domain, no scheme")

    for i, e in enumerate(edges):
        where = f"edge[{i}] {e.get('from')}->{e.get('to')}"
        for end in ("from", "to"):
            if e.get(end) not in seen:
                bad.append(f"{where}: {end} references unknown node {e.get(end)!r}")
        if e.get("kind") and e["kind"] not in EDGE_KINDS:
            bad.append(f"{where}: kind {e['kind']!r} not one of {sorted(EDGE_KINDS)}")
        cap("edge.label", e.get("label"), where)

    return bad


def render(scan: dict, code: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    for token in ("__SCAN_JSON__", "__CODE_JSON__"):
        if token not in tpl:
            raise SystemExit(f"{TEMPLATE.name} has no {token} placeholder")
    for token, payload in (("__SCAN_JSON__", scan), ("__CODE_JSON__", code)):
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        # The blob lands inside <script type="application/json">; a literal
        # closing tag anywhere in the data would end the element early.
        if "</script" in blob.lower():
            raise SystemExit(f"{token} data contains a literal </script and cannot be inlined")
        tpl = tpl.replace(token, blob)
    return tpl


def main() -> int:
    check_only = "--check" in sys.argv

    if not SCAN.exists():
        print(f"missing {SCAN}", file=sys.stderr)
        return 2
    scan = json.loads(SCAN.read_text(encoding="utf-8"))

    problems = validate(scan)
    if problems:
        print(f"scan.json fails the contract ({len(problems)} problem(s)):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    nodes = scan["graph"]["nodes"]
    edges = scan["graph"]["edges"]
    print(f"scan.json OK - {len(nodes)} nodes, {len(edges)} edges")

    if check_only:
        return 0

    if not TEMPLATE.exists():
        print(f"missing {TEMPLATE}", file=sys.stderr)
        return 2

    # The code-level blueprint is optional: without it the map still renders,
    # just with no class/function detail. Regenerate it with introspect.py.
    code: dict = {}
    if CODE.exists():
        code = json.loads(CODE.read_text(encoding="utf-8"))
        stale = [nid for nid in code if nid not in {n["id"] for n in nodes}]
        if stale:
            print(f"  note: code.json has {len(stale)} node(s) no longer in scan.json "
                  f"({', '.join(sorted(stale)[:4])}…) - rerun introspect.py")
    else:
        print("  note: no code.json - run `python .foglamp/introspect.py` for the blueprint")

    OUT.write_text(render(scan, code), encoding="utf-8")
    print(f"wrote {OUT.relative_to(HERE.parent)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
