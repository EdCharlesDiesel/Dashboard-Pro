"""Generate `docs/Architecture.md` — the module-level map of `src/`.

`.foglamp/` already maps the *deployment* graph: which container runs what, and
which environment each service needs. This answers the other question, the one a
person asks when they open the repo for the first time: **what are all these
files, and which ones call which?**

Everything here is read out of the code with `ast`, never hand-written, because a
hand-maintained architecture document is wrong within a fortnight and then
actively misleads. Re-run it after any change that adds a module or moves an
import:

    python docs/generate_architecture.py

Records only structure: the first line of each module docstring, the internal
imports, and the public callables. No file contents, no secrets.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict
from datetime import date
from typing import Dict, List, Set, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
_OUT = os.path.join(_REPO, "docs", "Architecture.md")

# One note per module, so Obsidian has something to draw a graph from.
#
# Obsidian builds its graph from [[wikilinks]] between *notes*. A markdown table
# listing module names in backticks — which is what this file emitted first —
# produces a graph of isolated dots, because backticks are not links and .py
# files only appear as nodes when a note links to them. A note per module, each
# wikilinking the modules it imports, turns the same 237 import edges into 237
# graph edges.
_NOTES = os.path.join(_REPO, "docs", "architecture")

# Vault-relative prefix for the junction that exposes src/ inside Obsidian
# (Obsidian/Dashboard-Pro/Src -> src). Links use it so a note can open the real
# file; the parallel markdown link keeps the same reference working on GitHub.
_VAULT_SRC = "Src"

# The layer each package sits in, and what it is allowed to know about. The rule
# the codebase actually follows: dependencies point *down* this list, never up.
# `core` is pure logic with no I/O; `pages_lib`/`ui` are the only Streamlit-aware
# layers; `db` and `services` sit between them.
LAYERS: List[Tuple[str, str]] = [
    ("instruments", "The registry — the one list of tradeable symbols. Imports nothing."),
    ("core", "Pure logic: signals, maths, indicators. No database, no network, no Streamlit."),
    ("indicators", "Technical-indicator maths, used by core and by pages."),
    ("db", "Postgres access and the OHLC read-through cache. Knows nothing about pages."),
    ("services", "Orchestration: fetches, sweeps, scoring, broker sync. Where I/O lives."),
    ("data_backbone", "The background store refresher (its own container)."),
    ("pages_lib", "Streamlit page bodies. The only place `st.*` is normal."),
    ("ui", "Theme and shared widgets. Owns every colour and font in the app."),
]


def _layer_of(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 2 else (parts[1] if len(parts) > 1 else "src")


def _module_name(rel: str) -> str:
    return rel.replace("\\", "/")[:-3].replace("/", ".")


def scan() -> Tuple[Dict[str, dict], Dict[str, Set[str]]]:
    """Every module under src/, plus the internal import edges between them."""
    modules: Dict[str, dict] = {}
    edges: Dict[str, Set[str]] = defaultdict(set)

    for root, _dirs, files in os.walk(_SRC):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            if not name.endswith(".py") or name == "__init__.py":
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, _REPO)
            mod = _module_name(rel)
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                tree = ast.parse(source)
            except Exception:
                continue

            doc = (ast.get_docstring(tree) or "").strip().splitlines()
            purpose = doc[0].rstrip(".") if doc else ""
            callables = [n.name for n in tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                           ast.ClassDef))
                         and not n.name.startswith("_")]

            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
                    target = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("src."):
                            edges[mod].add(alias.name)
                if target:
                    edges[mod].add(target)

            modules[mod] = {
                "rel": rel.replace("\\", "/"),
                "layer": _layer_of(rel),
                "purpose": purpose,
                "callables": callables,
                "lines": source.count("\n") + 1,
            }
    return modules, edges


def _resolve(edges: Dict[str, Set[str]], modules: Dict[str, dict]) -> Dict[str, Set[str]]:
    """Keep only edges whose target is a module we actually scanned."""
    known = set(modules)
    out: Dict[str, Set[str]] = {}
    for src_mod, targets in edges.items():
        if src_mod not in known:
            continue
        out[src_mod] = {t for t in targets if t in known}
    return out


def render(modules: Dict[str, dict], edges: Dict[str, Set[str]]) -> str:
    importers: Dict[str, Set[str]] = defaultdict(set)
    for src_mod, targets in edges.items():
        for target in targets:
            importers[target].add(src_mod)

    layer_edges: Dict[Tuple[str, str], int] = defaultdict(int)
    for src_mod, targets in edges.items():
        for target in targets:
            a, b = modules[src_mod]["layer"], modules[target]["layer"]
            if a != b:
                layer_edges[(a, b)] += 1

    lines: List[str] = []
    add = lines.append
    add("# Architecture — how the code connects")
    add("")
    add("> Generated by `python docs/generate_architecture.py`. Do not edit by "
        "hand: every fact below is read out of the source with `ast`, so it is "
        "correct on the day it is run and wrong the moment someone edits it "
        "instead of regenerating.")
    add("")
    add("_Last generated {0} — {1} modules, {2} internal import edges._".format(
        date.today().isoformat(), len(modules), sum(len(v) for v in edges.values())))
    add("")
    add("`.foglamp/scan.html` maps the **deployment** graph — which container "
        "runs what. This maps the **code** — which module calls which.")
    add("")
    add("Every module below has its own note listing what it imports and what "
        "imports it, so the Obsidian graph is the real dependency graph rather "
        "than a scatter of unlinked dots. Start anywhere and follow the edges.")
    add("")
    # Wikilinks, not prose: these are what stitch the documentation cluster to
    # the code cluster in the graph. Without them the docs float as an island.
    add("**The rest of the documentation:** [[README]] · [[System_Guide]] · "
        "[[Live_Session_Runbook]] · [[MT5_MCP]]")
    add("")

    add("## Layers")
    add("")
    add("Dependencies point down this list. A module importing from a layer "
        "above it is the signal that something is in the wrong place.")
    add("")
    add("| Layer | Modules | Role |")
    add("|---|---:|---|")
    for layer, role in LAYERS:
        count = sum(1 for m in modules.values() if m["layer"] == layer)
        if count:
            add("| `src/{0}/` | {1} | {2} |".format(layer, count, role))
    add("")

    add("## Layer dependency graph")
    add("")
    add("```mermaid")
    add("graph TD")
    order = [name for name, _ in LAYERS]
    for layer, _role in LAYERS:
        if any(m["layer"] == layer for m in modules.values()):
            add('    {0}["src/{0}"]'.format(layer))
    for (a, b), weight in sorted(layer_edges.items(), key=lambda kv: -kv[1]):
        if a in order and b in order:
            add("    {0} -->|{1}| {2}".format(a, weight, b))
    add("```")
    add("")
    add("Edge labels are the number of import statements, so a thick edge is a "
        "real coupling and a thin one is often a single helper.")
    add("")

    add("## The most-depended-on modules")
    add("")
    add("If you are changing one of these, the blast radius is the second "
        "column. These are the load-bearing walls.")
    add("")
    add("| Module | Imported by | Purpose |")
    add("|---|---:|---|")
    ranked = sorted(modules, key=lambda m: -len(importers.get(m, ())))
    for mod in ranked[:15]:
        n = len(importers.get(mod, ()))
        if n < 2:
            continue
        # r-string: Obsidian needs a literal backslash-pipe to alias a wikilink
        # inside a table cell, and "\|" is not a valid Python escape.
        add(r"| [[{0}\|{1}]] | {2} | {3} |".format(
            mod, os.path.basename(modules[mod]["rel"]), n,
            modules[mod]["purpose"][:90] or "—"))
    add("")

    for layer, role in LAYERS:
        members = sorted(m for m in modules if modules[m]["layer"] == layer)
        if not members:
            continue
        add("## `src/{0}/`".format(layer))
        add("")
        add("_{0}_".format(role))
        add("")
        add("| Module | Lines | Imported by | Purpose |")
        add("|---|---:|---:|---|")
        for mod in members:
            info = modules[mod]
            add(r"| [[{0}\|{1}]] | {2} | {3} | {4} |".format(
                mod, os.path.basename(info["rel"]), info["lines"],
                len(importers.get(mod, ())), info["purpose"][:100] or "—"))
        add("")

    add("## Orphans")
    add("")
    add("Modules nothing else imports. Some are entry points (a page body, a "
        "CLI); the rest are candidates for deletion.")
    add("")
    unreferenced = [m for m in sorted(modules) if not importers.get(m)]
    if unreferenced:
        for mod in unreferenced:
            add("- [[{0}]] — {1}".format(
                mod, modules[mod]["purpose"][:80] or "no docstring"))
    else:
        add("_None._")
    add("")
    return "\n".join(lines)


def write_notes(modules: Dict[str, dict], edges: Dict[str, Set[str]]) -> int:
    """One note per module, wikilinked to what it imports and what imports it.

    This is what makes the Obsidian graph real. Note names are the dotted module
    path (`src.core.signals`), which is unique across the repo, so wikilinks
    resolve without folder qualification.
    """
    importers: Dict[str, Set[str]] = defaultdict(set)
    for src_mod, targets in edges.items():
        for target in targets:
            importers[target].add(src_mod)

    os.makedirs(_NOTES, exist_ok=True)
    # Stale notes for deleted modules would linger as orphan graph nodes.
    for existing in os.listdir(_NOTES):
        if existing.endswith(".md") and existing[:-3] not in modules:
            os.remove(os.path.join(_NOTES, existing))

    for mod, info in modules.items():
        rel = info["rel"]
        vault_path = rel.replace("src/", _VAULT_SRC + "/", 1)
        out: List[str] = []
        add = out.append
        add("---")
        add("layer: {0}".format(info["layer"]))
        add("lines: {0}".format(info["lines"]))
        add("generated: true")
        add("---")
        add("")
        add("# `{0}`".format(rel))
        add("")
        add(info["purpose"] or "_No module docstring._")
        add("")
        add("Source: [[{0}]] · [open on disk]({1})".format(
            vault_path, os.path.relpath(rel, "docs").replace("\\", "/")))
        add("")

        outgoing = sorted(edges.get(mod, ()))
        add("## Imports ({0})".format(len(outgoing)))
        add("")
        if outgoing:
            for target in outgoing:
                add("- [[{0}]] — {1}".format(
                    target, modules[target]["purpose"][:70] or "—"))
        else:
            add("_Nothing internal. This module is a leaf._")
        add("")

        incoming = sorted(importers.get(mod, ()))
        add("## Imported by ({0})".format(len(incoming)))
        add("")
        if incoming:
            for source in incoming:
                add("- [[{0}]]".format(source))
        else:
            add("_Nothing imports this. Either an entry point or dead code._")
        add("")

        if info["callables"]:
            add("## Public surface")
            add("")
            add(", ".join("`{0}`".format(c) for c in info["callables"][:25]))
            add("")
        add("Back to [[Architecture]].")
        add("")

        with open(os.path.join(_NOTES, mod + ".md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(out))
    return len(modules)


def main() -> int:
    modules, raw_edges = scan()
    edges = _resolve(raw_edges, modules)
    with open(_OUT, "w", encoding="utf-8") as handle:
        handle.write(render(modules, edges))
    written = write_notes(modules, edges)
    print("wrote {0} and {1} module notes in {2}/ — {3} edges".format(
        os.path.relpath(_OUT, _REPO), written,
        os.path.relpath(_NOTES, _REPO),
        sum(len(v) for v in edges.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
