"""Extract the code-level blueprint behind `.foglamp/scan.json` into `code.json`.

`scan.json` says *what* each node is; this says *what you would actually write*
to build it. For every node with a `sourceRef` it walks the real file with
`ast` and records the classes (with their bases and methods), the top-level
functions with signatures, the module docstring's first line, the line count,
and which internal modules it imports. For the infrastructure nodes it records
the provisioning surface instead: the compose service, the run command, and the
**names** of the environment variables involved.

    python .foglamp/introspect.py          # rewrite code.json
    python .foglamp/introspect.py --print   # also dump a readable summary

Never records a secret. Environment variables are captured as bare key names
with their values discarded at parse time, and no file contents are copied into
the output beyond signatures and the first docstring line.

Run this after `scan.json` changes, then `render.py` to rebuild `scan.html`.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCAN = HERE / "scan.json"
OUT = HERE / "code.json"

# Directory nodes: sourceRef is a folder, so summarise it rather than listing
# every file. (pages/ alone is 57 modules.)
DIR_SAMPLE = 6


def first_line(doc: str | None) -> str:
    if not doc:
        return ""
    for line in doc.strip().splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return ""


def sig(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a def's parameter list, dropping annotations for readability."""
    a = fn.args
    parts: list[str] = []
    pos = list(a.posonlyargs) + list(a.args)
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, dflt in zip(pos, defaults):
        parts.append(arg.arg + ("=…" if dflt is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    for arg, dflt in zip(a.kwonlyargs, a.kw_defaults):
        parts.append(arg.arg + ("=…" if dflt is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    ret = ""
    if fn.returns is not None:
        try:
            ret = " -> " + ast.unparse(fn.returns)
        except Exception:
            ret = ""
    return f"({', '.join(parts)}){ret}"[:120]


def internal_imports(tree: ast.AST) -> list[str]:
    """Modules this file imports from inside the repo (src.*, deploy, pages)."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
        elif isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in ("src", "deploy", "pages"):
                    found.add(al.name)
            continue
        else:
            continue
        if mod.split(".")[0] in ("src", "deploy", "pages"):
            found.add(mod)
    return sorted(found)


def read_file(path: Path) -> dict | None:
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return None

    classes, funcs = [], []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                b.name + sig(b)
                for b in node.body
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not b.name.startswith("__")
            ]
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            classes.append({
                "name": node.name,
                "bases": bases,
                "doc": first_line(ast.get_docstring(node)),
                "methods": methods[:14],
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not node.name.startswith("__"):
                kind = "private"
            else:
                kind = "public"
            funcs.append({
                "name": node.name,
                "sig": sig(node),
                "doc": first_line(ast.get_docstring(node)),
                "kind": kind,
            })

    return {
        "path": path.relative_to(ROOT).as_posix(),
        "loc": len(src.splitlines()),
        "doc": first_line(ast.get_docstring(tree)),
        "classes": classes,
        "functions": funcs,
        "imports": internal_imports(tree),
    }


def compose_services() -> dict:
    """Service name -> {command, image, ports, depends_on, env_keys}.

    A deliberately small indent-aware reader (no PyYAML in requirements.txt).
    Environment values are split off and thrown away — only key names survive.
    """
    path = ROOT / "docker-compose.yml"
    if not path.exists():
        return {}
    services: dict[str, dict] = {}
    cur: str | None = None
    section: str | None = None
    in_services = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            in_services = line.startswith("services:")
            cur = section = None
            continue
        if not in_services:
            continue
        if indent == 2 and line.endswith(":"):
            cur = line[:-1].strip()
            services[cur] = {"command": "", "image": "", "ports": [],
                             "depends_on": [], "env_keys": []}
            section = None
            continue
        if cur is None:
            continue
        if indent == 4:
            section = None
            if line.startswith("command:"):
                services[cur]["command"] = line.split(":", 1)[1].strip()
            elif line.startswith("image:"):
                services[cur]["image"] = line.split(":", 1)[1].strip()
            elif line.startswith("build:"):
                services[cur]["image"] = "build ./Dockerfile"
            elif line.rstrip(":") in ("environment", "ports", "depends_on"):
                section = line.rstrip(":")
            continue
        if indent >= 6 and section:
            if section == "environment":
                key = line.lstrip("- ").split(":", 1)[0].split("=", 1)[0].strip()
                if key and key not in services[cur]["env_keys"]:
                    services[cur]["env_keys"].append(key)   # name only, value dropped
            elif section == "ports":
                services[cur]["ports"].append(line.lstrip("- ").strip().strip('"'))
            elif section == "depends_on" and line.endswith(":"):
                services[cur]["depends_on"].append(line[:-1].strip())
    return services


# Provisioning surface for the nodes that are infrastructure, not modules.
# `env` lists variable NAMES the node needs; values live in secrets.toml / env.
OPS = {
    "pg": {"how": "docker compose up db", "service": "db",
           "note": "Published on 5433, not 5432 - a native Windows Postgres owns 5432.",
           "env": ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]},
    "mt5-terminal": {"how": "Install MetaTrader 5 and log in; Windows only",
                     "note": "Needs the interactive desktop session - Task Scheduler cannot attach.",
                     "env": ["MT5_TERMINAL_PATH"]},
    "yahoo": {"how": "pip install yfinance", "env": []},
    "fred": {"how": "Free API key from fred.stlouisfed.org", "env": ["FRED_API_KEY"]},
    "fmp": {"how": "API key from financialmodelingprep.com", "env": ["FMP_API_KEY"]},
    "finnhub": {"how": "API key from finnhub.io", "env": ["FINNHUB_API_KEY"]},
    "anthropic-api": {"how": "pip install anthropic; key from console.anthropic.com",
                      "note": "Optional - every AI surface self-disables without it.",
                      "env": ["ANTHROPIC_API_KEY"]},
    "cftc": {"how": "pip install cot_reports", "env": []},
    "ff": {"how": "Public weekly JSON feed, no key", "env": []},
    "gmail": {"how": "Gmail account + app password", "env": ["GMAIL_SENDER", "GMAIL_APP_PASSWORD", "GMAIL_RECIPIENT"]},
    "telegram": {"how": "BotFather bot + chat id", "env": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]},
    "healthchecks": {"how": "Create a check at healthchecks.io", "env": ["HEALTHCHECK_URL"]},
    "notify-cache": {"how": "No setup - JSON files created on first alert",
                     "note": "Gitignored; each scanner passes its own namespace."},
    "mcp-server": {"how": "python -m src.services.mt5_mcp",
                   "note": "Registered with the MCP client at user scope; Windows only."},
    "mt5-sync": {"how": "deploy/mt5_sync_startup.cmd in the Startup folder",
                 "note": "Must stay a shortcut, not a copy."},
    "mt5-watchdog": {"how": "deploy/mt5_watchdog_startup.cmd in the Startup folder"},
}


def main() -> int:
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    services = compose_services()

    # node id -> compose service that runs it, matched on the module in `command`
    runner: dict[str, str] = {}
    for name, svc in services.items():
        for node in scan["graph"]["nodes"]:
            ref = (node.get("sourceRef") or "").split(":")[0]
            if not ref.endswith(".py"):
                continue
            module = ref[:-3].replace("/", ".")
            if module and module in svc.get("command", ""):
                runner[node["id"]] = name
    runner.setdefault("cockpit", "app")
    runner.setdefault("pg", "db")

    out: dict[str, dict] = {}
    missing: list[str] = []
    for node in scan["graph"]["nodes"]:
        entry: dict = {}
        ref = (node.get("sourceRef") or "").split(":")[0]
        if ref:
            target = ROOT / ref
            if target.is_dir():
                files = sorted(target.glob("*.py"))
                infos = [read_file(f) for f in files[:DIR_SAMPLE]]
                entry["dir"] = {
                    "path": ref,
                    "count": len(files),
                    "loc": sum((read_file(f) or {}).get("loc", 0) for f in files),
                    "sample": [i for i in infos if i],
                }
            elif target.is_file():
                info = read_file(target)
                if info:
                    entry["file"] = info
                else:
                    missing.append(ref)
            else:
                missing.append(ref)
        if node["id"] in OPS:
            entry["ops"] = dict(OPS[node["id"]])
        if node["id"] in runner and runner[node["id"]] in services:
            entry.setdefault("ops", {})["service"] = runner[node["id"]]
            entry["ops"]["compose"] = services[runner[node["id"]]]
        if entry:
            out[node["id"]] = entry

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    files = sum(1 for v in out.values() if "file" in v)
    dirs = sum(1 for v in out.values() if "dir" in v)
    ops = sum(1 for v in out.values() if "ops" in v)
    classes = sum(len(v.get("file", {}).get("classes", [])) for v in out.values())
    funcs = sum(len(v.get("file", {}).get("functions", [])) for v in out.values())
    print(f"wrote {OUT.name}: {len(out)} nodes  "
          f"({files} files, {dirs} dirs, {ops} with ops)  "
          f"{classes} classes, {funcs} functions")
    if missing:
        print("  sourceRef not found on disk (fix scan.json):", file=sys.stderr)
        for m in missing:
            print("   -", m, file=sys.stderr)
        return 1

    if "--print" in sys.argv:
        for nid, v in out.items():
            f = v.get("file") or {}
            if not f:
                continue
            print(f"\n{nid}  {f['path']}  ({f['loc']} loc)")
            for c in f["classes"]:
                base = f"({', '.join(c['bases'])})" if c["bases"] else ""
                print(f"   class {c['name']}{base}")
                for m in c["methods"]:
                    print(f"      .{m}")
            for fn in f["functions"]:
                if fn["kind"] == "public":
                    print(f"   def {fn['name']}{fn['sig']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
