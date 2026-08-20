#!/usr/bin/env bash
# Append-only audit of every tool call in a trading session.
#
# Deterministic on purpose: an agent can summarise six calls as "checked the
# book", and the one call that mattered is the one it did not mention. When the
# question later is "what actually reached the broker, at what size", this file
# is the only answer not written by the thing being audited.
#
# logs/ is gitignored and .dockerignored, so this never leaves the machine.
set -euo pipefail

LOG="$(dirname "$0")/../../logs/mt5_tool_audit.jsonl"
mkdir -p "$(dirname "$LOG")"

printf '{"ts":"%s","tool":"%s","input":%s}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${CLAUDE_TOOL_NAME:-unknown}" \
  "${CLAUDE_TOOL_INPUT:-null}" >> "$LOG"
