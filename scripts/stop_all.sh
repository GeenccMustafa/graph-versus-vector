#!/usr/bin/env bash
# Stop the local stack started by scripts/start_all.sh.
#   - Prefect server (from .run/prefect.pid)
#   - Langfuse Docker stack
#   - Neo4j (Homebrew) -- skipped if KEEP_NEO4J=1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.run"

echo "==> Prefect"
# `uv run prefect server start` spawns a child Python process that owns the
# port, so kill the child(ren) *and* the parent, then fall back to a match.
if [[ -f "$RUN_DIR/prefect.pid" ]]; then
  PID="$(cat "$RUN_DIR/prefect.pid")"
  pkill -P "$PID" 2>/dev/null || true
  kill "$PID" 2>/dev/null || true
  rm -f "$RUN_DIR/prefect.pid"
fi
pkill -f "prefect server start" 2>/dev/null || true

# Wait briefly for the port to free up.
for _ in $(seq 1 15); do
  curl -s -o /dev/null "http://127.0.0.1:4200" || break
  sleep 1
done
if curl -s -o /dev/null "http://127.0.0.1:4200"; then
  echo "  warning: something is still listening on :4200"
else
  echo "  stopped Prefect"
fi

echo "==> Langfuse"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.langfuse.yml down
else
  echo "  Docker not available; nothing to stop"
fi

echo "==> Neo4j"
if [[ "${KEEP_NEO4J:-0}" == "1" ]]; then
  echo "  KEEP_NEO4J=1, leaving Neo4j running"
elif command -v brew >/dev/null 2>&1; then
  brew services stop neo4j >/dev/null 2>&1 && echo "  stopped Neo4j" || echo "  Neo4j was not running"
fi

echo "Done."
