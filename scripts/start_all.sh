#!/usr/bin/env bash
# Start the whole local stack:
#   - Neo4j          (Homebrew service)      -> http://localhost:7474
#   - Langfuse       (Docker, if available)  -> http://localhost:3000
#   - Prefect server (local process)         -> http://127.0.0.1:4200
#
# Usage: bash scripts/start_all.sh   (or `make up`)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.run"
mkdir -p "$RUN_DIR"

PREFECT_URL="http://127.0.0.1:4200/api"

wait_for() { # url, seconds, label
  local url="$1" secs="$2" label="$3" i=0
  while (( i < secs )); do
    if curl -s -o /dev/null "$url"; then
      echo "  [ok]   $label ($url)"
      return 0
    fi
    sleep 1
    (( i++ ))
  done
  echo "  [warn] $label did not respond within ${secs}s ($url)"
  return 1
}

echo "==> Neo4j"
if command -v brew >/dev/null 2>&1 && brew list --versions neo4j >/dev/null 2>&1; then
  if ! nc -z localhost 7687 2>/dev/null; then
    brew services start neo4j >/dev/null 2>&1 || true
  fi
  wait_for "http://localhost:7474" 60 "Neo4j browser"
else
  echo "  [skip] Neo4j not installed via Homebrew. Use docker-compose.yml or set NEO4J_URI."
fi

echo "==> Langfuse"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose -f docker-compose.langfuse.yml up -d
  wait_for "http://localhost:3000/api/public/health" 120 "Langfuse web"
else
  echo "  [skip] Docker not available/running. Install Docker Desktop, then:"
  echo "         docker compose -f docker-compose.langfuse.yml up -d"
fi

echo "==> Prefect"
if curl -s -o /dev/null "http://127.0.0.1:4200"; then
  echo "  [ok]   Prefect server already running (http://127.0.0.1:4200)"
else
  nohup uv run --no-sync prefect server start >"$RUN_DIR/prefect.log" 2>&1 &
  echo $! >"$RUN_DIR/prefect.pid"
  wait_for "http://127.0.0.1:4200" 60 "Prefect server"
fi
# Point the CLI at the running server so flow runs show up in the UI.
uv run --no-sync prefect config set PREFECT_API_URL="$PREFECT_URL" >/dev/null 2>&1 || true

echo
echo "===================== Services ====================="
echo "  Neo4j      http://localhost:7474   (neo4j / )"
echo "  Langfuse   http://localhost:3000   (demo@example.com / )"
echo "  Prefect    http://127.0.0.1:4200"
echo "===================================================="
echo "Stop everything with: make down"
