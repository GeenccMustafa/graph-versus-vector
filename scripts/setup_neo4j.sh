#!/usr/bin/env bash
# Install and start Neo4j locally via Homebrew (macOS), and set the password
# used by this project's .env.
set -euo pipefail

PASSWORD="${1:-}"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 1
fi

if ! command -v neo4j >/dev/null 2>&1; then
  echo "Installing neo4j via Homebrew ..."
  brew install neo4j
fi

NEO4J_ADMIN="$(brew --prefix neo4j)/bin/neo4j-admin"

# Setting the initial password must happen before the first start.
"$NEO4J_ADMIN" dbms set-initial-password "$PASSWORD" || true

brew services start neo4j

echo "Waiting for Neo4j Bolt (7687) ..."
for _ in $(seq 1 60); do
  if nc -z localhost 7687 2>/dev/null; then
    echo "Neo4j is up. Browser: http://localhost:7474  user: neo4j  password: $PASSWORD"
    exit 0
  fi
  sleep 1
done

echo "Neo4j did not come up in time; check 'brew services list'." >&2
exit 1
