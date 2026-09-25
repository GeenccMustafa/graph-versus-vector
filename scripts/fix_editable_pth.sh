#!/usr/bin/env bash
# macOS-only fix for: "ModuleNotFoundError: No module named 'rag_compare'"
#
# uv's editable install writes .venv/lib/pythonX.Y/site-packages/rag_compare.pth
# with the macOS UF_HIDDEN flag set, and CPython's `site` module skips hidden
# .pth files -> the src/ directory is never added to sys.path.
#
# Either run this script, or install non-editable (`uv sync --no-editable`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SP="$(ls -d "$ROOT"/.venv/lib/python*/site-packages 2>/dev/null | head -n 1 || true)"

if [[ -z "${SP}" || ! -d "${SP}" ]]; then
  echo "Could not find the venv site-packages under ${ROOT}/.venv" >&2
  exit 1
fi

echo "Clearing the hidden flag on *.pth in ${SP}"
chflags nohidden "${SP}"/*.pth 2>/dev/null || true
ls -lO "${SP}"/*.pth 2>/dev/null || true
echo "Done. Re-run your command (no need to reinstall)."
