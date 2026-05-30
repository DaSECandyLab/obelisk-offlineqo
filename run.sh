#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG="${CONFIG:-etc/obelisk.toml}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG}. Copy etc/obelisk.toml.tpl to etc/obelisk.toml and fill local secrets." >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

cmd=(
  "${PYTHON_BIN}" src/run.py
  --config "${CONFIG}"
)

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

"${cmd[@]}"
