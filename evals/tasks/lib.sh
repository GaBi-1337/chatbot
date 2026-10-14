# shellcheck shell=bash disable=SC2034  # sourced by check.sh; PY is read by the sourcing script
# shared helpers for check.sh (sourced). Provides PY with pytest available, on the host.
fail() { echo "$*"; exit 1; }
setup_py() {
  if python3 -c 'import pytest' 2>/dev/null; then PY=python3; return; fi
  local v="${MUSE_VENV:-$HOME/.virtualenvs/env}"
  if [ -x "$v/bin/python" ] && "$v/bin/python" -c 'import pytest' 2>/dev/null; then PY="$v/bin/python"; return; fi
  local t="${EVAL_VENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv}"
  if [ ! -x "$t/bin/python" ]; then python3 -m venv "$t" && "$t/bin/pip" -q install pytest >/dev/null 2>&1; fi
  "$t/bin/python" -c 'import pytest' 2>/dev/null || { echo "could not verify: no pytest available"; exit 2; }
  PY="$t/bin/python"
}
