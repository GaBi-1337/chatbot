#!/usr/bin/env bash
# Repo checks, run by the pre-commit hook (.githooks/pre-commit). No GPU needed.
# Enable once per clone: git config core.hooksPath .githooks
# evals/selftest.sh queries live APIs for the web tasks; when offline, commit with --no-verify.
set -u
cd "$(dirname "$0")/.." || exit 1
# Needs shellcheck and pytest: an active venv, else $MUSE_VENV (default ~/.virtualenvs/env).
V="${MUSE_VENV:-$HOME/.virtualenvs/env}"
[ -n "${VIRTUAL_ENV:-}" ] || [ ! -f "$V/bin/activate" ] || source "$V/bin/activate"
rc=0
step() { echo "== $1"; shift; "$@" || rc=1; }
step shellcheck shellcheck -S warning muse server/serve.sh container/build.sh scripts/check.sh \
  evals/run.sh evals/selftest.sh evals/tasks/lib.sh evals/tasks/*/check.sh
step "mcp/web tests" bash -c 'cd mcp/web && .venv/bin/python -m pytest -q'
[ "${SKIP_SELFTEST:-0}" = 1 ] || step "eval selftest" evals/selftest.sh
[ $rc = 0 ] && echo "CHECKS PASS" || echo "CHECKS FAIL"
exit $rc
