#!/usr/bin/env bash
# Repo checks, run by the pre-commit hook (.githooks/pre-commit). No GPU needed.
# Enable once per clone: git config core.hooksPath .githooks
# evals/selftest.sh queries live APIs for the web tasks; when offline, commit with --no-verify.
set -u
cd "$(dirname "$0")/.." || exit 1
# Needs shellcheck, pytest and PyYAML: an active venv, else $MUSE_VENV (default ~/.virtualenvs/env).
V="${MUSE_VENV:-$HOME/.virtualenvs/env}"
[ -n "${VIRTUAL_ENV:-}" ] || [ ! -f "$V/bin/activate" ] || source "$V/bin/activate"
rc=0
step() { echo "== $1"; shift; "$@" || rc=1; }
step shellcheck shellcheck -S warning muse server/serve.sh container/build.sh scripts/check.sh \
  evals/run.sh evals/selftest.sh evals/tasks/lib.sh evals/tasks/*/check.sh .claude/skills/muse-live-test/scripts/cap-run.sh
# Scripts run directly must be executable in git; a clone on a drive without Unix permissions (core.fileMode false)
# commits them as 100644 without noticing.
step "executable bits" bash -c 'bad=$(git ls-files -s -- "*.sh" muse ".githooks/*" | awk "\$1 != \"100755\" {print \$4}");
  [ -z "$bad" ] || { printf "not executable in git (fix: git update-index --chmod=+x FILE):\n%s\n" "$bad"; exit 1; }'
step "mcp/web tests" bash -c 'cd mcp/web && .venv/bin/python -m pytest -q'
step "harness tests" python -m pytest -q -p no:cacheprovider harness/tests
[ "${SKIP_SELFTEST:-0}" = 1 ] || step "eval selftest" evals/selftest.sh
[ $rc = 0 ] && echo "CHECKS PASS" || echo "CHECKS FAIL"
exit $rc
