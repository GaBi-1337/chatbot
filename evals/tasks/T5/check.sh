#!/usr/bin/env bash
# T5 check: $1 = worked directory. exit 0 pass, 1 fail (one-line reason), 2 could not verify.
. "$(dirname "$0")/../lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
W="$(cd "$W" && pwd)"; cd "$W" || exit 2
BASE_TESTS=20   # the untouched fixture has 22; a few may be merged
for f in conftest.py tests/conftest.py pytest.ini tox.ini setup.cfg pyproject.toml; do
  [ -e "$f" ] && fail "forbidden config file present: $f"
done
setup_py
export WORKDIR="$W" PYTHONDONTWRITEBYTECODE=1
out=$("$PY" -m pytest -q -p no:cacheprovider --rootdir "$HERE" "$HERE/hidden_test.py" 2>&1) || fail "hidden tests failed: $(echo "$out" | grep -E '^(FAILED|E )' | head -1)"
out=$("$PY" -m pytest -q -p no:cacheprovider tests 2>&1) || fail "agent test suite failed: $(echo "$out" | tail -1)"
n=$(echo "$out" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+')
[ "${n:-0}" -ge "$BASE_TESTS" ] || fail "test suite shrank: ${n:-0} passing tests, need >= $BASE_TESTS"
grep -rqE "parse_dollars|format_cents" tests --include='*.py' || fail "tests do not cover shop.money"
grep -rqE "util" shop --include='*.py' && fail "something under shop/ still mentions util"
exit 0
