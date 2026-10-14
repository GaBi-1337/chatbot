#!/usr/bin/env bash
# T2 check: $1 = worked directory. exit 0 pass, 1 fail (one-line reason), 2 could not verify.
. "$(dirname "$0")/../lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
W="$(cd "$W" && pwd)"; cd "$W" || exit 2
BASE_TESTS=4   # tests in the untouched fixture
setup_py
export WORKDIR="$W"
# hidden behaviour tests (live outside the workdir)
out=$("$PY" -m pytest -q -p no:cacheprovider --rootdir "$HERE" "$HERE/hidden_test.py" 2>&1) || fail "hidden tests failed: $(echo "$out" | grep -E '^(FAILED|E )' | head -1)"
# agent's own tests: whole suite passes, count grew, and one mentions exclude
out=$("$PY" -m pytest -q -p no:cacheprovider tests 2>&1) || fail "agent test suite failed: $(echo "$out" | tail -1)"
n=$(echo "$out" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+')
[ "${n:-0}" -gt "$BASE_TESTS" ] || fail "no new test added (found ${n:-0} passing tests, need > $BASE_TESTS)"
grep -rqi "exclude" tests --include='*.py' || fail "no test mentions exclude"
exit 0
