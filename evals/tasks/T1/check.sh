#!/usr/bin/env bash
# T1 check: $1 = worked directory. exit 0 pass, 1 fail (one-line reason), 2 could not verify.
. "$(dirname "$0")/../lib.sh"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
cd "$W" || exit 2
EXPECT="da607086efc59b2415829adb3e61f719503188747c441d990aa1d730c34a2462  tests/test_intervals.py
008ac7d9a120676ca56d3c1ba1e16b8ba460080071f280b72d75493714a24013  tests/test_slots.py"
GOT=$(sha256sum tests/test_intervals.py tests/test_slots.py 2>/dev/null)
[ "$GOT" = "$EXPECT" ] || fail "test files were modified or missing"
for f in conftest.py tests/conftest.py pytest.ini tox.ini setup.cfg pyproject.toml; do
  [ -e "$f" ] && fail "forbidden config file present: $f"
done
setup_py
out=$("$PY" -m pytest -q -p no:cacheprovider tests 2>&1) || fail "pytest failed: $(echo "$out" | tail -1)"
echo "$out" | grep -q "passed" || fail "no tests passed"
exit 0
