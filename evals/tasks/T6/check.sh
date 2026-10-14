#!/usr/bin/env bash
# T6 check: $1 = worked directory. exit 0 pass, 1 fail (one-line reason), 2 could not verify.
. "$(dirname "$0")/../lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
W="$(cd "$W" && pwd)"; cd "$W" || exit 2
EXPECT="0dfd835da261f27a42c07e38bd5f90028dbd35e7cc9b1632a6a8b4a7c9eaefa7"
GOT=$(find tests -type f ! -path '*__pycache__*' ! -name '*.pyc' | LC_ALL=C sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1)
[ "$GOT" = "$EXPECT" ] || fail "files under tests/ were modified, added or removed"
for f in conftest.py tests/conftest.py pytest.ini tox.ini setup.cfg pyproject.toml; do
  [ -e "$f" ] && fail "forbidden config file present: $f"
done
setup_py
export WORKDIR="$W" PYTHONDONTWRITEBYTECODE=1
out=$("$PY" -m pytest -q -p no:cacheprovider tests 2>&1) || fail "pytest failed: $(echo "$out" | tail -1)"
echo "$out" | grep -q "18 passed" || fail "expected 18 passing tests: $(echo "$out" | tail -1)"
out=$("$PY" -m pytest -q -p no:cacheprovider --rootdir "$HERE" "$HERE/hidden_test.py" 2>&1) || fail "hidden tests failed: $(echo "$out" | grep -E '^(FAILED|E )' | head -1)"
exit 0
