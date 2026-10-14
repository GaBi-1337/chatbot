#!/usr/bin/env bash
# T3 check: $1 = worked directory. exit 0 pass, 1 fail (one-line reason), 2 could not verify.
. "$(dirname "$0")/../lib.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
[ -f "$W/results.json" ] || fail "results.json missing"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
python3 -I "$HERE/gen.py" --truth > "$T/truth.json" || { echo "could not verify: generator failed"; exit 2; }
# data/ must be untouched: compare with a regeneration
python3 -I "$HERE/gen.py" "$T/regen" && diff -rq "$T/regen/data" "$W/data" >/dev/null || fail "data/ was modified"
python3 -I - "$W/results.json" "$T/truth.json" <<'PY' || exit 1
import json, sys
try:
    got = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"results.json is not valid JSON: {e}"); sys.exit(1)
want = json.load(open(sys.argv[2]))
if not isinstance(got, dict) or set(got) != set(want):
    print(f"keys must be exactly {sorted(want)}, got {sorted(got) if isinstance(got, dict) else type(got).__name__}"); sys.exit(1)
for k, v in want.items():
    g = got[k]
    if isinstance(v, int) and (isinstance(g, bool) or not isinstance(g, int)) or g != v:
        s = repr(g) if len(repr(g)) < 80 else repr(g)[:80] + "..."
        print(f"wrong value for {k}: got {s}"); sys.exit(1)
PY
exit 0
