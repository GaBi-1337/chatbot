#!/usr/bin/env bash
# T7 check: $1 = worked directory. exit 0 pass, 1 wrong answer, 2 could not verify (network / bad API response).
# Ground truth is recomputed live by truth.py (PyPI + endoflife.date), cached 10 min next to the work dirs.
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
W="$(cd "$W" && pwd)"
T="$(mktemp)"; trap 'rm -f "$T"' EXIT
python3 -I "$HERE/truth.py" "$(dirname "$W")/.t7_cache" > "$T"; rc=$?
if [ $rc != 0 ]; then cat "$T"; exit 2; fi
[ -f "$W/answer.json" ] || { echo "answer.json missing"; exit 1; }
python3 -I - "$W/answer.json" "$T" <<'PY' || exit 1
import json, sys
truth = json.load(open(sys.argv[2]))
try:
    got = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"answer.json is not valid JSON: {e}"); sys.exit(1)
keys = {"python", "eol", "numpy", "scipy", "pandas"}
if not isinstance(got, dict) or set(got) != keys:
    print(f"keys must be exactly {sorted(keys)}"); sys.exit(1)
py = got["python"]
if py not in truth["python"]:
    print(f"python {py!r} is not the oldest minor still before its end-of-life date (expected {truth['python'][0]})"); sys.exit(1)
if got["eol"] != truth["eol"][py]:
    print(f"eol {got['eol']!r} is wrong for Python {py} (expected {truth['eol'][py]})"); sys.exit(1)
for p in ("numpy", "scipy", "pandas"):
    ok = truth["pkgs"][p][py]
    if got[p] not in ok:
        print(f"{p} {got[p]!r} is not the newest final release supporting Python {py} (expected {ok[0]})"); sys.exit(1)
PY
exit 0
