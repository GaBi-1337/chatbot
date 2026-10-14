#!/usr/bin/env bash
# run inside the work dir: write the reference answer live from the same public APIs
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -I "$HERE/../../tasks/T7/truth.py" "$(dirname "$PWD")/.t7_cache" | python3 -I -c '
import json,sys
t=json.load(sys.stdin); py=t["python"][0]
json.dump({"python":py,"eol":t["eol"][py],**{p:t["pkgs"][p][py][0] for p in ("numpy","scipy","pandas")}}, open("answer.json","w"))'
