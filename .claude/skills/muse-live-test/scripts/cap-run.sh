#!/usr/bin/env bash
# Run capability prompts headless through the real muse launcher, one fresh git project per prompt.
# usage: cap-run.sh [-s MUSE_STATE] /abs/PROMPT.txt...   (outputs in run/cap/<name>*; default state run/cap/home,
# home folder run/cap/homedir unless MUSE_HOME_DIR is set)
set -u
REPO="$(cd "$(dirname "$(readlink -f "$0")")/../../../.." && pwd)"
R="$REPO/run/cap"; ST="$R/home"
[ "${1:-}" = -s ] && { ST="$2"; shift 2; }
for p in "$@"; do
  n="$(basename "$p" .txt)"; W="$R/$n"; rm -rf "$W"; mkdir -p "$W"
  (cd "$W" && git init -q . && git -c user.email=t@t -c user.name=t commit -q --allow-empty -m init)
  t0=$(date +%s)
  (cd "$W" && MUSE_STATE="$ST" MUSE_HOME_DIR="${MUSE_HOME_DIR:-$R/homedir}" timeout 1200 "$REPO/muse" chat --query-file - --oneshot --yolo \
      --format stream-json --source tool --max-turns 40 < "$p" > "$R/$n.stream.jsonl" 2> "$R/$n.stderr.log")
  echo "$n rc=$? wall=$(( $(date +%s) - t0 ))s"
done
