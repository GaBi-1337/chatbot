#!/usr/bin/env bash
# evals/run.sh - run eval tasks headless through Hermes (muse-agent image) against the local Muse server.
#
#   evals/run.sh [-n RUNS] [-t TIMEOUT_S] [-r REASONING] [-m MAX_TURNS] [-o OUT_DIR] [--upstream PORT] [TASK...]
#     defaults: -n 3, -t 1200, -r high, -m 150, all of evals/tasks/T*, upstream 8080 (the llama-server)
#
# Per run: fresh fixture copy (OUT/work/<task>-r<n>, siblings under one parent, as check.sh expects),
# fresh HERMES_HOME from harness/hermes (OUT/<task>-r<n>/home, memory off), a logging proxy between Hermes
# and the server (OUT/<task>-r<n>/proxy.jsonl: requests, tool calls, tokens, cache_n), hermes stream-json
# events (stream.jsonl), check.sh verdict (check.log). Rows go to OUT/results.jsonl, the table to OUT/summary.md.
# Runs are sequential (single server slot). The server must be up (server/serve.sh start) unless --start.
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PY="$(command -v python3)"   # stdlib only (proxy, analysis)
IMAGE="${MUSE_IMAGE:-localhost/muse-agent:latest}"
RUNS=3; TIMEOUT=1200; REASONING=high; MAXTURNS=150; UPSTREAM=8080; OUT=""; START=0; TASKS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -n) RUNS="$2"; shift 2 ;;
    -t) TIMEOUT="$2"; shift 2 ;;
    -r) REASONING="$2"; shift 2 ;;
    -m) MAXTURNS="$2"; shift 2 ;;
    -o) OUT="$2"; shift 2 ;;
    --upstream) UPSTREAM="$2"; shift 2 ;;
    --start) START=1; shift ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "run.sh: unknown option $1" >&2; exit 2 ;;
    *) TASKS+=("$1"); shift ;;
  esac
done
[ ${#TASKS[@]} -gt 0 ] || for d in "$ROOT"/evals/tasks/T*/; do TASKS+=("$(basename "$d")"); done
OUT="${OUT:-$ROOT/evals/results/$(date +%Y%m%d-%H%M%S)}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac   # podman mounts need absolute paths
mkdir -p "$OUT/work"

podman image exists "$IMAGE" || { echo "run.sh: image $IMAGE missing (container/build.sh)" >&2; exit 1; }
if ! curl -sf --max-time 3 "http://127.0.0.1:$UPSTREAM/health" >/dev/null; then
  if [ "$START" = 1 ]; then "$ROOT/server/serve.sh" start || exit 1
  else echo "run.sh: no healthy server on 127.0.0.1:$UPSTREAM (start it: server/serve.sh start, or pass --start)" >&2; exit 1; fi
fi

free_port() { "$PY" -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }

{
  echo "date: $(date -Is)"; echo "tasks: ${TASKS[*]}  runs: $RUNS  timeout: ${TIMEOUT}s  reasoning: $REASONING  max_turns: $MAXTURNS"
  echo "image: $(podman image inspect "$IMAGE" --format '{{.Id}} {{.Created}}' 2>/dev/null)"
  echo "server: $(cat "$ROOT/run/server.flags" 2>/dev/null || echo unknown)"
  curl -sf --max-time 3 "http://127.0.0.1:$UPSTREAM/props" | "$PY" -I -c 'import json,sys; d=json.load(sys.stdin); print("props:", json.dumps({k: d.get(k) for k in ("model_alias","build_info","total_slots")}))' 2>/dev/null
} > "$OUT/meta.txt"
cp "$ROOT/harness/hermes/config.yaml" "$OUT/hermes-config.yaml"

for T in "${TASKS[@]}"; do
  TD="$ROOT/evals/tasks/$T"
  [ -d "$TD/fixture" ] && [ -f "$TD/prompt.txt" ] && [ -x "$TD/check.sh" ] || { echo "run.sh: skipping $T (needs fixture/, prompt.txt, executable check.sh)" >&2; continue; }
  for i in $(seq 1 "$RUNS"); do
    RID="$T-r$i"; RD="$OUT/$RID"; W="$OUT/work/$RID"
    rm -rf "$RD" "$W"; mkdir -p "$RD" "$W"
    cp -a "$TD/fixture/." "$W/"
    cp -a "$ROOT/harness/hermes/." "$RD/home/"
    PPORT="$(free_port)"
    # eval profile: through the logging proxy, no cross-session memory
    sed -i -e "s#http://127.0.0.1:8080/v1#http://127.0.0.1:$PPORT/v1#" \
           -e 's/^\(  memory_enabled:\) true/\1 false/' -e 's/^\(  user_profile_enabled:\) true/\1 false/' "$RD/home/config.yaml"
    "$PY" -I "$ROOT/evals/lib/log_proxy.py" "$PPORT" "$UPSTREAM" "$RD/proxy.jsonl" 2> "$RD/proxy.err" &
    PX=$!
    for _ in $(seq 50); do (exec 3<>"/dev/tcp/127.0.0.1/$PPORT") 2>/dev/null && break; sleep 0.1; done
    NAME="muse-eval-$RID-$$"
    echo "== $RID  (work: $W)"
    t0=$(date +%s.%N)
    timeout -k 30 "$TIMEOUT" podman run --rm -i --init --name "$NAME" --label muse.eval=1 \
        --network=host --userns=keep-id \
        -v "$W:$W" -w "$W" -v "$RD/home:/muse/hermes" -e HERMES_HOME=/muse/hermes \
        -v "$ROOT/harness/skills:/muse/skills:ro" \
        -v "$ROOT/harness/hooks:/muse/hooks:ro" \
        --mount type=tmpfs,destination=/muse/hermes/cache/scratch,tmpfs-size=2g,chown=true \
        "$IMAGE" hermes chat --query-file - --oneshot --yolo --format stream-json --source tool \
          --reasoning "$REASONING" --max-turns "$MAXTURNS" --run-budget "$(( TIMEOUT > 120 ? TIMEOUT - 60 : TIMEOUT ))" \
        < "$TD/prompt.txt" > "$RD/stream.jsonl" 2> "$RD/stderr.log"
    rc=$?
    t1=$(date +%s.%N)
    podman kill "$NAME" >/dev/null 2>&1; podman rm -f "$NAME" >/dev/null 2>&1
    kill "$PX" 2>/dev/null; wait "$PX" 2>/dev/null
    "$TD/check.sh" "$W" > "$RD/check.log" 2>&1; ck=$?
    wall=$("$PY" -I -c "print(round($t1-$t0,1))")
    "$PY" -I "$ROOT/evals/lib/collect.py" row "$RD" "$T" "$i" "$rc" "$ck" "$wall" >> "$OUT/results.jsonl"
    tail -1 "$OUT/results.jsonl" | "$PY" -I "$ROOT/evals/lib/collect.py" line
  done
done

[ -s "$OUT/results.jsonl" ] && "$PY" -I "$ROOT/evals/lib/collect.py" summary "$OUT/results.jsonl" > "$OUT/summary.md" && { echo; cat "$OUT/summary.md"; echo; echo "results: $OUT"; }
