#!/bin/bash
# serve.sh start|stop|status|restart [--no-vision] [--no-draft]
# Tuned llama-server for Muse Glimmer 30B (README.md, Measured performance, has the numbers behind the flags).
# pid / log / flags live in run/ (gitignored). Idempotent: `start` reuses a healthy server that runs
# the same flags; it restarts one that runs different flags (e.g. after a flag change here); it fails loudly
# if the server exits while loading.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/run"
PIDF="$RUN/server.pid"
LOGF="$RUN/llama-server.log"
FLAGF="$RUN/server.flags"
BIN="${LLAMA_SERVER:-$ROOT/tools/llama.cpp/build/bin/llama-server}"
HOST=127.0.0.1
PORT="${MUSE_PORT:-8080}"
WAIT="${MUSE_WAIT:-900}"          # seconds to wait for /health (loading takes a few minutes)
MODELS="$ROOT/Models/MuseGlimmer"

cmd="${1:-status}"; shift || true
VISION=1; DRAFT=1   # vision (mmproj, +1.6 GB VRAM) is on by default
for a in "$@"; do
  case "$a" in
    --vision)   VISION=1 ;;   # default; kept for old invocations
    --no-vision) VISION=0 ;;
    --no-draft) DRAFT=0 ;;
    *) echo "serve.sh: unknown option $a" >&2; exit 2 ;;
  esac
done

healthy() { curl -sf --max-time 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; }
alive()   { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

build_args() {
  ARGS=( -m "$MODELS/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" -a muse-glimmer-30B
         -ngl 99 -c 131072 -np 1 -fa on --jinja
         --temp 1.0 --top-p 0.95 --top-k 64
         --cache-ram 3072 -ctxcp 8
         --host "$HOST" --port "$PORT" )
  ARGS+=( -lm dio )   # direct I/O load: the fastest -lm mode here, no page-cache residue
  # z-lab DFlash2 drafter: +8% decode over the original DFlash GGUF, same prefill (README.md, Engine bake-off)
  if [ "$DRAFT" = 1 ];  then ARGS+=( -md "$MODELS/Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf" -ngld 99 --spec-draft-n-max 15 ); fi
  if [ "$VISION" = 1 ]; then ARGS+=( --mmproj "$MODELS/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf" ); fi
}

# Inference slows down a lot when the GPU runs well below its maximum power limit (power-saving modes, a weak
# power supply): warn when the enforced limit is under 75% of the maximum.
power_check() {
  local lim max
  IFS=', ' read -r lim max < <(nvidia-smi --query-gpu=enforced.power.limit,power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1)
  lim="${lim%%.*}"; max="${max%%.*}"
  [[ "$lim" =~ ^[0-9]+$ && "$max" =~ ^[0-9]+$ ]] || return 0   # [N/A] on some GPUs
  if [ "$lim" -lt $(( max * 3 / 4 )) ]; then
    echo "serve.sh: note: the GPU power limit is ${lim} W of a possible ${max} W, so inference is slower" \
         "(power-saving modes and weaker power supplies lower it)." >&2
  fi
  return 0
}

do_stop() {
  if alive; then
    local p; p="$(cat "$PIDF")"
    kill "$p" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; sleep 2; }
    echo "stopped pid $p"
  else
    echo "not running"
  fi
  rm -f "$PIDF" "$FLAGF"
  # wait for the driver to release VRAM
  for _ in $(seq 1 20); do
    u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
    [ -z "$u" ] || [ "$u" -lt 1500 ] && break; sleep 1
  done
  [ -n "${u:-}" ] && echo "VRAM in use: ${u} MiB"
  return 0
}

do_start() {
  mkdir -p "$RUN"
  power_check
  build_args
  local want="${ARGS[*]}"
  if alive && healthy; then
    if [ "$(cat "$FLAGF" 2>/dev/null)" = "$want" ]; then
      echo "already running (pid $(cat "$PIDF")), healthy on http://$HOST:$PORT"; return 0
    fi
    echo "running with different flags -> restarting"; do_stop
  elif alive; then
    echo "pid $(cat "$PIDF") alive but not healthy -> restarting"; do_stop
  elif healthy; then
    echo "serve.sh: something else already answers on $HOST:$PORT (not started by serve.sh); refusing" >&2; return 1
  fi
  [ -x "$BIN" ] || { echo "serve.sh: llama-server not found at $BIN" >&2; return 1; }
  [ -f "${ARGS[1]}" ] || { echo "serve.sh: model missing: ${ARGS[1]}" >&2; return 1; }
  echo "$want" > "$FLAGF"
  : > "$LOGF"
  nohup "$BIN" "${ARGS[@]}" > "$LOGF" 2>&1 &
  echo $! > "$PIDF"
  local t0; t0=$(date +%s)
  echo "starting llama-server pid $(cat "$PIDF") (log: $LOGF) ..."
  while :; do
    if healthy; then echo "ready after $(( $(date +%s)-t0 ))s on http://$HOST:$PORT"; return 0; fi
    if ! alive; then
      echo "serve.sh: llama-server exited during startup; last log lines:" >&2
      tail -25 "$LOGF" >&2; rm -f "$PIDF" "$FLAGF"; return 1
    fi
    if [ $(( $(date +%s)-t0 )) -gt "$WAIT" ]; then
      echo "serve.sh: not healthy after ${WAIT}s; see $LOGF" >&2; return 1
    fi
    sleep 2
  done
}

do_status() {
  power_check
  if alive && healthy; then
    echo "running: pid $(cat "$PIDF"), healthy on http://$HOST:$PORT"
    echo "flags: $(cat "$FLAGF" 2>/dev/null)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/VRAM: /'
    return 0
  elif alive; then echo "starting or unhealthy: pid $(cat "$PIDF") (see $LOGF)"; return 1
  elif healthy; then echo "a server answers on $HOST:$PORT but was not started by serve.sh"; return 1
  else echo "stopped"; return 3; fi
}

case "$cmd" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  *) echo "usage: serve.sh start|stop|restart|status [--no-vision] [--no-draft]" >&2; exit 2 ;;
esac
