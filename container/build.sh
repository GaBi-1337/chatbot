#!/usr/bin/env bash
# Build image muse-agent from a staged context (only what the image needs), then prune build leftovers.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CTX="$REPO/run/image-ctx"
rm -rf "$CTX"; mkdir -p "$CTX/mcp-web"
cp "$REPO/container/Containerfile" "$CTX/"
# mcp/web code + requirements (no venvs, caches or tests data)
if [ -d "$REPO/mcp/web" ]; then
  (cd "$REPO/mcp/web" && find . -maxdepth 1 -type f \( -name "server.py" -o -name "requirements.txt" \) -exec cp {} "$CTX/mcp-web/" \;)
fi
podman build --layers=false -t muse-agent:latest -f "$CTX/Containerfile" "$CTX"
rm -rf "$CTX"
# drop the build stage and dangling layers
podman image prune -f >/dev/null
podman builder prune -f >/dev/null 2>&1 || true
podman system df
