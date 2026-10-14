#!/usr/bin/env bash
# T4 check: $1 = worked directory. exit 0 pass, 1 wrong answer, 2 could not verify (network/rate limit).
# Accepts any of the newest 5 entries of the releases list, plus the release GitHub marks "Latest".
W="${1:?usage: check.sh WORKDIR}"; [ -d "$W" ] || { echo "could not verify: no workdir"; exit 2; }
W="$(cd "$W" && pwd)"
CACHE="$(dirname "$W")/.llamacpp_releases_cache.json"   # 10 min cache, shared by runs
LCACHE="$CACHE.latest"
API=https://api.github.com/repos/ggml-org/llama.cpp
hdr=(-H "Accept: application/vnd.github+json" -H "User-Agent: eval-check")
[ -n "$GITHUB_TOKEN" ] && hdr+=(-H "Authorization: Bearer $GITHUB_TOKEN")
fresh() { [ -s "$1" ] && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -lt 600 ]; }
if ! fresh "$CACHE"; then
  code=$(curl -s -m 30 -o "$CACHE.$$" -w '%{http_code}' "${hdr[@]}" "$API/releases?per_page=5")
  if [ "$code" != 200 ]; then rm -f "$CACHE.$$"; echo "could not verify: GitHub API returned HTTP $code"; exit 2; fi
  mv "$CACHE.$$" "$CACHE"
fi
fresh "$LCACHE" || { curl -s -m 30 -f -o "$LCACHE.$$" "${hdr[@]}" "$API/releases/latest" && mv "$LCACHE.$$" "$LCACHE" || rm -f "$LCACHE.$$"; }
TAGS=$(python3 -I -c 'import json,sys
d=json.load(open(sys.argv[1]))
print("\n".join(r["tag_name"] for r in d[:5]))' "$CACHE") || { rm -f "$CACHE"; echo "could not verify: bad API response"; exit 2; }
[ -n "$TAGS" ] || { echo "could not verify: no releases returned"; exit 2; }
if [ -s "$LCACHE" ]; then
  EXTRA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("tag_name",""))' "$LCACHE" 2>/dev/null)
  TAGS="$TAGS"$'\n'"$EXTRA"
fi
[ -f "$W/answer.txt" ] || { echo "answer.txt missing"; exit 1; }
[ "$(grep -c . "$W/answer.txt")" -le 1 ] || { echo "answer.txt must contain a single line"; exit 1; }
ANS=$(tr -d '[:space:]' < "$W/answer.txt")
[ -n "$ANS" ] && echo "$TAGS" | grep -qxF -- "$ANS" || { echo "answer '$ANS' is not among the latest 5 release tags"; exit 1; }
exit 0
