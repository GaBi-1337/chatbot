#!/usr/bin/env bash
# run inside the work dir: produce the reference answer live (newest entry of the releases list)
curl -s -m 30 -H "User-Agent: eval-ref" "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
 | python3 -I -c 'import json,sys; print(json.load(sys.stdin)[0]["tag_name"])' > answer.txt
