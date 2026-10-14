# web MCP server

Stdio MCP server giving an agent `web_search` (DuckDuckGo via ddgs, retry/backoff on empty results),
`web_fetch` (httpx + trafilatura -> markdown; text/JSON passthrough; PDF -> text with `--- page N ---` markers
via pypdf; png/jpeg/gif/webp saved to a temp file whose path is returned; other binaries refused; paging via
`start_index`; `render=true`, or automatically when a page has no static text, renders it in headless Chromium)
and `web_screenshot` (PNG of an http(s)/file:///localhost page, for the model's `vision_analyze`).

Chromium: the Debian `chromium-headless-shell` package (binary found on PATH, or `WEB_MCP_CHROMIUM`). Without a
browser the tools return an error message and everything else keeps working.

Install (Debian/Ubuntu python3, inside a venv or image):

    pip install -r requirements.txt
    python3 /path/to/mcp/web/server.py        # speaks MCP over stdio; only protocol on stdout

Env (all optional): `WEB_MCP_TIMEOUT` (20 s), `WEB_MCP_MAX_BYTES` (3000000 download cap),
`WEB_MCP_SEARCH_RETRIES` (4), `WEB_MCP_BACKOFF` (1.5 s, doubles), `WEB_MCP_USER_AGENT`,
`WEB_MCP_MAX_PDF_BYTES` (30000000), `WEB_MCP_MAX_IMAGE_BYTES` (15000000), `WEB_MCP_CHROMIUM` (browser path/name),
`WEB_MCP_RENDER_MS` (5000, virtual-time budget for scripts), `WEB_MCP_BROWSER_TIMEOUT` (40 s hard limit).
Downloaded images and screenshots go under the system temp dir (`TMPDIR`).

Tests: `pip install -r requirements-dev.txt && pytest tests` (no network).
Live smoke: `WEB_MCP_LIVE=1 pytest tests -k live`.

Paging: a call with `start_index` > 0 reuses the text of the first call for the same URL (last 8 documents,
30 minutes), so pages don't shift and PDFs aren't parsed again; `start_index=0` always fetches afresh.

Limits: scanned (image-only) PDFs have no text; no SSRF filter (the fetcher can reach localhost/LAN);
DuckDuckGo may throttle (search returns a "No results ... retry later" message after retries).
