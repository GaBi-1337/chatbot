---
name: web-research
description: "Web research with cited sources; pages that won't fetch."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [research, web, search, citations, sources, archive]
    category: research
    related_skills: [arxiv-papers, youtube-transcript]
---

# Web research

`mcp__internet__web_search` finds URLs; `mcp__internet__web_fetch` reads them. A search snippet is a
lead, not a source: cite only pages you fetched and read.

## Find

- 3-6 distinctive keywords. DuckDuckGo honours `"exact phrase"`, `site:docs.python.org` and `-word`.
  Try two or three phrasings before concluding something doesn't exist.
- Go to the primary source (official docs, changelog or release notes, the project's source code,
  the standard, the paper); use articles and forums to find it or for experience reports.
- For anything version- or time-sensitive (APIs, flags, prices, "latest"), note the page's date or
  version and prefer the newest primary page.
- JSON APIs beat HTML pages. Query them with `curl -s URL | jq '<fields>'` in the terminal so only
  the fields you need enter the context (web_fetch returns the raw JSON, 20,000 characters at a time):

  | Need | URL |
  |---|---|
  | PyPI package (versions, deps) | `https://pypi.org/pypi/<pkg>/json` |
  | npm package | `https://registry.npmjs.org/<pkg>/latest` |
  | GitHub release, issues | `https://api.github.com/repos/<owner>/<repo>/releases/latest`, `https://api.github.com/search/issues?q=repo:<owner>/<repo>+<words>` |
  | Wikipedia summary | `https://en.wikipedia.org/api/rest_v1/page/summary/<Title>` |
  | arXiv papers | load the `arxiv-papers` skill |

  GitHub allows 60 unauthenticated API requests per hour. A single file reads best from
  `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path>`; for more than a few files,
  `git clone --depth 1` into /tmp and search it locally.

## Read

- Check relevance cheaply first (`max_chars=4000`), then read the part you need; a long page ends
  with the `start_index` to continue from.
- PDFs arrive with `--- page N ---` markers: cite page numbers.
- Empty or navigation-only text from a JavaScript site: fetch again with `render=True`.
- Charts, figures and diagrams carry facts the text omits: fetch the image URL (you get a file path)
  or `mcp__internet__web_screenshot` the page, then `vision_analyze` it with a specific question.

## When a page won't fetch

Work down this list and stop at the first copy that contains the content you expect (check for
title words: rate-limit and "Just a moment..." pages also come back as text):

1. `render=True`, for bot walls and JavaScript shells.
2. The Wayback Machine: `https://web.archive.org/web/<full url>` redirects to the newest snapshot;
   `https://archive.org/wayback/available?url=<full url>` names it with a timestamp. Cite it as
   "archived <date>"; for current data (prices, status, news) say the copy may be stale.
3. The same content elsewhere: the site's RSS feed (`/feed`, `/rss`), `sitemap.xml`, a JSON endpoint
   on the same host, the docs' source repository on GitHub, a press release or mirror.

Report a page you could not read as unread; answer from other sources or say what is missing.

## Cite

- Put the source link right after the claims it supports (`[title](url)`), or number the sources
  `[1]` and list them at the end; long answers and written documents use the numbered form.
- Copy numbers, dates, names and version strings exactly as the source states them.
- When sources disagree, give each reading with its source and date.
