---
name: arxiv-papers
description: "arXiv papers: search, read, cite, count citations."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [arxiv, papers, research, citations, bibtex]
    category: research
    related_skills: [web-research, pdf]
---

# arXiv papers

## Search

```bash
python ${HERMES_SKILL_DIR}/scripts/arxiv.py "speculative decoding"            # all words must match
python ${HERMES_SKILL_DIR}/scripts/arxiv.py "speculative decoding" --title --sort date --max 10
python ${HERMES_SKILL_DIR}/scripts/arxiv.py --author "Tri Dao" --category cs.LG --sort date
python ${HERMES_SKILL_DIR}/scripts/arxiv.py --id 2402.03300,1706.03762
```

`--start N` pages on. Keep to one request every 3 seconds (arXiv's limit).

For queries the script can't express, call the API directly:
`curl -s "https://export.arxiv.org/api/query?search_query=QUERY&max_results=5"` (Atom XML).
Field prefixes `ti: au: abs: cat: all:`, operators `AND OR ANDNOT`, a phrase as `%22two+words%22`.
Join words with `+AND+`: arXiv ORs bare space-separated words, so `all:speculative+decoding`
sorted by date returns the newest papers that mention either word.

## Read

- Full text: `mcp__internet__web_fetch("https://arxiv.org/html/<id>")` gives clean text with section
  headings (most recent papers have one). If that page is missing, fetch `https://arxiv.org/pdf/<id>`
  (text with `--- page N ---` markers). Read the abstract and the sections you need, paging with
  `start_index`.
- Figures and tables that come through as noise: `curl -sL -o /tmp/p.pdf https://arxiv.org/pdf/<id>`,
  then `pdftoppm -f N -l N -r 100 -png -singlefile /tmp/p.pdf /tmp/page` and `vision_analyze` /tmp/page.png.
- The bare ID resolves to the latest version. Cite the version you read (`2402.03300v2`); the
  script prints it.
- A withdrawn paper says so in its abstract.

## Cite

- BibTeX: `curl -s https://arxiv.org/bibtex/<id>`.
- Citation count and citing papers (OpenAlex, free, JSON; arXiv DOI is `10.48550/arXiv.<id>`):
  ```bash
  curl -s "https://api.openalex.org/works/doi:10.48550/arXiv.1706.03762?select=id,display_name,cited_by_count"
  curl -s "https://api.openalex.org/works?filter=cites:<W... part of that id>&sort=cited_by_count:desc&per_page=10&select=display_name,publication_year,doi"
  ```
  This counts citations of the arXiv record only; a paper also published at a venue has a second
  record (search `filter=title.search:<title words>`). State which count you report.
- Semantic Scholar (`api.semanticscholar.org/graph/v1/paper/arXiv:<id>`) often answers 429 without a
  key; use OpenAlex unless the user has a key.
