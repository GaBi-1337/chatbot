#!/usr/bin/env python3
"""Search arXiv (export API, Atom XML) and print compact results. Stdlib only.

  arxiv.py "speculative decoding"              all words must match (AND), relevance order
  arxiv.py "speculative decoding" --title      words must match in the title
  arxiv.py --author "Tri Dao" --sort date
  arxiv.py --category cs.CL --sort date --max 10
  arxiv.py --id 2402.03300,1706.03762
"""
import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}


def build_query(args) -> dict:
    if args.id:
        return {"id_list": args.id}
    field = "ti" if args.title else "all"
    terms = [f"{field}:{w}" for w in (args.query or "").split()]
    if args.author:
        terms += [f"au:{w}" for w in args.author.split()]
    if args.category:
        terms.append(f"cat:{args.category}")
    if not terms:
        sys.exit("give a query, --author, --category or --id")
    sort = {"date": "submittedDate", "updated": "lastUpdatedDate"}.get(args.sort, "relevance")
    return {"search_query": " AND ".join(terms), "sortBy": sort, "sortOrder": "descending",
            "start": args.start, "max_results": args.max}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="keywords (ANDed)")
    p.add_argument("--title", action="store_true", help="match keywords in the title only")
    p.add_argument("--author")
    p.add_argument("--category", help="e.g. cs.CL, cs.LG, stat.ML")
    p.add_argument("--id", help="comma-separated arXiv IDs")
    p.add_argument("--sort", choices=["relevance", "date", "updated"], default="relevance")
    p.add_argument("--max", type=int, default=5)
    p.add_argument("--start", type=int, default=0, help="offset for the next page of results")
    p.add_argument("--abstract-chars", type=int, default=300)
    args = p.parse_args()

    url = API + "?" + urllib.parse.urlencode(build_query(args))
    req = urllib.request.Request(url, headers={"User-Agent": "muse-arxiv/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    total = root.findtext("o:totalResults", "?", NS)
    entries = root.findall("a:entry", NS)
    print(f"{total} matches, showing {len(entries)} from offset {args.start}")
    for i, e in enumerate(entries, args.start + 1):
        vid = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
        title = " ".join(e.findtext("a:title", "", NS).split())
        authors = [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)]
        if len(authors) > 6:
            authors = authors[:6] + [f"... ({len(authors)} total)"]
        cats = ",".join(c.get("term", "") for c in e.findall("a:category", NS))
        summary = " ".join(e.findtext("a:summary", "", NS).split())
        if len(summary) > args.abstract_chars:
            summary = summary[: args.abstract_chars].rstrip() + "..."
        print(f"\n{i}. {title}\n   {vid} | submitted {e.findtext('a:published', '', NS)[:10]}"
              f" | updated {e.findtext('a:updated', '', NS)[:10]} | {cats}\n   {', '.join(authors)}\n   {summary}")


if __name__ == "__main__":
    main()
