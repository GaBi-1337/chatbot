#!/usr/bin/env python3
"""T7 ground truth from live public APIs (PyPI JSON, endoflife.date), with a 10 minute file cache.

  truth.py CACHE_DIR [--today YYYY-MM-DD]   prints JSON {"python": [...], "eol": {...}, "pkgs": {name: {py: [accepted versions]}}}
  exit 0 ok, 2 could not fetch / parse (no verdict possible).
Run with python3 -I.
"""
import json
import os
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

PKGS = ("numpy", "scipy", "pandas")
TTL = 600


def fetch(url, path):
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < TTL:
        with open(path) as f:
            return json.load(f)
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eval-check", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
            data = json.loads(raw)
            tmp = f"{path}.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, path)
            return data
        except Exception as e:  # network, HTTP, JSON
            last = e
            time.sleep(2)
    raise RuntimeError(f"{url}: {last}")


def tup(s):
    return tuple(int(x) for x in s.split("."))


def pad(t, n):
    return t + (0,) * (n - len(t))


def admits(spec, v):
    """Does a PEP 440 requires_python specifier set admit Python version tuple v (e.g. (3, 11))?"""
    if not spec:
        return True
    for term in spec.split(","):
        m = re.fullmatch(r"\s*(~=|==|!=|<=|>=|<|>)\s*([0-9][0-9.]*?)(\.\*)?\s*", term)
        if not m:
            raise ValueError(f"cannot parse specifier {term!r}")
        op, ver, star = m.group(1), tup(m.group(2)), bool(m.group(3))
        n = max(len(ver), len(v))
        pv, ps = pad(v, n), pad(ver, n)
        if op == "~=":
            ok = pv >= ps and v[: len(ver) - 1] == ver[:-1]
        elif op in ("==", "!="):
            ok = (v[: len(ver)] == ver) if star else (pv == ps)
            if op == "!=":
                ok = not ok
        elif op == ">=":
            ok = pv >= ps
        elif op == ">":
            ok = pv > ps
        elif op == "<=":
            ok = pv <= ps
        else:
            ok = pv < ps
        if not ok:
            return False
    return True


def supported_pythons(cycles, today):
    out = []
    for c in cycles:
        cyc = str(c.get("cycle", ""))
        eol = c.get("eol")
        rel = c.get("releaseDate")
        if not re.fullmatch(r"3\.\d+", cyc) or not isinstance(eol, str):
            continue
        if date.fromisoformat(eol) >= today and (not rel or date.fromisoformat(rel) <= today):
            out.append(cyc)
    return sorted(out, key=lambda s: int(s.split(".")[1]))


def accepted_versions(data, v, now):
    """Newest final, non-yanked release admitting python v; also the runner-up if the newest is < 36 h old."""
    found = []
    for ver, files in data["releases"].items():
        if not re.fullmatch(r"\d+(\.\d+)*", ver):
            continue
        files = [f for f in files if not f.get("yanked")]
        if not files or not any(admits(f.get("requires_python"), v) for f in files):
            continue
        first = min(datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00")) for f in files)
        found.append((tup(ver), ver, first))
    found.sort()
    if not found:
        raise ValueError("no compatible release")
    out = [found[-1][1]]
    if len(found) > 1 and now - found[-1][2] < timedelta(hours=36):
        out.append(found[-2][1])
    return out


def main():
    args = sys.argv[1:]
    cache = args[0]
    today = date.fromisoformat(args[args.index("--today") + 1]) if "--today" in args else datetime.now(timezone.utc).date()
    now = datetime.now(timezone.utc)
    try:
        os.makedirs(cache, exist_ok=True)
        cycles = fetch("https://endoflife.date/api/python.json", os.path.join(cache, "python-eol.json"))
        pk = {p: fetch(f"https://pypi.org/pypi/{p}/json", os.path.join(cache, f"{p}.json")) for p in PKGS}
        pys = []
        for d in (today, today - timedelta(days=1)):    # tolerate a UTC/local date difference
            s = supported_pythons(cycles, d)
            if s and s[0] not in pys:
                pys.append(s[0])
        if not pys:
            raise ValueError("no supported python cycle found")
        eol = {str(c["cycle"]): c["eol"] for c in cycles}
        res = {"python": pys, "eol": {p: eol[p] for p in pys}, "pkgs": {}}
        for p, data in pk.items():
            res["pkgs"][p] = {py: accepted_versions(data, tup(py), now) for py in pys}
    except Exception as e:
        print(f"could not verify: {type(e).__name__}: {e}")
        return 2
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
