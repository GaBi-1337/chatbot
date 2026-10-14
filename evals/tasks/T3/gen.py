#!/usr/bin/env python3
"""Deterministic generator for the T3 tree.
  gen.py OUTDIR            write the tree to OUTDIR/data
  gen.py --truth           print ground-truth JSON (no files written)
"""
import json
import os
import random
import sys

SEED = 20240611
SERVICES = ["auth", "billing", "gateway", "search", "mailer", "scheduler", "storage"]
LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN", "ERROR"]
WORDS = "alpha beta gamma delta request timeout retry cache miss hit user session token queue flush commit rollback".split()


def build():
    """Return dict relpath -> bytes."""
    rnd = random.Random(SEED)
    files = {}
    dirs = [f"{a}/{b}" for a in ("app", "var", "srv", "opt", "home") for b in ("alpha", "beta", "gamma", "delta", "eps")]
    for i in range(420):
        d = rnd.choice(dirs)
        kind = rnd.choice(["log", "log", "log", "txt", "json", "csv", "md", "log.bak"])
        name = f"f{i:03d}.{kind}"
        path = f"data/{d}/{name}"
        if kind in ("log", "log.bak"):
            lines = []
            for _ in range(rnd.randint(0, 40)):
                svc = rnd.choice(SERVICES)
                lvl = rnd.choice(LEVELS)
                msg = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(2, 6)))
                lines.append(f"2024-06-{rnd.randint(1,28):02d}T{rnd.randint(0,23):02d}:00:00 {svc} {lvl} {msg}")
            body = "\n".join(lines) + ("\n" if lines else "")
        elif kind == "txt":
            body = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(0, 200)))
        elif kind == "json":
            body = json.dumps({"id": i, "tags": [rnd.choice(WORDS) for _ in range(3)], "note": "x" * rnd.randint(0, 50)})
        elif kind == "csv":
            body = "id,val\n" + "\n".join(f"{j},{rnd.randint(0,999)}" for j in range(rnd.randint(0, 30))) + "\n"
        else:
            body = "# notes\n" + " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(0, 80))) + "\n"
        files[path] = body.encode()
    # a few empty files
    for i in range(7):
        files[f"data/var/empty/z{i}.dat"] = b""
    # buried markers: some files get a TODO(release) line, decoys get TODO(later)
    keys = sorted(files)
    for k in rnd.sample(keys, 14):
        if files[k]:
            files[k] = files[k] + b"\nTODO(release): bump version before shipping\n"
        else:
            files[k] = b"TODO(release): empty before\n"
    for k in rnd.sample(keys, 20):
        if b"TODO(release)" not in files[k]:
            files[k] += b"\nTODO(later): tidy up\n"
    return files


def truth():
    files = build()
    counts = {}
    total = 0
    for p, b in files.items():
        if p.endswith(".log"):  # only real .log files, not .log.bak
            for line in b.decode().splitlines():
                parts = line.split(" ", 3)
                if len(parts) >= 3 and parts[2] == "ERROR":
                    counts[parts[1]] = counts.get(parts[1], 0) + 1
                    total += 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    assert len(top) < 2 or top[0][1] > top[1][1], "top service must be unique"
    return {
        "total_errors": total,
        "top_error_service": top[0][0],
        "release_todo_files": sorted(p for p, b in files.items() if b"TODO(release)" in b),
        "empty_file_count": sum(1 for b in files.values() if not b),
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--truth"]:
        print(json.dumps(truth(), indent=1))
    else:
        out = sys.argv[1]
        for p, b in build().items():
            fp = os.path.join(out, p)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as fh:
                fh.write(b)
