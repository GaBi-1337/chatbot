"""Summarize Hermes stream-json runs: tool calls (with errors) and the final text. usage: cap-sum.py STREAM.jsonl..."""
import json, sys
for f in sys.argv[1:]:
    calls, final = [], ""
    for line in open(f):
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") == "tool_use":
            inp = e.get("input") or {}
            arg = inp.get("command") or inp.get("url") or inp.get("path") or inp.get("name") or inp.get("query") or inp.get("goal") or ""
            calls.append(f"{e['name']}({str(arg)[:70]})")
        if e.get("type") == "tool_result" and e.get("is_error"):
            calls.append("  !error: " + str(e.get("output"))[:120].replace("\n", " "))
        if e.get("type") in ("result", "text") and (e.get("text") or e.get("result")):
            final = str(e.get("text") or e.get("result"))[:700]
    print("=====", f.split("/")[-1], len([c for c in calls if not c.startswith("  !")]), "calls")
    for c in calls: print("  ", c)
    print("  FINAL:", final.replace("\n", " | "))
