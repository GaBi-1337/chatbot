"""collect.py row RUN_DIR TASK RUN_NO EXIT CHECK_RC WALL_S  -> prints one results.jsonl row (JSON)
   collect.py summary RESULTS_JSONL                         -> prints a markdown summary (table + per-task totals)

Per-run inputs in RUN_DIR: proxy.jsonl (model traffic via log_proxy.py), stream.jsonl (hermes --format stream-json),
check.log (check.sh output)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import summarize  # noqa: E402


def stream_stats(path):
    s = {"tool_uses": 0, "tool_errors": 0, "unknown_tools": 0, "hermes_exit": None, "hermes_error": None,
         "answer_chars": 0, "tools": {}}
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except FileNotFoundError:
        return s
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        if t == "tool_use":
            s["tool_uses"] += 1
            s["tools"][ev.get("name", "?")] = s["tools"].get(ev.get("name", "?"), 0) + 1
        elif t == "tool_result":
            if ev.get("is_error"):
                s["tool_errors"] += 1
            if "unknown tool" in (ev.get("output") or "").lower()[:200]:
                s["unknown_tools"] += 1
        elif t == "result":
            s["hermes_exit"] = ev.get("exit_code"); s["hermes_error"] = ev.get("error")
            s["answer_chars"] = len(ev.get("text") or "")
    return s


def row(rd, task, run_no, exit_code, check_rc, wall):
    p = summarize(os.path.join(rd, "proxy.jsonl"))
    st = stream_stats(os.path.join(rd, "stream.jsonl"))
    try:
        check_msg = open(os.path.join(rd, "check.log"), errors="replace").read().strip().splitlines()[-1:] or [""]
    except FileNotFoundError:
        check_msg = [""]
    check_rc = int(check_rc); exit_code = int(exit_code)
    result = {0: "pass", 1: "fail", 2: "unscored"}.get(check_rc, f"check_rc={check_rc}")
    return {
        "task": task, "run": int(run_no), "result": result, "check_rc": check_rc, "check_msg": check_msg[0][:200],
        "exit": exit_code, "timeout": exit_code in (124, 137), "wall_s": round(float(wall), 1),
        "requests": p["requests"], "tool_calls": p["tool_calls"],
        "toolcall_failures": {"text_toolcalls": p["text_toolcalls"], "bad_args": p["bad_args"],
                              "http_errors": p["http_errors"], "empty_turns": p["empty_turns"],
                              "unknown_tools": st["unknown_tools"]},
        "tool_errors": st["tool_errors"],
        "prompt_tokens": p["prompt_tokens"], "cache_tokens": p["cache_tokens"], "gen_tokens": p["gen_tokens"],
        "first_prompt_tokens": p["first_prompt_tokens"], "max_context": p["max_context"],
        "model_seconds": p["model_seconds"], "decode_tps": p["decode_tps"], "prefill_tps": p["prefill_tps"],
        "draft_n": p["draft_n"], "draft_accepted": p["draft_accepted"], "draft_accept": p["draft_accept"], "finish": p["finish"], "tools_used": p["tools_used"],
        "sent": p["sent"], "hermes_exit": st["hermes_exit"], "hermes_error": st["hermes_error"],
        "examples": p["examples"], "dir": rd,
    }


def summary(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out = ["| task | run | result | exit | wall s | requests | tool calls | call failures | tool errors |"
           " prompt tok | cache_n tok | gen tok | cache hit % | decode t/s | draft accept |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], r["run"])):
        f = r["toolcall_failures"]; nf = f["text_toolcalls"] + f["bad_args"] + f["http_errors"] + f["unknown_tools"]
        tot = r["prompt_tokens"] + r["cache_tokens"]
        hit = f"{100 * r['cache_tokens'] / tot:.0f}" if tot else "-"
        ex = "timeout" if r["timeout"] else r["exit"]
        out.append(f"| {r['task']} | {r['run']} | {r['result']} | {ex} | {r['wall_s']} | {r['requests']} | {r['tool_calls']} "
                   f"| {nf} | {r['tool_errors']} | {r['prompt_tokens']} | {r['cache_tokens']} | {r['gen_tokens']} | {hit} "
                   f"| {r.get('decode_tps') or '-'} | {r.get('draft_accept') if r.get('draft_accept') is not None else '-'} |")
    out += ["", "| task | pass | fail | unscored | median wall s |", "|---|---|---|---|---|"]
    for t in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == t]; walls = sorted(r["wall_s"] for r in rs)
        c = lambda k: sum(r["result"] == k for r in rs)
        out.append(f"| {t} | {c('pass')}/{len(rs)} | {c('fail')} | {c('unscored')} | {walls[len(walls) // 2]} |")
    out += ["", "call failures = calls printed as text (ATEM left in content) + unparsable arguments + HTTP errors + unknown tool names;",
            "tool errors = tool results Hermes flagged is_error (the call parsed, the tool failed). prompt tok = tokens",
            "evaluated (llama-server timings.prompt_n); cache_n tok = prefix tokens reused from the cache."]
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "row":
        print(json.dumps(row(*sys.argv[2:8])))
    elif sys.argv[1] == "line":          # one-line progress view of a row read from stdin
        r = json.loads(sys.stdin.read()); f = r["toolcall_failures"]
        print(f"   {r['result']:8s} exit={r['exit']} wall={r['wall_s']}s requests={r['requests']} tool_calls={r['tool_calls']} "
              f"call_failures={sum(f.values()) - f['empty_turns']} tool_errors={r['tool_errors']} prompt={r['prompt_tokens']} "
              f"cache_n={r['cache_tokens']} gen={r['gen_tokens']}  {r['check_msg']}")
    elif sys.argv[1] == "summary":
        print(summary(sys.argv[2]))
    else:
        sys.exit("usage: collect.py row RUN_DIR TASK RUN_NO EXIT CHECK_RC WALL_S | line < ROW | summary RESULTS_JSONL")
