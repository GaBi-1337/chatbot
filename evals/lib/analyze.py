"""analyze.py PROXY_JSONL -> JSON summary of one run's model traffic (OpenAI chat-completions).
requests: chat completion requests; tool_calls: calls the server returned (parsed from ATEM by llama-server);
text_toolcalls: replies whose *content* still contains ATEM/function-call markup (call printed as text, unparsed);
bad_args: returned tool calls whose arguments are not valid JSON; http_errors: non-200 replies;
empty_turns: replies with neither text nor tool calls; prompt/cache/gen tokens from llama-server `timings`."""
import json, re, sys
MARK = re.compile(r"<atem:|</?atem|function_calls>|<invoke name=|<tool_call>", re.I)

def sse(text):
    for line in text.splitlines():
        if line.startswith("data:"):
            d = line[5:].strip()
            if d and d != "[DONE]":
                try: yield json.loads(d)
                except Exception: pass

def parse(body):
    txt, rsn, calls, finish, usage, timings = "", "", {}, None, None, None
    if body.lstrip().startswith("{"):
        j = json.loads(body); ch = (j.get("choices") or [{}])[0]; m = ch.get("message") or {}
        for i, tc in enumerate(m.get("tool_calls") or []):
            calls[i] = {"name": tc["function"]["name"], "args": tc["function"].get("arguments") or ""}
        return m.get("content") or "", m.get("reasoning_content") or "", list(calls.values()), ch.get("finish_reason"), j.get("usage"), j.get("timings")
    for ev in sse(body):
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            txt += d.get("content") or ""; rsn += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""}); f = tc.get("function") or {}
                c["name"] += f.get("name") or ""; c["args"] += f.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
        usage = ev.get("usage") or usage; timings = ev.get("timings") or timings
    return txt, rsn, list(calls.values()), finish, usage, timings

def summarize(path):
    s = {"requests": 0, "other_requests": 0, "tool_calls": 0, "tools_used": {}, "text_toolcalls": 0, "bad_args": 0,
         "http_errors": 0, "empty_turns": 0, "finish": {}, "prompt_tokens": 0, "cache_tokens": 0, "gen_tokens": 0,
         "first_prompt_tokens": None, "max_context": 0, "prompt_ms": 0.0, "predicted_ms": 0.0,
         "draft_n": 0, "draft_accepted": 0, "model_seconds": 0.0, "tools_declared": None,
         "sent": {}, "examples": []}
    try: recs = [json.loads(l) for l in open(path)]
    except FileNotFoundError: return s
    for r in recs:
        if not r["path"].split("?")[0].endswith("/chat/completions"):
            s["other_requests"] += 1; continue
        s["requests"] += 1; s["model_seconds"] += (r["t1"] or r["t0"]) - r["t0"]
        req = r.get("req") if isinstance(r.get("req"), dict) else {}
        if s["tools_declared"] is None: s["tools_declared"] = sorted(t.get("function", {}).get("name", "?") for t in req.get("tools") or [])
        for k in ("temperature", "top_p", "top_k", "max_tokens", "parallel_tool_calls", "reasoning_effort", "chat_template_kwargs", "tool_choice", "stream"):
            if k in req: s["sent"].setdefault(k, set()).add(json.dumps(req[k], sort_keys=True))
        if r["status"] != 200 or r.get("err"):
            s["http_errors"] += 1; s["examples"].append({"req": r["id"], "http": r["status"], "err": r.get("err"), "resp": (r["resp"] or "")[:300]}); continue
        try: txt, rsn, calls, finish, usage, tm = parse(r["resp"] or "")
        except Exception as e:
            s["http_errors"] += 1; s["examples"].append({"req": r["id"], "parse_error": repr(e)}); continue
        s["finish"][str(finish)] = s["finish"].get(str(finish), 0) + 1
        s["tool_calls"] += len(calls)
        for c in calls:
            s["tools_used"][c["name"]] = s["tools_used"].get(c["name"], 0) + 1
            try: json.loads(c["args"] or "{}")
            except Exception:
                s["bad_args"] += 1; s["examples"].append({"req": r["id"], "bad_args": c["args"][:300]})
        if MARK.search(txt or ""):
            s["text_toolcalls"] += 1; s["examples"].append({"req": r["id"], "text_toolcall": txt[:400]})
        if not calls and not (txt or "").strip(): s["empty_turns"] += 1
        if tm:
            pn, cn, gn = tm.get("prompt_n") or 0, tm.get("cache_n") or 0, tm.get("predicted_n") or 0
            s["prompt_ms"] += tm.get("prompt_ms") or 0; s["predicted_ms"] += tm.get("predicted_ms") or 0
            s["draft_n"] += tm.get("draft_n") or 0; s["draft_accepted"] += tm.get("draft_n_accepted") or 0
        else:
            u = usage or {}; pn, cn, gn = u.get("prompt_tokens") or 0, 0, u.get("completion_tokens") or 0
        s["prompt_tokens"] += pn; s["cache_tokens"] += cn; s["gen_tokens"] += gn
        s["max_context"] = max(s["max_context"], pn + cn + gn)
        if s["first_prompt_tokens"] is None: s["first_prompt_tokens"] = pn + cn
    s["decode_tps"] = round(s["gen_tokens"] / (s["predicted_ms"] / 1000), 1) if s["predicted_ms"] else None
    s["prefill_tps"] = round(s["prompt_tokens"] / (s["prompt_ms"] / 1000), 1) if s["prompt_ms"] else None
    s["draft_accept"] = round(s["draft_accepted"] / s["draft_n"], 3) if s["draft_n"] else None
    s["sent"] = {k: sorted(v) for k, v in s["sent"].items()}
    s["model_seconds"] = round(s["model_seconds"], 1); s["examples"] = s["examples"][:6]
    return s

if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1]), indent=1))
