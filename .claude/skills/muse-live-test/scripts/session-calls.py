"""Print the tool calls and user/assistant text of one Hermes session. usage: session-calls.py HERMES_HOME SESSION_ID"""
import json, sqlite3, sys
db = sqlite3.connect(f"file:{sys.argv[1]}/state.db?mode=ro", uri=True)
for role, content, calls in db.execute("select role, content, tool_calls from messages where session_id=? order by id", (sys.argv[2],)):
    if role == "assistant" and calls:
        for c in json.loads(calls):
            f = c.get("function", c)
            print("  call", f.get("name"), str(f.get("arguments"))[:120])
    elif role in ("user", "assistant"):
        print(f"  {role}:", (content or "")[:160].replace("\n", " "))
