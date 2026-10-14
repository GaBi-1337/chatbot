"""Drive one INTERACTIVE muse session through a pty (one-shot runs get a weaker skills prompt): send the prompt,
wait until state.db shows a final assistant message, then /exit.
usage: tui-drive.py WORKDIR MUSE_STATE PROMPT_FILE LOG [TIMEOUT_S]"""
import os, pty, sqlite3, sys, threading, time
from pathlib import Path
MUSE = str(Path(__file__).resolve().parents[4] / "muse")
work, state, prompt_file, log = sys.argv[1:5]
timeout = float(sys.argv[5]) if len(sys.argv) > 5 else 900
prompt = open(prompt_file).read().strip().replace("\n", " ")
t_start = time.time()
pid, fd = pty.fork()
if pid == 0:
    os.chdir(work)
    env = dict(os.environ, MUSE_STATE=state, TERM="xterm-256color", COLUMNS="160", LINES="50")
    os.execvpe(MUSE, ["muse"], env)
out = open(log, "wb")
def drain():
    while True:
        try: data = os.read(fd, 65536)
        except OSError: return
        if not data: return
        out.write(data); out.flush()
threading.Thread(target=drain, daemon=True).start()
def last_final():
    try:
        db = sqlite3.connect(f"file:{state}/state.db?mode=ro", uri=True, timeout=2)
        sid = db.execute("select id from sessions where started_at >= ? order by started_at desc limit 1", (t_start - 5,)).fetchone()
        if not sid: return None, None
        row = db.execute("select role, content, tool_calls from messages where session_id=? order by id desc limit 1", (sid[0],)).fetchone()
        db.close()
        if row and row[0] == "assistant" and not row[2] and (row[1] or "").strip():
            return sid[0], row[1]
        return sid[0], None
    except Exception:
        return None, None
time.sleep(30)
os.write(fd, prompt.encode())
time.sleep(2)                       # a separate Enter: in the same write it counts as part of a paste
os.write(fd, b"\r")
sid, final = None, None
while time.time() - t_start < timeout:
    time.sleep(5)
    sid, final = last_final()
    if final: break
time.sleep(3)
os.write(fd, b"/exit\r")
for _ in range(30):
    if os.waitpid(pid, os.WNOHANG)[0]: break
    time.sleep(1)
else:
    os.kill(pid, 9)
print(f"session={sid} wall={time.time()-t_start:.0f}s final={'yes' if final else 'NO'}")
print((final or "")[:800])
