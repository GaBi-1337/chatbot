"""Drive one INTERACTIVE muse session through a pty (one-shot runs get a weaker skills prompt): send the prompt,
wait until state.db shows a final assistant message, then /exit.
usage: tui-drive.py WORKDIR MUSE_STATE PROMPT_FILE LOG [TIMEOUT_S] [SETTLE_S]
SETTLE_S > 0: after the answer, wait (up to SETTLE_S) until the model server has been idle for 20 s, so work
Hermes starts after a turn (the background review) can finish before /exit."""
import os, pty, sqlite3, sys, threading, time
from pathlib import Path
MUSE = str(Path(__file__).resolve().parents[4] / "muse")
work, state, prompt_file, log = sys.argv[1:5]
timeout = float(sys.argv[5]) if len(sys.argv) > 5 else 900
settle = float(sys.argv[6]) if len(sys.argv) > 6 else 0
prompt = open(prompt_file).read().strip().replace("\n", " ")
t_start = time.time()
pid, fd = pty.fork()
if pid == 0:
    os.chdir(work)
    env = dict(os.environ, MUSE_STATE=state, TERM="xterm-256color", COLUMNS="160", LINES="50")
    env.setdefault("MUSE_HOME_DIR", str(Path(MUSE).parent / "run/cap/homedir"))  # never the user's state/home
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
def busy():
    import json, urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{os.environ.get('MUSE_PORT', '8080')}/slots", timeout=5) as r:
            return any(s.get("is_processing") for s in json.load(r))
    except Exception:
        return False
def user_sent():
    try:
        db = sqlite3.connect(f"file:{state}/state.db?mode=ro", uri=True, timeout=2)
        sid = db.execute("select id from sessions where started_at >= ? order by started_at desc limit 1", (t_start - 5,)).fetchone()
        n = db.execute("select count(*) from messages where session_id=? and role='user'", (sid[0],)).fetchone()[0] if sid else 0
        db.close()
        return n > 0
    except Exception:
        return False
def ready():
    try:
        return b"Type your message" in open(log, "rb").read()
    except OSError:
        return False
# Wait for the TUI's welcome line, not a fixed delay: on a loaded machine it starts slowly, and text typed while it
# recovers its terminal mode is dropped. Then check the prompt landed (a user message, or the model working).
t0 = time.monotonic()
while time.monotonic() - t0 < 300 and not ready():
    time.sleep(1)
time.sleep(5)
for attempt in range(3):
    os.write(fd, prompt.encode())
    time.sleep(2)                   # a separate Enter: in the same write it counts as part of a paste
    os.write(fd, b"\r")
    t1 = time.monotonic()
    while time.monotonic() - t1 < 90 and not (user_sent() or busy()):
        time.sleep(3)
    if user_sent() or busy():
        break
    os.write(fd, b"\x15")          # Ctrl+U: clear whatever is left in the input line, then type again
sid, final = None, None
t0 = time.monotonic()
while time.monotonic() - t0 < timeout:
    time.sleep(5)
    sid, final = last_final()
    if final: break
time.sleep(3)
quiet, t_settle = 0.0, time.monotonic()
while settle > 0 and time.monotonic() - t_settle < settle and quiet < 20:
    time.sleep(2)
    quiet = 0.0 if busy() else quiet + 2
os.write(fd, b"/exit\r")
for _ in range(30):
    if os.waitpid(pid, os.WNOHANG)[0]: break
    time.sleep(1)
else:
    os.kill(pid, 9)
print(f"session={sid} wall={time.time()-t_start:.0f}s final={'yes' if final else 'NO'}")
print((final or "")[:800])
