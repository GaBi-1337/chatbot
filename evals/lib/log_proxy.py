"""Transparent logging reverse proxy for one eval run.
usage: log_proxy.py LISTEN_PORT UPSTREAM_PORT LOGFILE
Forwards every request to 127.0.0.1:UPSTREAM_PORT, streams the response back unchanged (HTTP/1.0,
close-delimited) and appends one JSONL record per request: request body + full raw response text
(SSE included, so llama-server's final `timings` chunk with prompt_n/cache_n/predicted_n is kept).
Authorization headers are never logged."""
import http.client, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LPORT, UPORT, LOG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lock = threading.Lock(); counter = [0]
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade", "te", "trailer", "content-length", "host"}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    def log_message(self, *a): pass
    def _proxy(self):
        with lock:
            counter[0] += 1; rid = counter[0]
        t0 = time.time()
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        rec = {"id": rid, "t0": t0, "method": self.command, "path": self.path, "ua": self.headers.get("User-Agent"),
               "req": None, "status": None, "resp": None, "t1": None, "err": None}
        try:
            rec["req"] = json.loads(body) if body else None
        except Exception:
            rec["req"] = {"_raw": body[:4000].decode(errors="replace")}
        chunks = []
        try:
            conn = http.client.HTTPConnection("127.0.0.1", UPORT, timeout=3600)
            conn.request(self.command, self.path, body=body, headers=hdrs)
            r = conn.getresponse(); rec["status"] = r.status
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() not in HOP: self.send_header(k, v)
            self.send_header("Connection", "close"); self.end_headers()
            while True:
                c = r.read1(65536)
                if not c: break
                chunks.append(c)
                try:
                    self.wfile.write(c); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    rec["err"] = "client disconnected"; break
            conn.close()
        except Exception as e:
            rec["err"] = repr(e)
            try: self.send_error(502, "proxy error")
            except Exception: pass
        rec["t1"] = time.time(); rec["resp"] = b"".join(chunks).decode(errors="replace")
        with lock, open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = _proxy

ThreadingHTTPServer(("127.0.0.1", LPORT), H).serve_forever()
