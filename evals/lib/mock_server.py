"""Mock OpenAI chat-completions endpoint for wiring tests without the GPU.
usage: mock_server.py PORT LOGFILE   (logs each request as JSONL; answers every chat request with plain text)"""
import json, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, LOG = int(sys.argv[1]), sys.argv[2]
TEXT = "MOCK OK: wiring works."

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _send(self, code, obj=None, ctype="application/json", body=None):
        data = body if body is not None else json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def _log(self, body):
        with open(LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "method": self.command, "path": self.path, "body": body}) + "\n")
    def do_GET(self):
        self._log(None)
        p = self.path.split("?")[0].rstrip("/")
        if p.endswith("/models"):
            return self._send(200, {"object": "list", "data": [{"id": "muse-glimmer-30B", "object": "model", "owned_by": "mock",
                                    "meta": {"n_ctx_train": 131072}}]})
        if p.endswith("/props"):
            return self._send(200, {"default_generation_settings": {"n_ctx": 131072}, "model_alias": "muse-glimmer-30B"})
        if p.endswith("/health"):
            return self._send(200, {"status": "ok"})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); raw = self.rfile.read(n)
        try: body = json.loads(raw)
        except Exception: body = {"_raw": raw[:2000].decode(errors="replace")}
        self._log(body)
        if not self.path.split("?")[0].endswith("/chat/completions"):
            return self._send(404, {"error": "not found"})
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]; now = int(time.time()); model = body.get("model")
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if not body.get("stream"):
            return self._send(200, {"id": cid, "object": "chat.completion", "created": now, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": TEXT}, "finish_reason": "stop"}], "usage": usage})
        chunks = [{"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
                   "choices": [{"index": 0, "delta": {"role": "assistant", "content": TEXT}, "finish_reason": None}]},
                  {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}]
        data = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
        self._send(200, body=data, ctype="text/event-stream")

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
