import json
import os
import sys
import time
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent
SERVER = HERE.parent / "server.py"
sys.path.insert(0, str(HERE.parent))
import server  # noqa: E402

HTML = (HERE / "fixtures" / "page.html").read_bytes()


def mock_client(handler):
    def make():
        return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True, max_redirects=8)
    return make


@pytest.fixture
def fetch(monkeypatch):
    def run(handler, **kw):
        monkeypatch.setattr(server, "_make_client", mock_client(handler))
        return server.fetch_impl(**kw)
    return run


def resp(body, ctype, status=200):
    return lambda req: httpx.Response(status, content=body, headers={"content-type": ctype})


def test_html_to_markdown(fetch):
    out = fetch(resp(HTML, "text/html; charset=utf-8"), url="https://x.test/a")
    assert "Llama Demo" in out and "first paragraph" in out
    assert "[link](https://example.com/x)" in out
    assert "copyright" not in out
    assert "Truncated" not in out


def test_truncation_and_paging(fetch):
    text = "word " * 5000
    h = resp(text.encode(), "text/plain")
    a = fetch(h, url="https://x.test/t", max_chars=1000)
    assert "start_index=1000" in a and "Truncated" in a
    b = fetch(h, url="https://x.test/t", max_chars=1000, start_index=1000)
    assert "start_index=2000" in b
    last = fetch(h, url="https://x.test/t", max_chars=40000, start_index=24000)
    assert "Truncated" not in last
    assert "past the end" in fetch(h, url="https://x.test/t", start_index=99999)


def test_json(fetch):
    out = fetch(resp(b'{"a":1,"b":[1,2]}', "application/json"), url="https://x.test/j")
    assert '"a": 1' in out and "Type: json" in out


def test_binary_refused(fetch):
    out = fetch(resp(b"PK\x03\x04" + b"\x00" * 64, "application/zip"), url="https://x.test/p")
    assert out.startswith("Error") and "binary" in out
    out = fetch(resp(b"\x00\x01" * 64, "application/octet-stream"), url="https://x.test/p")
    assert out.startswith("Error") and "binary" in out


def test_http_error_and_bad_scheme(fetch):
    assert "HTTP 404" in fetch(resp(b"no", "text/html", 404), url="https://x.test/")
    assert server.fetch_impl("file:///etc/passwd").startswith("Error")


def test_redirect(fetch):
    def h(req):
        if req.url.path == "/old":
            return httpx.Response(302, headers={"location": "https://x.test/new"})
        return httpx.Response(200, content=b"hello", headers={"content-type": "text/plain"})
    out = fetch(h, url="https://x.test/old")
    assert "redirected from" in out and "hello" in out and "x.test/new" in out


def test_redirect_loop(fetch):
    out = fetch(lambda r: httpx.Response(302, headers={"location": str(r.url)}), url="https://x.test/l")
    assert "redirects" in out


def test_timeout(fetch):
    def h(req):
        raise httpx.ReadTimeout("slow")
    assert "timed out" in fetch(h, url="https://x.test/")


def test_size_cap(fetch, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 1000)
    out = fetch(resp(b"a" * 2500, "text/plain"), url="https://x.test/big", max_chars=40000)
    assert "download truncated" in out and "of 1000" in out


def test_search_retry_then_success(monkeypatch):
    calls = []
    sleeps = []
    def fake(q, n):
        calls.append(q)
        if len(calls) < 3:
            return []
        return [{"title": "T", "href": "https://a.test", "body": "s " * 400}]
    monkeypatch.setattr(server, "_ddgs_text", fake)
    monkeypatch.setattr(server, "_sleep", sleeps.append)
    out = server.search_impl("q")
    assert len(calls) == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0]
    assert "1. T" in out and "https://a.test" in out and out.endswith("...")


def test_search_exception_gives_message(monkeypatch):
    def boom(q, n):
        raise RuntimeError("ratelimit")
    monkeypatch.setattr(server, "_ddgs_text", boom)
    monkeypatch.setattr(server, "_sleep", lambda s: None)
    out = server.search_impl("q")
    assert "No results" in out and "ratelimit" in out


# --- PDF ---
def make_pdf(pages):
    """Minimal valid PDF with one line of Helvetica text per page ("" = blank page)."""
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {n} >>"]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents {4 + 2 * i} 0 R "
            "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
        )
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET" if text else ""
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_pdf_text_with_page_markers(fetch):
    pdf = make_pdf(["Hello first page", "Second page here", ""])
    out = fetch(resp(pdf, "application/pdf"), url="https://x.test/a.pdf")
    assert "Type: pdf" in out
    assert out.index("--- page 1 ---") < out.index("Hello first page") < out.index("--- page 2 ---")
    assert "Second page here" in out and "--- page 3 ---" in out
    assert out.count("--- page ") == 3


def test_pdf_paging(fetch):
    pdf = make_pdf([f"Page text number {i}" for i in range(1, 80)])
    h = resp(pdf, "application/pdf")
    a = fetch(h, url="https://x.test/a.pdf", max_chars=500)
    assert "Truncated" in a and "start_index=500" in a and "Page text number 1" in a
    b = fetch(h, url="https://x.test/a.pdf", max_chars=500, start_index=500)
    assert "chars 500-1000" in b and "--- page" in b


def test_pdf_detected_by_magic_when_octet_stream(fetch):
    out = fetch(resp(make_pdf(["Magic works"]), "application/octet-stream"), url="https://x.test/download")
    assert "Type: pdf" in out and "Magic works" in out
    out = fetch(resp(make_pdf(["Magic works"]), ""), url="https://x.test/download")
    assert "Magic works" in out


def test_pdf_bigger_than_html_cap_but_within_pdf_cap(fetch, monkeypatch):
    monkeypatch.setattr(server, "MAX_BYTES", 500)
    monkeypatch.setattr(server, "MAX_PDF_BYTES", 100_000)
    pdf = make_pdf(["Still readable"])
    assert len(pdf) > 500
    assert "Still readable" in fetch(resp(pdf, "application/pdf"), url="https://x.test/a.pdf")
    monkeypatch.setattr(server, "MAX_PDF_BYTES", 300)
    out = fetch(resp(pdf, "application/pdf"), url="https://x.test/a.pdf")
    assert out.startswith("Error") and "PDF larger" in out


def test_pdf_too_large_without_content_length(fetch, monkeypatch):
    monkeypatch.setattr(server, "MAX_PDF_BYTES", 300)
    pdf = make_pdf(["x"] * 3)

    def h(req):
        stream = httpx.ByteStream(pdf)  # no content-length header
        return httpx.Response(200, stream=stream, headers={"content-type": "application/pdf"})

    out = fetch(h, url="https://x.test/a.pdf")
    assert out.startswith("Error") and "PDF larger" in out


def test_pdf_broken_and_scanned(fetch):
    out = fetch(resp(b"%PDF-1.4\nthis is not a pdf at all, just junk bytes", "application/pdf"), url="https://x.test/a.pdf")
    assert out.startswith("Error") and "PDF" in out
    out = fetch(resp(make_pdf(["", ""]), "application/pdf"), url="https://x.test/a.pdf")
    assert "No text layer" in out and "--- page 2 ---" in out


# --- images ---
PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\xa5\xf6E@\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.mark.parametrize(
    "body,ctype,ext",
    [
        (PNG, "image/png", ".png"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 40, "image/jpeg", ".jpg"),
        (b"GIF89a" + b"\x00" * 40, "image/gif", ".gif"),
        (b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 40, "image/webp", ".webp"),
        (PNG, "application/octet-stream", ".png"),
    ],
)
def test_image_saved_to_tmp(fetch, monkeypatch, tmp_path, body, ctype, ext):
    monkeypatch.setattr(server.tempfile, "tempfile_tempdir", str(tmp_path), raising=False)
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    out = fetch(resp(body, ctype), url="https://x.test/i")
    path = [ln.split("Saved to: ")[1] for ln in out.splitlines() if ln.startswith("Saved to: ")][0]
    assert os.path.isabs(path) and path.startswith(str(tmp_path)) and path.endswith(ext)
    assert Path(path).read_bytes() == body
    assert "vision_analyze" in out


def test_image_unsupported_or_fake_refused(fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    assert fetch(resp(b"<svg></svg>" * 5, "image/svg+xml"), url="https://x.test/i").startswith("Error")
    out = fetch(resp(b"<html>not an image</html>", "image/png"), url="https://x.test/i")
    assert out.startswith("Error") and "image" in out
    assert not list(tmp_path.iterdir())


def test_image_size_cap(fetch, monkeypatch, tmp_path):
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(server, "MAX_IMAGE_BYTES", 30)
    out = fetch(resp(PNG, "image/png"), url="https://x.test/i")
    assert out.startswith("Error") and "image larger" in out
    assert not list(tmp_path.iterdir())


# --- browser rendering (browser subprocess mocked) ---
JS_SHELL = b"<html><head><title>app</title></head><body><div id=root></div><script>x()</script></body></html>"
RENDERED = HTML.replace(b"Llama Demo", b"Rendered Demo")


@pytest.fixture
def browser(monkeypatch):
    calls = []

    def fake_run(args, timeout):
        calls.append(args)
        for a in args:
            if a.startswith("--screenshot="):
                Path(a.split("=", 1)[1]).write_bytes(PNG)
                return 0, b"", b""
        return 0, RENDERED, b"noise on stderr"

    monkeypatch.setattr(server, "_find_browser", lambda: "/usr/bin/fake-chromium")
    monkeypatch.setattr(server, "_run_browser", fake_run)
    return calls


def test_auto_render_when_no_static_text(fetch, browser):
    out = fetch(resp(JS_SHELL, "text/html"), url="https://x.test/spa")
    assert "Type: html (rendered)" in out and "Rendered Demo" in out
    args = browser[0]
    assert args[0] == "/usr/bin/fake-chromium" and args[-1] == "https://x.test/spa"
    for flag in ("--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--dump-dom"):
        assert flag in args
    assert any(a.startswith("--virtual-time-budget=") for a in args)


def test_no_render_when_static_text_exists(fetch, browser):
    out = fetch(resp(HTML, "text/html"), url="https://x.test/a")
    assert "Llama Demo" in out and "rendered" not in out and not browser


def test_render_true_forces_browser(fetch, browser):
    out = fetch(resp(HTML, "text/html"), url="https://x.test/a", render=True)
    assert "Rendered Demo" in out and "Type: html (rendered)" in out and len(browser) == 1


def test_render_true_ignored_for_non_html(fetch, browser):
    out = fetch(resp(b"plain text", "text/plain"), url="https://x.test/t", render=True)
    assert "plain text" in out and not browser


def test_render_true_after_http_error(fetch, browser):
    out = fetch(resp(b"forbidden", "text/html", 403), url="https://x.test/a", render=True)
    assert "Rendered Demo" in out and len(browser) == 1
    assert "HTTP 403" in fetch(resp(b"forbidden", "text/html", 403), url="https://x.test/a")


def test_no_browser_gives_clear_message(fetch, monkeypatch):
    monkeypatch.setattr(server, "_find_browser", lambda: None)
    out = fetch(resp(JS_SHELL, "text/html"), url="https://x.test/spa")
    assert "No main text" in out and "no headless Chromium" in out and not out.startswith("Traceback")
    out = fetch(resp(HTML, "text/html"), url="https://x.test/a", render=True)
    assert "Llama Demo" in out and "Rendering failed" in out
    shot = server.screenshot_impl("https://x.test/")
    assert shot.startswith("Error") and "no headless Chromium" in shot


def test_browser_timeout_and_failure(fetch, monkeypatch):
    monkeypatch.setattr(server, "_find_browser", lambda: "/x/chromium")

    def hang(args, timeout):
        raise server.BrowserError("browser timed out after 40s")

    monkeypatch.setattr(server, "_run_browser", hang)
    out = fetch(resp(JS_SHELL, "text/html"), url="https://x.test/spa")
    assert "timed out" in out
    monkeypatch.setattr(server, "_run_browser", lambda a, t: (1, b"", b"[1:2:ERROR:x.cc:3] Page load failed: net::ERR_X"))
    out = server.screenshot_impl("https://x.test/")
    assert out.startswith("Error") and "ERR_X" in out and "[1:2" not in out
    monkeypatch.setattr(server, "_run_browser", lambda a, t: (0, b"  ", b""))
    assert "empty page" in fetch(resp(JS_SHELL, "text/html"), url="https://x.test/spa", render=True)


def test_browser_missing_binary_oserror(monkeypatch):
    monkeypatch.setattr(server, "_find_browser", lambda: "/nonexistent/chromium")
    assert server.screenshot_impl("https://x.test/").startswith("Error: screenshot failed")


def test_run_browser_real_subprocess_hard_timeout(tmp_path):
    script = tmp_path / "slow.sh"
    script.write_text("#!/bin/sh\nsleep 30 &\nsleep 30\n")
    script.chmod(0o755)
    t0 = time.time()
    with pytest.raises(server.BrowserError, match="timed out"):
        server._run_browser([str(script)], 0.5)
    assert time.time() - t0 < 10
    ok = tmp_path / "ok.sh"
    ok.write_text("#!/bin/sh\necho dom; echo err >&2\n")
    ok.chmod(0o755)
    assert server._run_browser([str(ok)], 5) == (0, b"dom\n", b"err\n")


def test_find_browser(monkeypatch, tmp_path):
    exe = tmp_path / "mychrome"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setenv("WEB_MCP_CHROMIUM", str(exe))
    assert server._find_browser() == str(exe)
    monkeypatch.setenv("WEB_MCP_CHROMIUM", str(tmp_path / "missing"))
    assert server._find_browser() is None
    monkeypatch.delenv("WEB_MCP_CHROMIUM")
    found = {"chromium": "/usr/bin/chromium"}
    monkeypatch.setattr(server.shutil, "which", lambda n: found.get(n))
    assert server._find_browser() == "/usr/bin/chromium"
    found.clear()
    assert server._find_browser() is None


# --- screenshots ---
def test_screenshot_default_path_and_size(browser, monkeypatch, tmp_path):
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    out = server.screenshot_impl("http://localhost:3000/")
    path = out.split("saved to: ")[1].splitlines()[0]
    assert path.startswith(str(tmp_path)) and path.endswith(".png") and Path(path).read_bytes() == PNG
    assert "Size: 1x1 px" in out and "vision_analyze" in out
    args = browser[0]
    assert f"--screenshot={path}" in args and "--window-size=1280,900" in args and "--hide-scrollbars" in args
    assert "--no-sandbox" in args and args[-1] == "http://localhost:3000/"


def test_screenshot_custom_path_creates_parents_and_clamps(browser, tmp_path):
    target = tmp_path / "a" / "b" / "shot.png"
    out = server.screenshot_impl("file:///tmp/app.html", path=str(target), width=99999, height=1)
    assert str(target) in out and target.read_bytes() == PNG
    assert "--window-size=3840,200" in browser[0]
    assert server.screenshot_impl("https://x.test", path=str(tmp_path / "d.png"), width="bad", height=None)
    assert "--window-size=1280,900" in browser[1]


def test_screenshot_url_forms(browser, tmp_path):
    f = tmp_path / "app.html"
    f.write_text("<html></html>")
    server.screenshot_impl("localhost:8000/x")
    server.screenshot_impl(str(f))
    assert browser[0][-1] == "http://localhost:8000/x"
    assert browser[1][-1] == f.resolve().as_uri()
    for bad in ("ftp://x", "javascript:alert(1)", "", "/no/such/file.html", "example.com"):
        assert server.screenshot_impl(bad).startswith("Error: url must be")
    assert len(browser) == 2


def test_screenshot_no_image_produced(monkeypatch):
    monkeypatch.setattr(server, "_find_browser", lambda: "/x/chromium")
    monkeypatch.setattr(server, "_run_browser", lambda a, t: (0, b"", b""))
    out = server.screenshot_impl("https://x.test/")
    assert out.startswith("Error") and "no image" in out


# --- stdio client end-to-end (no network: tools fail gracefully / bad url) ---
@pytest.mark.asyncio
async def test_stdio_client_lists_and_calls():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)],
                                   env={**os.environ, "WEB_MCP_SEARCH_RETRIES": "1"})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name: t for t in (await s.list_tools()).tools}
            assert set(tools) == {"web_search", "web_fetch", "web_screenshot"}
            assert "untrusted" in tools["web_fetch"].description
            assert "untrusted" in tools["web_screenshot"].description
            assert "render" in tools["web_fetch"].inputSchema["properties"]
            res = await s.call_tool("web_fetch", {"url": "ftp://nope"})
            assert not res.isError and "Error" in res.content[0].text
            res = await s.call_tool("web_search", {"query": "   "})
            assert "empty query" in res.content[0].text


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("WEB_MCP_LIVE"), reason="set WEB_MCP_LIVE=1 for live smoke")
def test_live_smoke():
    out = server.search_impl("llama.cpp github")
    assert not out.startswith("No results"), out
    assert "github.com" in out
    page = server.fetch_impl("https://github.com/ggml-org/llama.cpp")
    assert not page.startswith("Error"), page[:200]
    assert "llama.cpp" in page and len(page) < 20000 + 2000


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("WEB_MCP_LIVE"), reason="set WEB_MCP_LIVE=1 for live smoke")
def test_live_pdf_image_render_screenshot(tmp_path):
    pdf = server.fetch_impl("https://arxiv.org/pdf/1706.03762", max_chars=2000)
    assert "--- page 1 ---" in pdf and "Attention" in pdf
    img = server.fetch_impl("https://www.python.org/static/img/python-logo.png")
    assert "Saved to: " in img
    if not server._find_browser():
        pytest.skip("no headless Chromium on PATH")
    page = server.fetch_impl("https://react.dev/", render=True, max_chars=2000)
    assert "rendered" in page and "components" in page
    shot = server.screenshot_impl("https://example.com", path=str(tmp_path / "s.png"))
    assert not shot.startswith("Error"), shot
    assert server._png_size(str(tmp_path / "s.png"))[0] == 1280


@pytest.fixture(autouse=True)
def _empty_cache():
    server._CACHE.clear()
    yield
    server._CACHE.clear()


def test_paging_reuses_the_first_fetch(fetch):
    calls = []
    text = "word " * 5000

    def h(req):
        calls.append(1)
        body = f"<html><body><article><p>{text}</p><p>v{len(calls)}</p></article></body></html>"
        return httpx.Response(200, content=body, headers={"content-type": "text/html"})

    a = fetch(h, url="https://x.test/c", max_chars=1000)
    b = fetch(h, url="https://x.test/c", max_chars=1000, start_index=1000)
    assert len(calls) == 1 and "chars 1000-2000" in b
    fetch(h, url="https://x.test/c", max_chars=1000)  # a first page always refetches
    assert len(calls) == 2 and "start_index=1000" in a


def test_screenshot_does_not_report_an_old_image(monkeypatch, tmp_path):
    out = tmp_path / "shot.png"
    out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (100).to_bytes(4, "big") + (50).to_bytes(4, "big"))
    monkeypatch.setattr(server, "_browser_run", lambda extra, url: (b"", b""))  # exits 0, writes nothing
    res = server.screenshot_impl("https://x.test/", path=str(out))
    assert res.startswith("Error") and not out.exists()
