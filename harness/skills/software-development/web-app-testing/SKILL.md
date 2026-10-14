---
name: web-app-testing
description: "Test web apps in headless Chromium: flows, forms, console."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [web, testing, e2e, playwright, chromium, qa, frontend]
    category: software-development
    related_skills: [visual-check]
---

# Web app testing

A single page's look needs only `mcp__internet__web_screenshot` + `vision_analyze`. Clicks, forms,
navigation, console errors and multi-step flows need a Playwright script driving the image's
headless Chromium.

## Start the app

- Run the dev server with `terminal(background=true)`, output redirected to a log file, then wait
  until it answers: `for i in $(seq 60); do curl -sf -o /dev/null localhost:PORT && break; sleep 1; done`.
  The container shares the host's network: pick a port nothing else answers on, and stop the server
  with `process_manage` when you are done.
- Before blaming the UI, `curl -s localhost:PORT/api/...` the endpoints it calls.

## Drive it

`pip install playwright` once (it stays in ~/venv). Launch the system browser by path; the browser
`playwright install` downloads cannot start here (missing system libraries).

```python
import shutil
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=shutil.which("chromium-headless-shell"))
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    problems = []
    page.on("console", lambda m: m.type in ("error", "warning") and problems.append(f"console {m.type}: {m.text}"))
    page.on("pageerror", lambda e: problems.append(f"uncaught: {e}"))
    page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
    page.on("response", lambda r: r.status >= 400 and problems.append(f"HTTP {r.status}: {r.url}"))

    page.goto("http://localhost:PORT/")
    page.get_by_label("Email").fill("a@example.com")
    page.get_by_role("button", name="Sign in").click()
    page.get_by_text("Welcome").wait_for()          # fails with a timeout if the flow broke
    page.screenshot(path="shots/after-login.png", full_page=True)

    print("\n".join(problems) or "no console errors or failed requests")
    browser.close()
```

- Save the script as a file and rerun it after every fix; it is the regression test for the flow.
  In a project with a test suite, make it a pytest test.
- Locate elements by role, label or visible text; these survive markup changes that break CSS
  selectors.
- Decide pass or fail from DOM state (text present, URL changed, element enabled). Use screenshots
  with `vision_analyze` for layout questions only.

## What to cover

Every navigation link; each form with valid, empty and invalid input; reload in the middle of a
flow; an unknown URL (404 page); a 390x844 viewport. For each problem record the URL, steps,
expected and actual result, console lines and the screenshot path.
