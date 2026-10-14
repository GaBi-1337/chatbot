# muse

A Claude-Code-style terminal agent running on a local model: [Hermes Agent](https://github.com/NousResearch/hermes-agent) v0.21.7 driving **Muse Glimmer 30B** (Q4_K_M GGUF) through llama.cpp. The agent works inside a rootless Podman sandbox and has internet access through a small MCP server.

## Setup

**What you need**
- **GPU:** an NVIDIA GPU with about 22 GB of free VRAM. The default server holds the model, the speculative drafter and the vision projector at the full 131K context (see the VRAM row under [Measured performance](#measured-performance)).
- **OS:** Linux, or Windows with WSL2, with the NVIDIA driver installed. The model server runs on the host. The agent's sandbox gets the GPU as well under WSL2; on native Linux the sandbox runs without it.
- **To build llama.cpp:** the CUDA toolkit (12.8 or newer for RTX 50-series GPUs), CMake, a C++ compiler and git.
- **To run the agent:** rootless Podman (tested with 5.7) and curl.
- **To download the model:** the `hf` command from `pip install huggingface_hub`.
- **Disk:** about 20 GB for the model files, 3 GB for the agent image and 1–2 GB for the llama.cpp build.

**Steps**, from the repo root:
1. Build llama.cpp at the tested commit into `tools/llama.cpp`:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp tools/llama.cpp
   git -C tools/llama.cpp checkout cbe28bb
   cmake -S tools/llama.cpp -B tools/llama.cpp/build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
   cmake --build tools/llama.cpp/build -j --target llama-server
   ```
2. Download the model, its vision projector and the DFlash2 drafter into `Models/MuseGlimmer/`. `server/serve.sh` expects these file names. Both repos are Apache-2.0 and not gated.
   ```bash
   hf download meta-models/Muse-Glimmer-30B-GGUF Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf \
       mmproj-Muse-Glimmer-30B-Q4_K_M.gguf --local-dir Models/MuseGlimmer
   hf download z-lab/Muse-Glimmer-30B-DFlash2-GGUF Muse-Glimmer-30B-DFlash2-Q4_K_M.gguf --local-dir Models/MuseGlimmer
   ```
3. Build the agent image with `container/build.sh`. It downloads the Python base image, Hermes Agent (checksum-pinned), Node.js, the language servers and the Python packages.
4. Put `muse` on your PATH: `ln -s "$PWD/muse" ~/.local/bin/muse`.
5. In a project directory, run `muse`. The first run does three things:
   - starts the model server and waits until the model has loaded;
   - seeds the Hermes home `state/hermes` from `harness/hermes`;
   - creates the agent's persistent home `state/home`.

To work on muse itself, also install shellcheck and create the test venv for the web tools, then enable the pre-commit hook:
```bash
python3 -m venv mcp/web/.venv && mcp/web/.venv/bin/pip install -r mcp/web/requirements-dev.txt
git config core.hooksPath .githooks
```

## Usage

```bash
cd ~/some/project
muse                       # starts the model server if needed, then opens Hermes on this directory
muse --reasoning xhigh     # low | medium | high (default) | xhigh, for this session
muse shell [cmd]           # bash in the same sandbox, without Hermes
muse prompt                # the system prompt and tools a new session in this directory gets
muse reflect --stage       # review recent sessions and propose skills and memory (muse reflect -h)
muse status                # server state, VRAM, running sessions
muse stop                  # stop the server and free the VRAM
```

- The server stays up between sessions until `muse stop`.
- **Instructions:** Muse's standing instructions are `state/hermes/SOUL.md`, Hermes's identity file at the top of every system prompt. Edit it to change how Muse works; new sessions pick it up. Each past session's exact system prompt is stored in `state/hermes/state.db` (table `sessions`, column `system_prompt`).
- **Images:** Muse can see. The server loads the vision projector by default, and Hermes's `vision_analyze` tool hands an image file (a plot the agent made, a screenshot you saved in the project) straight to the model.
- **Web:** the `internet` tools search the web (DuckDuckGo) and fetch pages as markdown. They also read PDFs page by page, save images for Muse to look at, render JavaScript sites in headless Chromium, and screenshot a page, a local web app or an HTML file.
- **Skills:** procedures Muse loads when a task calls for them.
  - Its own, in `harness/skills/`: web research, arXiv papers, watching a video (timestamped contact sheets of its frames, read with vision), YouTube transcripts, checking visual output, testing web apps. Edits take effect in the next session, with no image rebuild.
  - Hermes's bundled skills for PDF, Word, Excel and PowerPoint files, and two for HTML design. `skills.disabled` in the config hides the bundled skills that need tools or accounts the sandbox lacks.
  - Skills you add go to `state/hermes/skills/` (private, not tracked); skills Muse writes go there once you approve them.
  - **Skill router:** Muse rarely loads a skill from the skills list on its own, but follows a hint next to the request. `harness/hooks/skill_router.py`, a Hermes `pre_llm_call` hook, scores the enabled skills against each message and names at most two. With it, an arXiv question took 7 tool calls instead of 36 and found the newest papers.
- **Subagents:** Muse can hand a side task that would flood its context, like searching a large codebase or reading many pages, to a subagent with `delegate_task`. Subagents take turns on the one model slot. When a subagent returns, llama-server restores Muse's own context from its RAM prompt cache instead of recomputing it, so a subagent costs about its own prompt plus its work.
- **Learning during sessions:** every ~10 turns, Hermes's background review re-runs the conversation on the same model (reusing its cache) and proposes memory and skill saves. Every save waits for your approval: `/memory pending` and `/skills pending` in a session list them, and Muse's own memory saves ask you inline. A message you send while a review runs waits for it.
- **Learning between sessions:** `muse reflect` has Muse review the sessions since the last run and turn recurring procedures, mistakes and your corrections into skills and memory. It runs only when no session uses the model. Each run may change at most 3 distinct skills and write at most 5 memory entries and writes a report to `state/hermes/reflect/<time>/`. With `--stage`, nothing is applied: approve or reject each change in a session with `/skills pending` and `/memory pending`.
- **Code checks:** after each edit in a git repo, language servers report the errors it introduced: pyright for Python, typescript-language-server, and bash-language-server with shellcheck.

Run the eval suite (results go to `evals/results/<timestamp>/summary.md`):
```bash
evals/run.sh -n 3 -t 1200 -r high --start T1 T2 T3 T4
```
To check the pipeline without a GPU, start the mock model server `evals/lib/mock_server.py PORT LOGFILE`, which answers every request with fixed text, and pass `--upstream PORT`.

Rebuild the agent image after changing `container/` or `mcp/web/` (skills and config need no rebuild):
```bash
container/build.sh
```

Run the repo checks with `scripts/check.sh`: shellcheck, the `mcp/web` tests, the `harness` tests (reflect's limits, the skill router) and the eval selftest. GitHub Actions runs the same script on every push, without the selftest. No GPU is needed. The pre-commit hook runs them too, once it's enabled for a clone with `git config core.hooksPath .githooks`. The selftest queries live APIs, so commit with `--no-verify` when offline.

## How it fits together

```
terminal ── muse ── podman run (rootless, --network=host, only $PWD mounted, GPU shared)
                      └─ hermes (image muse-agent, /opt/hermes)
                           ├─ terminal / file / code / todo tools  → $PWD (language servers check each edit)
                           ├─ vision_analyze: image files → the model, natively
                           ├─ skills: harness/skills (read-only) + curated bundled ones + Muse's own;
                           │          hook harness/hooks/skill_router.py names matching ones per message
                           ├─ delegate_task: one subagent at a time on the same model slot
                           ├─ MCP "internet": web_search (DuckDuckGo), web_fetch (pages, PDFs, images,
                           │                 JavaScript via headless Chromium; paged), web_screenshot
                           └─ OpenAI API → http://127.0.0.1:8080/v1
host ── server/serve.sh ── llama-server (Muse Q4_K_M + DFlash2 drafter + vision projector, 131K ctx)
```

| Path | What it is |
|---|---|
| `muse` | Launcher: sandbox, server start, Hermes home, persistent home, `.env` masking, `muse prompt` |
| `server/serve.sh` | `start/stop/restart/status [--no-vision] [--no-draft]` for the tuned llama-server |
| `container/` | `Containerfile` and `build.sh` for the `muse-agent` image |
| `harness/hermes/config.yaml` | Hermes home template. It is copied to `state/hermes` on first launch and fresh for each eval run |
| `harness/skills/` | Muse's own skills, mounted read-only at `/muse/skills` in sessions and eval runs |
| `harness/hooks/` | The skill router hook, mounted read-only at `/muse/hooks` |
| `harness/reflect.py` | The headless run behind `muse reflect`: session digest, write limits, report |
| `mcp/web/` | stdio MCP server with `web_search`, `web_fetch` and `web_screenshot`, plus its tests |
| `evals/` | Tasks T1–T7 (fixture, prompt, checker), `run.sh`, analysis helpers, `selftest.sh` |
| `tools/llama.cpp/` | llama.cpp build `cbe28bb` (gitignored) |
| `Models/`, `run/`, `state/`, `evals/results/` | Weights, logs, benchmarks and their notes, agent state and eval outputs (gitignored) |

## Sandbox

- **What the agent sees:** only the project directory, mounted at the same path and read-write, plus its own Hermes home, its persistent home and its skills (read-only).
- **Tools in the image:**
  - CLIs: gcc/g++ and make, Node.js 24 LTS, uv, ruff, sqlite3, ripgrep, fd, jq, poppler (`pdftotext`, `pdftoppm`), graphviz, headless Chromium, headless LibreOffice (`soffice`, to convert and render Word, Excel and PowerPoint files) and yt-dlp. A static ffmpeg comes with imageio-ffmpeg (path: `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`); it is not on PATH.
  - Python packages: numpy, pandas, matplotlib, pillow, requests, httpx, beautifulsoup4, lxml, pyyaml, pypdf, pdfplumber, reportlab, openpyxl, python-docx, python-pptx and youtube-transcript-api.
  - They load fast because they live in the image, not on a slow mount.
- **Persistent home:** `state/home` is the agent's home (`/home/muse`) in every session and project, created on first launch.
  - `python` and `pip` there are a global venv (`~/venv`), which also sees the image's packages. It is active in the agent's shell, so `uv pip` installs there too.
  - `npm install -g`, pipx or uv tools and programs in `~/.local/bin` persist too. apt packages can't (the agent has no root), so add those to `container/Containerfile`.
  - Install something yourself with `muse shell pip install …`.
  - **Reset**, with no session running (`muse status` lists them), from the repo: `rm -rf state/home/venv` drops only the Python packages; `rm -rf state/home` drops everything (packages, tools, dotfiles). The next `muse` or `muse shell` rebuilds what's missing.
  - It sits with the repo, so on a slow mount (such as a Windows drive under WSL) Python starts slower, by seconds for `pip` and heavy imports. Set `MUSE_HOME_DIR` to move it.
- **Secrets:** existing top-level `.env*` files are masked: they read as empty and are read-only. No new ones are created.
- **Approvals:** the agent runs with auto-approve (`--yolo`) inside the sandbox.
- **Network:** open (host network), so the agent reaches the model server on 127.0.0.1 and the web.

## Context window

- **Window:** 131,072 tokens (128K), the model's trained maximum, on a single server slot.
- **Compaction:** Hermes summarizes older turns once the conversation reaches **97%** of the window (~127K tokens; `compression.threshold` in `harness/hermes/config.yaml`). Its summary request is bounded at about 40K tokens in and 10K out, so compaction always fits.
- **Per-call caps:** a command's output is cut to 30,000 characters, keeping its head and tail. One `read_file` call returns at most 50,000 characters, so longer files are read in pages. Both are lower than Hermes's defaults, which are sized for larger windows (`tool_output`, `file_read_max_chars`).

## Measured performance

Measured with a llama-server bench and real Hermes runs on one machine. `serve.sh` has since switched to the DFlash2 drafter, which is about 8% faster (see the engine bake-off below).

| | No drafter | DFlash drafter, `--spec-draft-n-max 15` |
|---|---|---|
| Decode, code-heavy bench | 31 tok/s | 67–72 tok/s |
| Decode, real Hermes eval runs | – | 81–118 tok/s on coding, 64–83 tok/s on web research |
| Prefill, 15k-token prompt | 1260 tok/s | ~1000 tok/s |
| One agent turn at 15k ctx (+1.4k tool result) | 3.7–4.1 s | 2.2–2.9 s |
| VRAM at 131K ctx | 17.6 GB | 20.0 GB (21.3 GB with the vision projector, the default now) |

- **Prefix caching:** the conversation so far stays cached between turns, so each turn only processes its new tokens. In the evals, 76–97% of prompt tokens came from the cache.
- **Drafter setting:** `--spec-draft-n-max 15`, the value the DFlash2 drafter's model card recommends (its draft block is 16 tokens). llama.cpp's default of 3 leaves most of the speedup unused.

## Engine bake-off

Every candidate was measured back to back with the same bench:

| Engine and model | Decode | Agent turn (15k ctx) | VRAM | Tool calls |
|---|---|---|---|---|
| **llama.cpp, Q4_K_M + DFlash2 drafter (now the default)** | **77.5 tok/s** | 2.05–2.56 s | 19.9 GB | 9/9 |
| llama.cpp, Q4_K_M + original DFlash drafter | 71.5 tok/s | 2.28–2.55 s | 20.1 GB | 9/9 |
| ExLlamaV3 + TabbyAPI, EXL3 4.0 bpw + EXL3 drafter | 53 tok/s | 2.96–3.81 s | 17.9 GB | 9/9 |
| vLLM 0.31, INT4 + fp8 DFlash drafter | 40–46 tok/s | 3.15–7.09 s | 23.2 GB | 9/9 |

- **DFlash2:** it changes only the draft model, so the main model's outputs are unchanged. With it, the eval suite again passed 12/12, with overall decode at 92 tok/s.
- **ExLlamaV3:** its draft-verification step is slower.
- **vLLM:** it only starts with a forced KV-cache size, because its own memory check leaves no room for the KV cache.
- **Ruled out before measuring:** NVFP4 checkpoints with a drafter (too big for the GPU) and TensorRT-LLM (no support for the model). SGLang was skipped because it faces the same memory squeeze as vLLM.

## Eval results

Hermes ran headless with reasoning `high` on the default server config, 3 runs per task:

| Task | What it tests | Pass | Median wall time |
|---|---|---|---|
| T1 | Fix a bug so the existing (checksummed) tests pass | 3/3 | 62 s |
| T2 | Two-file feature, graded by hidden tests | 3/3 | 112 s |
| T3 | Shell investigation of a 427-file tree, answered as JSON | 3/3 | 94 s |
| T4 | Web: newest llama.cpp release tag | 3/3 | 64 s |

All 12 runs passed, and none of the 195 tool calls was malformed. The suite takes about 19 minutes. These tasks are too easy to tell configurations apart, so three harder ones were added. Each passed its one calibration run:

| Task | What it tests | Wall time | Requests |
|---|---|---|---|
| T5 | Multi-file refactor from float dollars to integer cents, graded by hidden tests | 487 s | 69 |
| T6 | Debugging: wrong totals whose cause sits away from the failing tests | 310 s | 33 |
| T7 | Web synthesis: oldest supported Python and the newest numpy/scipy/pandas for it, checked live | 296 s | 18 |

Use `-n 3` on T5–T7 to compare configurations such as reasoning levels.

## Known issues

- **Truncated tool calls:** llama-server returns HTTP 500 when the history holds a tool call that was cut off. Hermes sends no `max_tokens`, so the one way left to hit this is the context edge: with compaction at 97%, a single reply longer than ~3.9K tokens that starts just below the threshold gets cut off at the window. Lower `compression.threshold` (e.g. to 0.90) if that happens.
- **TypeScript diagnostics:** the first edit in a session usually gets none, because the TypeScript server is still starting; later edits do. Loose `.ts` files without a `tsconfig.json` get them unreliably.
- **Rollback is off:** Hermes can snapshot the project before each turn's first edit (`/rollback`), but every snapshot rereads the whole project. For a 500-file project on a slow mount that took 6–23 s per edited turn. Turn it on with `checkpoints: {enabled: true}` in `state/hermes/config.yaml` for projects on a fast disk.

