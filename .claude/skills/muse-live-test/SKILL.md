---
name: muse-live-test
description: Live-test Muse on the real model, headless or in an interactive session, and read back its tool calls and timing. Use when checking that a Muse skill, tool, config or prompt change works in practice.
---

# Live-testing Muse

Offline tests and mocks prove wiring; only the real model shows what Muse *does* (which tools it reaches for, whether it loads a skill). A mock that sends call shapes the model never sends passes while the live path fails.

## 1. Check the slot

`./muse status` must show the server running and `sessions: none`. The model has one slot: a running session is the user's, so stop and ask instead of sending traffic. `last model activity` tells a busy session from one left open.

## 2. Use a scratch Hermes home

Copy `state/hermes/config.yaml` and `state/hermes/SOUL.md` (plus `state/hermes/skills/methods/` when the user's own skills matter) into `run/cap/home/`. Tests then never write the user's sessions, memory or approval queue. Apply the config change under test to this copy first.

The agent's home folder is separate from the Hermes home: `muse` uses `state/home` unless `MUSE_HOME_DIR` is set. Both runners below default it to `run/cap/homedir`; when you call `muse` any other way, set `MUSE_HOME_DIR` yourself.

## 3. Run in the mode that matches the question

- **Interactive** (what the user gets; Hermes's strong "load matching skills" prompt, `skill_manage` available):
  `python3 .claude/skills/muse-live-test/scripts/tui-drive.py WORKDIR /abs/run/cap/home /abs/PROMPT.txt LOG [TIMEOUT_S]`
  drives `muse` in a pty and returns when the session's final answer lands in `state.db`. Afterwards `podman stop` any `muse-*` container still listed by `./muse status`.
- **One-shot** (what the evals run; Hermes tells the model to skip general skills and hides `skill_manage`):
  `.claude/skills/muse-live-test/scripts/cap-run.sh [-s HOME] /abs/PROMPT.txt ...` gives each prompt a fresh git project in `run/cap/<name>/` and a stream at `run/cap/<name>.stream.jsonl`.

Prompt paths are absolute: the runner changes into the project directory.

## 4. Read the evidence

- One-shot: `python3 .claude/skills/muse-live-test/scripts/cap-sum.py run/cap/<name>.stream.jsonl`.
- Interactive: `python3 .claude/skills/muse-live-test/scripts/session-calls.py run/cap/home SESSION_ID` (the newest `sessions` row with `source='cli'`).
- Time a run from `state.db` message timestamps (first user message to last message), never from the driver's wall clock, which includes start-up and any stalled attempt.
- Look at what the run produced (an image with Read, a file's content) before calling the result right.

Done when the report states, per prompt: the tool calls in order, whether the behaviour under test happened (e.g. `skill_view` first), the timed duration and whether the answer was checked correct.

Stop processes by PID (`for p in $(pgrep -f '[n]ame'); do kill "$p"; done`): `pkill -f name` also matches the calling shell.
