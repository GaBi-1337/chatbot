"""The skill router hook (harness/hooks/skill_router.py) against Muse's own skills in harness/skills.

Bundled Hermes skills and private ones live in the Hermes home, not in the repo, so these cases use the muse
skills only (rarer words weigh more, so scores depend on the skill set: keep the cases to clear matches).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1]
ROUTER = HARNESS / "hooks" / "skill_router.py"
sys.path.insert(0, str(ROUTER.parent))
import skill_router as r  # noqa: E402


@pytest.fixture
def skills(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "HOME", tmp_path)                 # no config.yaml: nothing disabled
    monkeypatch.setattr(r, "ROOTS", [tmp_path / "skills", HARNESS / "skills"])
    return r.load_skills()


@pytest.mark.parametrize("text, first", [
    ("List the three most recently submitted arXiv papers whose titles mention speculative decoding", "arxiv-papers"),
    ("Summarize what happens in this YouTube video and when the demo starts", "watch-video"),
    ("Get the transcript of this talk on youtube and quote what they say about latency", "youtube-transcript"),
    ("Make a bar chart of these sales numbers and make sure it looks right", "visual-check"),
    ("Research what changed in Python 3.14 and cite sources", "web-research"),
    ("Click through the signup flow of my local web app and report console errors", "web-app-testing"),
])
def test_routes_to_the_matching_skill(skills, text, first):
    hits = r.route(text, skills)
    assert hits and hits[0][1] == first, hits


@pytest.mark.parametrize("text", [
    "Write a Python function that adds two numbers",
    "Rename the variable foo to bar across the project",
    "Refactor the auth module and run the tests",
    "What is 17 times 23",
    "Write a haiku about autumn",
    "/arxiv list papers",
])
def test_stays_silent_without_a_clear_match(skills, text):
    assert r.route(text, skills) == []


def test_a_malformed_skill_does_not_silence_the_others(monkeypatch, tmp_path):
    bad = tmp_path / "skills" / "x" / "bad" / "SKILL.md"
    bad.parent.mkdir(parents=True)
    bad.write_text("---\nname: bad\nmetadata: just a string\ndescription: broken\n---\n")
    monkeypatch.setattr(r, "HOME", tmp_path)
    monkeypatch.setattr(r, "ROOTS", [tmp_path / "skills", HARNESS / "skills"])
    assert r.route("find recent arXiv papers", r.load_skills())[0][1] == "arxiv-papers"


def run_hook(stdin, **env):
    full = {**os.environ, "HERMES_HOME": "/nonexistent", "MUSE_SKILLS_DIR": str(HARNESS / "skills"), **env}
    p = subprocess.run([sys.executable, "-I", str(ROUTER)], input=stdin, capture_output=True, text=True,
                       env=full, timeout=60)
    return p.returncode, p.stdout


def test_hook_prints_context_json():
    rc, out = run_hook(json.dumps({"extra": {"user_message": "find recent arXiv papers"}}))
    assert rc == 0 and "arxiv-papers" in json.loads(out)["context"]


def test_hook_reads_multimodal_text_parts():
    msg = [{"type": "text", "text": "find recent arXiv papers"}, {"type": "image_url", "image_url": {}}]
    rc, out = run_hook(json.dumps({"extra": {"user_message": msg}}))
    assert rc == 0 and "arxiv-papers" in out


@pytest.mark.parametrize("stdin", ["", "not json", "[]", "null", '{"extra": "x"}', '{"extra": {"user_message": 3}}'])
def test_hook_never_fails_a_turn(stdin):
    assert run_hook(stdin) == (0, "")


def test_hook_off_switch():
    rc, out = run_hook(json.dumps({"extra": {"user_message": "find recent arXiv papers"}}), MUSE_SKILL_ROUTER="off")
    assert (rc, out) == (0, "")
