#!/usr/bin/env python3
"""Hermes pre_llm_call hook: name the skills that match the user's message, right next to it.

Muse rarely loads a skill from the system prompt's skills index on its own, but follows a hint placed beside the
request. This reads the hook payload (JSON on stdin), scores the enabled skills by the words of their name, tags
and description that the message shares (rarer words count more), and prints {"context": ...} naming at most
two of them, or nothing. Stdlib + PyYAML (in the image); never fails the turn.

Test: echo '{"extra": {"user_message": "find recent arXiv papers"}}' | python3 skill_router.py
"""
import json
import math
import os
import re
import sys
from pathlib import Path

HOME = Path(os.environ.get("HERMES_HOME", "/muse/hermes"))
ROOTS = [HOME / "skills", Path(os.environ.get("MUSE_SKILLS_DIR", "/muse/skills"))]
MIN_SCORE = 3.0
MAX_SKILLS = 2
STOP = set("""
a about after again all also an and any are as at be been before but by can could did do does doing done each
for from get gets got had has have help here how i if in into is it its just let like make me more my need new
no not now of on one only or other our out over please really see should show so some still such tell than that
the their them then there these they this those through to too try up us use used using very want was way we
were what when where which while who why will with within would yes you your file files code thing things
python test tests testing run work
""".split())
WORD = re.compile(r"[a-z0-9][a-z0-9+#]*")


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 2 and w not in STOP}


def stem(w: str) -> str:
    """Crude plural/verb folding so 'papers' meets 'paper' and 'videos' meets 'video'."""
    for suf in ("ings", "ing", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def frontmatter(text: str) -> dict:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    try:
        import yaml
        data = yaml.safe_load(text[3:end] if end > 0 else "") or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def disabled() -> set[str]:
    try:
        import yaml
        cfg = yaml.safe_load((HOME / "config.yaml").read_text()) or {}
        return {str(n) for n in (cfg.get("skills") or {}).get("disabled") or []}
    except Exception:
        return set()


def load_skills() -> dict[str, tuple[str, dict[str, float]]]:
    """name -> (description, {stemmed keyword: weight}); the Hermes home's tier first, as in Hermes."""
    off, out = disabled(), {}
    for root in ROOTS:
        if not root.is_dir():
            continue
        for f in sorted(root.glob("**/SKILL.md")):
            rel = f.relative_to(root).parts
            if any(p.startswith(".") for p in rel):
                continue
            try:  # one malformed skill (Muse writes its own) must not silence the others
                fm = frontmatter(f.read_text(errors="replace")[:6000])
                name = str(fm.get("name") or f.parent.name)
                if name in out or name in off or f.parent.name in off:
                    continue
                meta = fm.get("metadata")
                meta = meta.get("hermes") if isinstance(meta, dict) else None
                tags = meta.get("tags") if isinstance(meta, dict) else None
                tags = tags if isinstance(tags, list) else [tags] if isinstance(tags, str) else []
                desc = str(fm.get("description") or "")
                kw: dict[str, float] = {}
                for weight, text in ((1.0, desc), (2.0, " ".join(str(t) for t in tags)),
                                     (3.0, name.replace("-", " "))):
                    for w in words(text):
                        kw[stem(w)] = max(kw.get(stem(w), 0.0), weight)
                out[name] = (desc, kw)
            except Exception:
                continue
    return out


def user_text(payload: dict) -> str:
    msg = (payload.get("extra") or {}).get("user_message")
    if isinstance(msg, list):  # multimodal: keep the text parts
        msg = " ".join(p.get("text", "") for p in msg if isinstance(p, dict))
    return msg if isinstance(msg, str) else ""


def route(text: str, skills: dict) -> list[tuple[float, str, str]]:
    if not text.strip() or text.lstrip().startswith("/"):
        return []
    asked = {stem(w) for w in words(text)}
    df: dict[str, int] = {}
    for _, kw in skills.values():
        for w in kw:
            df[w] = df.get(w, 0) + 1
    n = max(len(skills), 1)
    scored = []
    for name, (desc, kw) in skills.items():
        score = sum(kw[w] * math.log(1 + n / df[w]) for w in asked & kw.keys())
        if score >= MIN_SCORE:
            scored.append((round(score, 2), name, desc))
    return sorted(scored, reverse=True)[:MAX_SKILLS]


def main() -> None:
    if os.environ.get("MUSE_SKILL_ROUTER", "").lower() in ("off", "0", "false", "no"):
        return  # muse reflect: reviewing past sessions, not doing a task
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        hits = route(user_text(payload), load_skills())
    except Exception:
        return
    if not hits:
        return
    names = ", ".join(f'{name} ("{desc}")' if desc else name for _, name, desc in hits)
    print(json.dumps({"context": f"[Skill router] Skills that match this request: {names}. If one fits, load it "
                                 f"with skill_view before other tools: it holds tested commands for this sandbox."}))


if __name__ == "__main__":
    if "--explain" in sys.argv:  # debugging: score every skill for the text given on the command line
        text = " ".join(a for a in sys.argv[1:] if a != "--explain")
        for hit in route(text, load_skills()) or [(0, "(no skill above the threshold)", "")]:
            print(hit)
    else:
        main()
