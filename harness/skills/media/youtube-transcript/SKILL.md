---
name: youtube-transcript
description: "YouTube transcripts: summarise, quote or search a video."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [youtube, video, transcript, captions, summary]
    category: media
    related_skills: [web-research]
---

# YouTube transcript

web_fetch gets no useful text from a video page; read the captions instead.

```bash
pip install youtube-transcript-api          # once; it stays in ~/venv
python ${HERMES_SKILL_DIR}/scripts/transcript.py "<url or 11-char id>" --timestamps > /tmp/yt.txt
wc -w /tmp/yt.txt                            # speech runs ~150 words a minute
```

- Options: `--lang de,en` (preference order), `--list` (languages the video has; "generated" means
  YouTube's automatic captions, which have no punctuation and misspell names).
- Read the file with read_file; for a long video, read it in parts and keep notes per part instead
  of loading the whole transcript at once.
- Title, channel and thumbnail: `curl -s "https://www.youtube.com/oembed?url=<url>&format=json"`.
- Errors: `NoTranscriptFound` names the languages that do exist; `TranscriptsDisabled` means the
  video has no captions; `RequestBlocked` or `IpBlocked` means YouTube refused this connection. In
  the last two cases tell the user: there is no way to get the speech here.
- Quote with timestamps and link to the moment: `https://youtu.be/<id>?t=<seconds>`.
