#!/usr/bin/env python3
"""Print a YouTube video's transcript. Needs: pip install youtube-transcript-api

  transcript.py URL_OR_ID [--lang de,en] [--timestamps] [--list]
"""
import argparse
import re
import sys


def video_id(s: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    sys.exit(f"not a YouTube URL or video ID: {s}")


def stamp(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--lang", default="en", help="preferred languages, comma-separated (default en)")
    p.add_argument("--timestamps", action="store_true", help="prefix each line with m:ss")
    p.add_argument("--list", action="store_true", help="list the available transcript languages")
    args = p.parse_args()
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        sys.exit("missing package: pip install youtube-transcript-api")
    vid, api = video_id(args.video), YouTubeTranscriptApi()
    try:
        if args.list:
            for t in api.list(vid):
                print(f"{t.language_code}\t{t.language}\t{'generated' if t.is_generated else 'manual'}")
            return
        fetched = api.fetch(vid, languages=[x.strip() for x in args.lang.split(",") if x.strip()])
    except Exception as e:  # the library raises one class per failure; its message explains it
        sys.exit(f"{type(e).__name__}: {str(e).strip()[:1500]}")
    for s in fetched:
        text = " ".join(s.text.split())
        print(f"[{stamp(s.start)}] {text}" if args.timestamps else text)


if __name__ == "__main__":
    main()
