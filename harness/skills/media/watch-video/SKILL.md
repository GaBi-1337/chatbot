---
name: watch-video
description: "Watch a video (YouTube or file): what happens, on screen."
version: 1.0.0
author: muse
platforms: [linux]
metadata:
  hermes:
    tags: [video, youtube, frames, vision, screencast, contact-sheet]
    category: media
    related_skills: [youtube-transcript, visual-check]
---

# Watch a video

A transcript has the speech only. For what is shown or done (actions, slides, code, diagrams, UI), look at frames.

```bash
python ${HERMES_SKILL_DIR}/scripts/watch.py "<url or file>"
```

It prints the sheet paths with the time each covers. A sheet is up to 9 frames, each labelled
`N  m:ss`. By default one frame per ~5 s, plus one for every shot that grid skipped; at most
6 sheets whatever the length (~1200 tokens each, and each stays in your context), so a long
video is sampled coarsely.

1. `vision_analyze` each sheet in order. Ask for every tile in order: the scene, what changed since
   the previous tile, and all readable on-screen text. Keep notes per sheet.
2. Answer from the notes. A question about one stretch, or a video over ~10 min: narrow first with
   `--start 4:00 --end 4:30 --frames 9`. Anything shorter than the frame spacing (a click, a flash)
   can fall between frames, so narrow around it before saying it did not happen.
3. Overview of a long video: `--cols 4 --rows 4` gives 16 smaller frames per sheet (headlines
   stay readable, small text does not).
4. Zoom on small text or detail: `--at 1:23` writes one full frame; for a URL add `--height 720`
   (downloads the video again at 720p). `vision_analyze` with `region=[x1,y1,x2,y2]` crops
   before any downscaling.
5. Speech and names: `youtube-transcript` with `--timestamps`; line `[1:23]` belongs
   near the tile stamped `1:23`. A local file has no transcript.

Cite moments as `m:ss`; for YouTube link `https://youtu.be/<id>?t=<seconds>`. Say how densely you
sampled when the answer rests on something being absent.

Files go to `/tmp/watch-<hash>/`; the video is reused on later calls. An unavailable or private
video fails with yt-dlp's message: tell the user. Other sites work if yt-dlp supports them.
