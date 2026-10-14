#!/usr/bin/env python3
"""Turn a video into timestamp-labelled contact sheets, or pull one frame.
Needs: pip install yt-dlp imageio-ffmpeg pillow

  watch.py SOURCE                       # sheets of the whole video
  watch.py SOURCE --start 1:20 --end 2:00 --frames 18
  watch.py SOURCE --at 1:23 [--height 720]   # one full-size frame

SOURCE is a local file or a URL (YouTube or any site yt-dlp supports). Files go to
$TMPDIR/watch-<hash of SOURCE>/ (--out to change); a downloaded video is reused on later calls.
Times are seconds or [h:]m:ss.
"""
import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SHEET_WIDTH = 1290      # px; the vision model reads about one token per 28x28 px
SHEET_BYTES = 190_000   # Hermes re-encodes images larger than ~192 KB, which blurs text


def die(msg: str) -> None:
    sys.exit(f"watch.py: {msg}")


def parse_time(s: str) -> float:
    try:
        parts = [float(x) for x in s.split(":")]
    except ValueError:
        die(f"bad time {s!r}; use seconds or [h:]m:ss")
    if not 1 <= len(parts) <= 3:
        die(f"bad time {s!r}; use seconds or [h:]m:ss")
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def stamp(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg") or die("missing package: pip install imageio-ffmpeg")


def probe(ffmpeg: str, video: str):
    """(duration s, width, height); ffmpeg has no ffprobe here, so read its banner."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", video], capture_output=True, text=True, timeout=60)
    d = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    v = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", r.stderr)
    if not v:
        die("no video stream found:\n" + r.stderr.strip()[-500:])
    dur = int(d[1]) * 3600 + int(d[2]) * 60 + float(d[3]) if d else 0.0
    return dur, int(v[1]), int(v[2])


def download(url: str, workdir: Path, height: int) -> Path:
    """Video-only download (frames need no audio, so nothing is merged)."""
    import importlib.util
    if importlib.util.find_spec("yt_dlp") is None:
        die("missing package: pip install yt-dlp")
    for old in workdir.glob("video.*"):
        old.unlink()
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-simulate", "--quiet", "--no-warnings",
           "-f", f"bv*[height<={height}]/b[height<={height}]/worst",
           "-P", str(workdir), "-o", "video.%(ext)s",
           "--print", "%(title)s | %(uploader)s | %(duration_string)s", "--print", "after_move:filepath"]
    if shutil.which("node"):
        cmd += ["--js-runtimes", "node"]  # YouTube's signature challenge needs a JS runtime
    for attempt in range(3):  # YouTube answers an occasional 403 that the next try does not
        r = subprocess.run(cmd + [url], capture_output=True, text=True, timeout=1200)
        if r.returncode == 0:
            break
    else:
        die("download failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    lines = r.stdout.strip().splitlines()
    if len(lines) < 2:
        die("download printed no file path")
    print("video:", lines[-2])
    return Path(lines[-1])


def grab(ffmpeg: str, video: str, t: float, out: Path, width: int = 0) -> None:
    vf = ["-vf", f"scale='min({width},iw)':-2"] if width else []
    r = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", video, "-frames:v", "1",
                        *vf, "-q:v", "2", str(out)], capture_output=True, text=True, timeout=120)
    if r.returncode or not out.exists():
        die(f"no frame at {stamp(t)}: {r.stderr.strip()[-300:]}")


def scene_times(ffmpeg: str, video: str, start: float, end: float, threshold: float) -> list:
    r = subprocess.run([ffmpeg, "-hide_banner", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", video,
                        "-an", "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=1200)
    return [start + float(m) for m in re.findall(r"pts_time:(\d+(?:\.\d+)?)", r.stderr)]


def pick_times(ffmpeg, video, start, end, n, fill, threshold):
    """n moments, the middle of n equal slices. With fill, every shot (1 s or longer) that no slice
    lands in gets one more frame at its middle, at most n/2 extra, longest shots first."""
    span = (end - start) / n
    times = [start + (i + 0.5) * span for i in range(n)]
    if fill:
        bounds = [start] + scene_times(ffmpeg, video, start, end, threshold) + [end]
        shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 1.0]
        missed = [(b - a, (a + b) / 2) for a, b in shots if not any(a <= t < b for t in times)]
        times += [t for _, t in sorted(missed, reverse=True)[: n // 2]]
    return sorted(times)


def font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def save_jpeg(img, path: Path, budget: int = SHEET_BYTES) -> int:
    for q in (88, 80, 72, 64, 56):
        img.save(path, "JPEG", quality=q)
        if path.stat().st_size <= budget:
            break
    return path.stat().st_size


def make_sheets(frames: list, cols: int, rows: int, outdir: Path, tag: str) -> list:
    from PIL import Image, ImageDraw
    gap = 4
    tw = (SHEET_WIDTH - gap * (cols - 1)) // cols
    first = Image.open(frames[0][1])
    th = round(tw * first.height / first.width)
    f = font(max(14, tw // 13))
    per = cols * rows
    sheets = []
    for s in range(0, len(frames), per):
        chunk = frames[s:s + per]
        nrows = math.ceil(len(chunk) / cols)
        sheet = Image.new("RGB", (SHEET_WIDTH, nrows * th + (nrows - 1) * gap), "black")
        draw = ImageDraw.Draw(sheet)
        for k, (t, path) in enumerate(chunk):
            x, y = (k % cols) * (tw + gap), (k // cols) * (th + gap)
            sheet.paste(Image.open(path).convert("RGB").resize((tw, th), Image.LANCZOS), (x, y))
            label = f"{s + k + 1}  {stamp(t)}"
            box = draw.textbbox((x + 4, y + 4), label, font=f)
            draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill="black")
            draw.text((x + 4, y + 4), label, fill="yellow", font=f)
        out = outdir / f"sheet-{tag}-{len(sheets) + 1}.jpg"
        size = save_jpeg(sheet, out)
        sheets.append((out, chunk[0][0], chunk[-1][0], size, sheet.size))
    return sheets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file or URL")
    p.add_argument("--start", default="0", help="first moment to cover")
    p.add_argument("--end", help="last moment to cover (default: end of video)")
    p.add_argument("--frames", type=int, help="uniform frames (default: one per 5 s, 9 to 36); the scene-cut pass adds up to half as many")
    p.add_argument("--no-fill", action="store_true", help="uniform frames only: skip the scene-cut pass")
    p.add_argument("--threshold", type=float, default=0.3, help="scene-cut sensitivity, lower finds more (0.3)")
    p.add_argument("--cols", type=int, help="tiles per row (3 landscape, 4 portrait)")
    p.add_argument("--rows", type=int, help="rows per sheet (3 landscape, 2 portrait)")
    p.add_argument("--at", help="extract one full-size frame at this time instead of sheets")
    p.add_argument("--height", type=int, default=480, help="download height limit (480); 720 with --at for small text")
    p.add_argument("--out", help="working directory")
    a = p.parse_args()

    src = a.source
    is_url = re.match(r"https?://", src) is not None
    if not is_url and not os.path.isfile(src):
        die(f"not a file or URL: {src}")
    key = hashlib.sha1((src if is_url else os.path.abspath(src)).encode()).hexdigest()[:8]
    workdir = Path(a.out) if a.out else Path(tempfile.gettempdir()) / f"watch-{key}"
    workdir.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_exe()

    if is_url:
        vdir = workdir if a.height == 480 else workdir / f"h{a.height}"
        vdir.mkdir(exist_ok=True)
        have = sorted(vdir.glob("video.*"))
        video = str(have[0]) if have else str(download(src, vdir, a.height))
    else:
        video = src
    dur, w, h = probe(ffmpeg, video)

    if a.at:
        from PIL import Image
        t = parse_time(a.at)
        if dur and t >= dur:
            die(f"{stamp(t)} is past the end; the video is {stamp(dur)} long")
        out = workdir / f"frame-{int(t)}s-{h}p.jpg"
        grab(ffmpeg, video, t, out, width=1500)  # Hermes halves images wider than 1568 px
        size = save_jpeg(Image.open(out).convert("RGB"), out)
        print(f"frame: {out}  ({stamp(t)}, source {w}x{h}, {size // 1000} KB)")
        return

    if dur <= 0:
        die("could not read the duration; pass --end")
    start = parse_time(a.start)
    end = min(parse_time(a.end), dur) if a.end else dur
    if end - start < 1:
        die(f"empty range {stamp(start)}-{stamp(end)} (video is {stamp(dur)} long)")
    n = a.frames or min(36, max(9, round((end - start) / 5)))
    landscape = w >= h
    cols = a.cols or (3 if landscape else 4)
    rows = a.rows or (3 if landscape else 2)
    times = pick_times(ffmpeg, video, start, end, n, not a.no_fill, a.threshold)

    fdir = workdir / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir()
    frames = []
    for i, t in enumerate(times):
        path = fdir / f"{i + 1:03d}.jpg"
        grab(ffmpeg, video, t, path, width=640)
        frames.append((t, path))
    tag = f"{int(start)}-{int(end)}s"  # a narrower range later never overwrites these sheets
    for old in workdir.glob(f"sheet-{tag}-*.jpg"):
        old.unlink()
    sheets = make_sheets(frames, cols, rows, workdir, tag)
    shutil.rmtree(fdir)

    print(f"video {w}x{h}, {stamp(dur)} long; {len(frames)} frames over {stamp(start)}-{stamp(end)}, "
          f"about one every {(end - start) / len(frames):.1f} s")
    for k, (path, t0, t1, size, (sw, sh)) in enumerate(sheets):
        print(f"sheet: {path}  {stamp(t0)}-{stamp(t1)}  {sw}x{sh}  {size // 1000} KB")


if __name__ == "__main__":
    main()
