"""
Record the demo video: a real chat session with the agent, rendered as captioned video.

Not a screen capture. chat.py replays demo/demo_script.txt under a pseudo-terminal against
the real model; every chunk of output is timestamped as it arrives, then replayed through a
terminal emulator and drawn frame by frame. The video therefore shows the agent's progress
lines, tool calls and answers appearing at the speed they really did, and nothing else
that happens to be on the screen. Captions are the "## " lines in the script; each one
switches on the moment the message after it is sent. The only change to the timing is a
pause before each new question, holding the previous answer on screen so it can be read.

    python scripts/record_demo.py            # record a live session (calls the API), then render
    python scripts/record_demo.py --replay   # re-render demo/session.json, no API calls

Adapted from Milestone 2's scripts/record_terminal_session.py. Needs pyte, Pillow and ffmpeg.
"""
from __future__ import annotations

import argparse
import codecs
import fcntl
import json
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import textwrap
import time
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"
SCRIPT = DEMO_DIR / "demo_script.txt"
SESSION = DEMO_DIR / "session.json"
VIDEO = DEMO_DIR / "demo.mp4"

COLS, ROWS = 118, 30
FPS = 10
WIDTH, HEIGHT = 1920, 1080
PAD_X, PAD_Y, LINE_H = 60, 40, 30
CAPTION_TOP = PAD_Y + ROWS * LINE_H + 22
TITLE_SECONDS, END_SECONDS, TAIL_SECONDS = 6, 8, 3
HOLD_SECONDS = 14  # each finished answer stays on screen this long before the next question

BG, FG, CAPTION_BG = (13, 17, 23), (201, 209, 217), (33, 38, 45)
COLOURS = {"default": FG, "white": (255, 255, 255), "green": (63, 185, 80), "blue": (88, 166, 255),
           "cyan": (57, 197, 187), "yellow": (210, 153, 34), "red": (248, 81, 73),
           "magenta": (188, 140, 255), "black": (110, 118, 129), "brightblack": (110, 118, 129)}
# macOS fonts first (where the Milestone 3 video was made), then their Linux counterparts.
FONT_CHOICES = {
    "mono": [("/System/Library/Fonts/Menlo.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0)],
    "mono_bold": [("/System/Library/Fonts/Menlo.ttc", 1),
                  ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 0)],
    "sans": [("/System/Library/Fonts/Helvetica.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)],
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path, index in FONT_CHOICES[kind]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size, index=index)
    raise SystemExit(f"No {kind} font found; tried {[p for p, _ in FONT_CHOICES[kind]]}")


MONO = font("mono", 22)
MONO_BOLD = font("mono_bold", 22)
SANS = font("sans", 30)
SANS_BIG = font("sans", 64)
CELL_W = MONO.getlength("M")  # measured, so columns line up with the emulator's grid
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def read_script(path: Path) -> list[tuple[str, str]]:
    """[(message, caption shown while it is answered), ...] from the demo script."""
    pairs, caption = [], ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("## "):
            caption = line[3:]
        elif line and not line.startswith("#"):
            pairs.append((line, caption))
    return pairs


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record(command: list[str]) -> list[tuple[float, str]]:
    """Run `command` under a PTY, returning (seconds since start, output) chunks."""
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    environment = {**os.environ, "TERM": "xterm-256color", "COLUMNS": str(COLS), "LINES": str(ROWS)}
    process = subprocess.Popen(command, cwd=ROOT, stdin=slave, stdout=slave, stderr=slave,
                               env=environment, close_fds=True)
    os.close(slave)
    chunks, started = [], time.monotonic()
    # One decoder for the whole stream: a character split across two reads is held until complete.
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    while True:
        ready, _, _ = select.select([master], [], [], 0.2)
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            text = decoder.decode(data)
            if text:
                chunks.append((time.monotonic() - started, text))
                sys.stdout.write(text)
                sys.stdout.flush()
        elif process.poll() is not None:
            break
    tail = decoder.decode(b"", final=True)
    if tail:
        chunks.append((time.monotonic() - started, tail))
    os.close(master)
    process.wait()
    return chunks


def caption_times(chunks: list[tuple[float, str]], script: list[tuple[str, str]]) -> list[tuple[float, str]]:
    """When each caption starts: the moment chat.py echoes the message it belongs to."""
    times, seen, position = [], "", 0
    for at, text in chunks:
        seen += ANSI.sub("", text)
        while position < len(script) and f"You: {script[position][0][:24]}" in " ".join(seen.split()):
            times.append((at, script[position][1]))
            seen = seen[-200:]  # keep only the tail, so the next search starts after this message
            position += 1
    return times


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def draw_terminal(screen: pyte.Screen, caption: str, progress: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    canvas = ImageDraw.Draw(image)
    for row in range(ROWS):
        line = screen.buffer.get(row, {})
        for col in range(COLS):
            cell = line.get(col)
            if cell is None or cell.data in ("", " "):
                continue
            colour = (255, 255, 255) if cell.bold and cell.fg == "default" else COLOURS.get(cell.fg, FG)
            canvas.text((PAD_X + col * CELL_W, PAD_Y + row * LINE_H), cell.data,
                        font=MONO_BOLD if cell.bold else MONO, fill=colour)
    canvas.rectangle([0, CAPTION_TOP, WIDTH, HEIGHT], fill=CAPTION_BG)
    for i, text in enumerate(textwrap.wrap(caption, 105)[:2]):
        canvas.text((PAD_X, CAPTION_TOP + 16 + i * 38), text, font=SANS, fill=(255, 255, 255))
    canvas.rectangle([0, HEIGHT - 6, int(WIDTH * min(progress, 1.0)), HEIGHT], fill=(88, 166, 255))
    return image


def draw_card(title: str, lines: list[str]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    canvas = ImageDraw.Draw(image)
    canvas.text((160, 300), title, font=SANS_BIG, fill=(255, 255, 255))
    for i, text in enumerate(lines):
        canvas.text((160, 420 + i * 52), text, font=SANS, fill=FG)
    return image


def render(chunks: list[tuple[float, str]], script: list[tuple[str, str]], destination: Path,
           hold: float = HOLD_SECONDS) -> float:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is needed to render the video (brew install ffmpeg).")
    captions = caption_times(chunks, script)
    session_seconds = chunks[-1][0] + TAIL_SECONDS
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(destination)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def hold_frames(image: Image.Image, seconds: float) -> None:
        frame = image.tobytes()
        for _ in range(round(seconds * FPS)):
            encoder.stdin.write(frame)

    hold_frames(draw_card("Agentic Profile Matching", [
        "A LangGraph agent that screens 100 resumes in three rounds, with a recruiter in the loop.",
        "Round 1: Milestone 2's hybrid RAG search keeps the top 10.",
        "Round 2: Claude judges each requirement as met, partial or missing, quoting the resume.",
        "Round 3: hire / borderline / no hire, with an explained, saved report.",
        "Recorded live against claude-opus-5-5 and shown at the speed it ran, with a pause",
        f"before each new question so the answer above it can be read ({hold:g} s)."]), TITLE_SECONDS)
    screen = pyte.Screen(COLS, ROWS)
    stream, index, caption_index = pyte.Stream(screen), 0, -1
    pauses = [at for at, _ in captions[1:]]  # before every question but the first
    for frame in range(int(session_seconds * FPS)):
        now = frame / FPS
        while pauses and pauses[0] <= now:  # freeze on the finished answer before the next question
            frozen = draw_terminal(screen, captions[caption_index][1] if caption_index >= 0 else "",
                                   now / session_seconds)
            hold_frames(frozen, hold)
            pauses.pop(0)
        while index < len(chunks) and chunks[index][0] <= now:
            stream.feed(chunks[index][1])
            index += 1
        while caption_index + 1 < len(captions) and captions[caption_index + 1][0] <= now:
            caption_index += 1
        caption = captions[caption_index][1] if caption_index >= 0 else ""
        encoder.stdin.write(draw_terminal(screen, caption, now / session_seconds).tobytes())
        if frame % (FPS * 30) == 0:
            print(f"  rendered {now:5.0f}s / {session_seconds:.0f}s", flush=True)
    hold_frames(draw_card("What the session showed", [
        "Natural-language search, side-by-side comparison, and 'why did A rank above B?'",
        "Changing requirements mid-conversation: only new requirements are re-judged,",
        "so every move in the ranking is traced to the change that caused it.",
        "Multi-round screening with quoted evidence, borderline suggestions and interview questions.",
        "Code, tests and the state machine diagram: see the README."]), END_SECONDS)
    encoder.stdin.close()
    encoder.wait()
    return TITLE_SECONDS + session_seconds + hold * (len(captions) - 1) + END_SECONDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and render the demo video.")
    parser.add_argument("--replay", action="store_true", help="re-render the saved session without calling the API")
    parser.add_argument("--pause", type=float, default=8.0, help="seconds chat.py waits before each scripted message")
    parser.add_argument("--hold", type=float, default=HOLD_SECONDS,
                        help="seconds the video holds each answer before the next question")
    args = parser.parse_args()
    script = read_script(SCRIPT)
    if args.replay:
        chunks = [(t, s) for t, s in json.loads(SESSION.read_text())]
    else:
        chunks = record([str(ROOT / ".venv" / "bin" / "python"), "chat.py", "--script", str(SCRIPT),
                         "--pause", str(args.pause)])
        SESSION.write_text(json.dumps(chunks))
    print(f"\nrendering {len(chunks)} output chunks ...", flush=True)
    seconds = render(chunks, script, VIDEO, hold=args.hold)
    print(f"wrote {VIDEO.relative_to(ROOT)} ({seconds / 60:.1f} minutes, {VIDEO.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
