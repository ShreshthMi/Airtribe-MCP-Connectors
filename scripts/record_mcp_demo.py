"""
Record and render the Milestone 4 demo video: the MCP server, the agent using it, end to end.

Like Milestone 3's scripts/record_demo.py (whose terminal renderer this reuses), it is not
a screen capture: each segment's command runs under a pseudo-terminal, every chunk of output
is timestamped, and the chunks are replayed through a terminal emulator into frames.

  python scripts/record_mcp_demo.py                  # record every segment, then render demo/mcp_demo.mp4
  python scripts/record_mcp_demo.py --only chat      # record one segment (e.g. on a machine with an API key)
  python scripts/record_mcp_demo.py --replay         # render from the saved demo/segments/*.json, no recording

Segments
  tour         scripts/mcp_tour.py: handshake, discovery and errors in raw JSON-RPC (no API key)
  watch_batch  scripts/mcp_tour.py: watch_directory and batch_process (no API key)
  chat         chat.py --show-mcp --mcp-servers with demo/mcp_demo_script.txt: the live agent (API key)
  tests        the offline test suite

Captions: the tour prints "## " lines, which become captions and are kept off the screen;
for the chat they are the "## " lines in demo/mcp_demo_script.txt, as in Milestone 3.
In the chat segment any wait longer than MAX_IDLE seconds with no output is shortened to
MAX_IDLE (the title card says so); nothing else is changed or reordered.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_demo as rd  # noqa: E402  (Milestone 3's recorder: PTY recording, terminal frames, cards)
import pyte  # noqa: E402

ROOT = rd.ROOT
SEGMENT_DIR = ROOT / "demo" / "segments"
CHAT_SCRIPT = ROOT / "demo" / "mcp_demo_script.txt"
VIDEO = ROOT / "demo" / "mcp_demo.mp4"
PYTHON = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").is_file() else sys.executable
MAX_IDLE = 4.0
CHAT_HOLD = 9.0  # seconds each finished answer stays on screen before the next question
CAPTION_LINE = re.compile(r"^## (.*)$")

SEGMENTS = {
    "tour": {"title": "1. The MCP server, in raw JSON-RPC 2.0",
             "command": [PYTHON, "scripts/mcp_tour.py", "--part", "discovery,errors", "--pause", "2.2"]},
    "watch_batch": {"title": "2. watch_directory() and batch_process()",
                    "command": [PYTHON, "scripts/mcp_tour.py", "--part", "watch,batch", "--pause", "2.2"]},
    "chat": {"title": "3. The LangGraph agent, through MCP, end to end",
             "command": [PYTHON, "chat.py", "--show-mcp", "--mcp-servers", "--session", "video-demo",
                         "--script", str(CHAT_SCRIPT), "--pause", "2"]},
    "tests": {"title": "4. The offline test suite",
              "command": [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header"],
              "caption": "The offline test suite (no API key): JSON-RPC over stdio, every error-table row, watch, "
                         "batch, config, the agent through the real server process, and Milestone 3's flows."},
}

TITLE = ("Milestone 4: MCP integration", [
    "filesystem_mcp_server.py serves the Milestone 1 file tools over the Model Context Protocol",
    "(JSON-RPC 2.0, revision 2025-11-25), plus watch_directory() and batch_process().",
    "The Milestone 3 LangGraph agent now reaches every file through an MCP client, and can add",
    "other MCP servers (here the reference fetch server) from mcp_servers.json.",
    "Terminal sessions recorded on this machine and replayed at their real speed; in the live chat,",
    f"waits over {MAX_IDLE:g} s with no output are shortened to {MAX_IDLE:g} s."])
ARCHITECTURE = ("How the pieces connect", [
    "chat.py  ->  MatchingAgent (LangGraph: parse JD, search, rank, report, human feedback, agent turn)",
    "   ->  Corpus  ->  MCPClient (sync facade)  ==stdio, JSON-RPC==>  filesystem_mcp_server.py",
    "          ->  Milestone 1 fs_tools (sandbox)  ->  resumes/, job_descriptions/, reports/",
    "   ->  MCPHub  ==stdio==>  mcp-server-fetch (uvx)          tools named fetch__fetch",
    "   ->  Milestone 2 hybrid RAG search (ChromaDB + BM25) for Round 1, unchanged",
    "No agent module imports the file tools any more; a test checks that."])
END = ("Limitations", [
    "The server answers over stdio for the agent; streamable HTTP is available but not secured (no auth).",
    "Round 1's embedding index is not updated when watch_directory sees a new resume:",
    "the agent can read the new file through MCP, but search finds it only after build_index.py.",
    "More batch threads did not speed up reading the local corpus (docs/benchmarks.md).",
    "Screening recommendations support a human decision; the Milestone 3 evaluation caveats still hold."])


def strip_printed_captions(chunks: list[tuple[float, str]]) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Split '## ' lines out of the output: (chunks without them, [(time, caption), ...])."""
    kept, captions, pending, pending_at = [], [], "", 0.0
    for at, text in chunks:
        text = pending + text
        pending_at = pending_at if pending else at
        *lines, pending = text.split("\n")
        out = []
        for line in lines:
            match = CAPTION_LINE.match(rd.ANSI.sub("", line).strip("\r"))
            if match:
                captions.append((at, match.group(1)))
            else:
                out.append(line)
        if out:
            kept.append((at, "\n".join(out) + "\n"))
    if pending:
        kept.append((pending_at, pending))
    return kept, captions


def shorten_idle(chunks: list[tuple[float, str]], limit: float) -> list[tuple[float, str]]:
    shifted, removed, previous = [], 0.0, 0.0
    for at, text in chunks:
        gap = at - previous
        if gap > limit:
            removed += gap - limit
        previous = at
        shifted.append((at - removed, text))
    return shifted


def record_segment(name: str) -> None:
    spec = SEGMENTS[name]
    print(f"\n=== recording {name}: {' '.join(spec['command'][1:])}", flush=True)
    chunks = rd.record(spec["command"])
    SEGMENT_DIR.mkdir(parents=True, exist_ok=True)
    (SEGMENT_DIR / f"{name}.json").write_text(json.dumps(chunks))


def timeline(name: str) -> tuple[list[tuple[float, str]], list[tuple[float, str]], list[float]]:
    """(chunks, captions, hold points) for one recorded segment."""
    chunks = [(t, s) for t, s in json.loads((SEGMENT_DIR / f"{name}.json").read_text())]
    if name == "chat":
        chunks = shorten_idle(chunks, MAX_IDLE)
        captions = rd.caption_times(chunks, rd.read_script(CHAT_SCRIPT))
        return chunks, captions, [at for at, _ in captions[1:]]
    if "caption" in SEGMENTS[name]:
        return chunks, [(0.0, SEGMENTS[name]["caption"])], []
    chunks, captions = strip_printed_captions(chunks)
    return chunks, captions, []


def render(names: list[str]) -> float:
    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg is needed to render the video.")
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{rd.WIDTH}x{rd.HEIGHT}", "-r", str(rd.FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(VIDEO)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    total = 0.0

    def hold(image, seconds: float) -> None:
        nonlocal total
        frame = image.tobytes()
        for _ in range(round(seconds * rd.FPS)):
            encoder.stdin.write(frame)
        total += seconds

    hold(rd.draw_card(*TITLE), 9)
    hold(rd.draw_card(*ARCHITECTURE), 13)
    for name in names:
        chunks, captions, pauses = timeline(name)
        hold(rd.draw_card(SEGMENTS[name]["title"], []), 2.5)
        screen = pyte.Screen(rd.COLS, rd.ROWS)
        stream, index, caption_index = pyte.Stream(screen), 0, -1
        length = chunks[-1][0] + rd.TAIL_SECONDS
        for frame in range(int(length * rd.FPS)):
            now = frame / rd.FPS
            while pauses and pauses[0] <= now:
                hold(rd.draw_terminal(screen, captions[caption_index][1] if caption_index >= 0 else "", now / length),
                     CHAT_HOLD)
                pauses.pop(0)
            while index < len(chunks) and chunks[index][0] <= now:
                stream.feed(chunks[index][1])
                index += 1
            while caption_index + 1 < len(captions) and captions[caption_index + 1][0] <= now:
                caption_index += 1
            caption = captions[caption_index][1] if caption_index >= 0 else ""
            encoder.stdin.write(rd.draw_terminal(screen, caption, now / length).tobytes())
            total += 1 / rd.FPS
        print(f"  rendered {name}: {length:.0f}s of terminal", flush=True)
    hold(rd.draw_card(*END), 11)
    encoder.stdin.close()
    encoder.wait()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Record and render the Milestone 4 demo video.")
    parser.add_argument("--only", help=f"record just these segments (comma-separated: {', '.join(SEGMENTS)})")
    parser.add_argument("--replay", action="store_true", help="render from demo/segments/*.json without recording")
    parser.add_argument("--skip", default="", help="segments to leave out of the video")
    args = parser.parse_args()
    skip = {s for s in args.skip.split(",") if s}
    if args.only:
        for name in args.only.split(","):
            record_segment(name)
        return
    names = [n for n in SEGMENTS if n not in skip]
    if not args.replay:
        for name in names:
            record_segment(name)
    missing = [n for n in names if not (SEGMENT_DIR / f"{n}.json").is_file()]
    if missing:
        sys.exit(f"not recorded yet: {', '.join(missing)} (record with --only {','.join(missing)})")
    seconds = render(names)
    print(f"wrote {VIDEO.relative_to(ROOT)} ({seconds / 60:.1f} minutes, {VIDEO.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
