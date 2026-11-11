"""scripts/record_demo.py: reading the demo script and timing its captions (no recording, no API)."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("record_demo", ROOT / "scripts" / "record_demo.py")
record_demo = importlib.util.module_from_spec(_spec)
sys.modules["record_demo"] = record_demo
_spec.loader.exec_module(record_demo)


def test_each_message_takes_the_caption_written_above_it(tmp_path):
    script = tmp_path / "demo.txt"
    script.write_text("# a comment chat.py also ignores\n## First caption\nFind React people\n\n"
                      "Compare them\n## Second caption\nquit\n")

    assert record_demo.read_script(script) == [("Find React people", "First caption"),
                                               ("Compare them", "First caption"), ("quit", "Second caption")]


def test_captions_start_when_chat_echoes_the_message_even_through_colour_codes_and_wrapping():
    script = [("Find me candidates with React and 3+ years experience", "one"), ("Compare the top 3", "two")]
    chunks = [(0.4, "\x1b[1mYou:\x1b[0m Find me candidates\r\nwith React and 3+ years experience\r\n"),
              (30.0, "...a long answer...\r\n"), (41.5, "\x1b[1mYou:\x1b[0m Compare the top 3\r\n")]

    assert record_demo.caption_times(chunks, script) == [(0.4, "one"), (41.5, "two")]


def test_the_committed_demo_script_parses_and_every_message_has_a_caption():
    pairs = record_demo.read_script(record_demo.SCRIPT)

    assert len(pairs) >= 8 and pairs[-1][0] == "quit"
    assert all(caption for _, caption in pairs)


def test_a_short_session_renders_to_a_playable_video(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg is not installed")
    monkeypatch.setattr(record_demo, "TITLE_SECONDS", 1)
    monkeypatch.setattr(record_demo, "END_SECONDS", 1)
    monkeypatch.setattr(record_demo, "TAIL_SECONDS", 0)
    chunks = [(0.2, "\x1b[1mYou:\x1b[0m Find React people\r\n"), (1.0, "\x1b[32m✓ Search resumes\x1b[0m\r\n")]
    video = tmp_path / "demo.mp4"

    seconds = record_demo.render(chunks, [("Find React people", "Round 1 runs")], video)

    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, check=True)
    assert seconds == 3 and abs(float(probe.stdout) - 3) < 0.3


def test_each_later_question_adds_one_hold_to_the_video(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not installed")
    monkeypatch.setattr(record_demo, "TITLE_SECONDS", 0)
    monkeypatch.setattr(record_demo, "END_SECONDS", 0)
    monkeypatch.setattr(record_demo, "TAIL_SECONDS", 0)
    chunks = [(0.1, "You: first\r\n"), (0.5, "answer\r\n"), (1.0, "You: second\r\n"), (2.0, "done\r\n")]

    seconds = record_demo.render(chunks, [("first", "a"), ("second", "b")], tmp_path / "v.mp4", hold=1.5)

    assert seconds == 2.0 + 1.5


def test_a_character_split_across_two_terminal_reads_is_recorded_whole():
    # "│" is three bytes in UTF-8. The child writes them in two parts with a pause, so the
    # recorder gets them in two reads, as happened with macOS's smaller PTY reads.
    writer = ("import sys, time; out = sys.stdout.buffer; out.write(b'\\xe2\\x94'); out.flush(); "
              "time.sleep(0.5); out.write(b'\\x82\\n'); out.flush()")

    text = "".join(chunk for _, chunk in record_demo.record([sys.executable, "-c", writer]))

    assert "�" not in text
    assert "│" in text
