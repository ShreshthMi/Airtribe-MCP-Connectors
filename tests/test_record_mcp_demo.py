"""The Milestone 4 video recorder's timeline helpers (no recording, no ffmpeg)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import record_mcp_demo as rec  # noqa: E402


def test_printed_captions_become_captions_and_leave_the_screen():
    chunks = [(0.0, "\n## Handshake fir"), (0.5, "st\n→ {\"id\": 1}\n"), (1.0, "\x1b[0m## Errors next\n← ok\npartial")]

    kept, captions = rec.strip_printed_captions(chunks)

    assert [text for _, text in captions] == ["Handshake first", "Errors next"]
    screen = "".join(text for _, text in kept)
    assert "##" not in screen and '→ {"id": 1}' in screen and "← ok" in screen and screen.endswith("partial")


def test_only_waits_longer_than_the_limit_are_shortened():
    chunks = [(0.0, "a"), (2.0, "b"), (30.0, "c"), (31.0, "d")]

    assert rec.shorten_idle(chunks, 4.0) == [(0.0, "a"), (2.0, "b"), (6.0, "c"), (7.0, "d")]


def test_the_offline_segment_is_labelled_as_a_stand_in():
    assert {"chat", "chat_offline"} <= set(rec.SEGMENTS)
    assert "stand-in" in rec.SEGMENTS["chat_offline"]["title"]
