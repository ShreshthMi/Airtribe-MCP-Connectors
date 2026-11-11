"""The terminal chat loop in chat.py, driven by a script file with the scripted Claude behind it."""
from datetime import date

import pytest

import chat
from matching_agent import MatchingAgent
from tests.fakes import FakeCorpus, ScriptedClaude, reply


@pytest.fixture
def use_agent(monkeypatch):
    """Make chat.py build its agent around the given scripted Claude."""
    def install(claude: ScriptedClaude):
        monkeypatch.setattr(chat, "MatchingAgent", lambda: MatchingAgent(claude, FakeCorpus(), date(2026, 10, 31)))
    return install


def run_chat(monkeypatch, tmp_path, lines: list[str]) -> None:
    script = tmp_path / "script.txt"
    script.write_text("# a comment\n\n" + "\n".join(lines) + "\n")
    monkeypatch.setattr("sys.argv", ["chat.py", "--script", str(script)])
    chat.main()


def test_a_scripted_conversation_streams_progress_answers_and_usage(monkeypatch, tmp_path, capsys, use_agent):
    use_agent(ScriptedClaude(replies=[reply(("compare_candidates", {"candidate_ids": ["ana_ruiz", "eve_moss"]})),
                                      reply("Ana and Eve both meet the React requirement.")]))

    run_chat(monkeypatch, tmp_path, ["Find me candidates with React and 3+ years experience",
                                     "Compare the top 2", "quit"])

    out = capsys.readouterr().out
    for expected in ("Agentic Profile Matching", "✓ Extract requirements", "Round 1: top 10",
                     "✓ Rank candidates - Round 2", "Screening round 1: React Developer",
                     'tool: compare_candidates(candidate_ids=["ana_ruiz", "eve_moss"])',
                     "Ana and Eve both meet the React requirement.", "Goodbye.", "Claude usage: 0 calls"):
        assert expected in out, expected


def test_a_script_that_runs_out_of_messages_ends_the_conversation(monkeypatch, tmp_path, capsys, use_agent):
    use_agent(ScriptedClaude())

    run_chat(monkeypatch, tmp_path, ["Find me candidates with React and 3+ years experience"])

    assert "Goodbye." in capsys.readouterr().out


def test_a_failed_step_in_a_script_exits_with_an_error(monkeypatch, tmp_path, capsys, use_agent):
    claude = ScriptedClaude()
    claude.fail_next = "Rate limited by the Anthropic API."
    use_agent(claude)

    with pytest.raises(SystemExit) as exit_info:
        run_chat(monkeypatch, tmp_path, ["Find me candidates with React"])

    assert exit_info.value.code == 1
    assert "That step failed: Rate limited by the Anthropic API." in capsys.readouterr().out


def test_an_interactive_user_can_retry_a_failed_step(monkeypatch, capsys, use_agent):
    claude = ScriptedClaude()
    claude.fail_next = "Could not reach the Anthropic API."
    use_agent(claude)
    typed = iter(["Find me candidates with React", "", "quit"])  # message, Enter to retry, quit
    monkeypatch.setattr(chat.console, "input", lambda prompt="": next(typed))
    monkeypatch.setattr("sys.argv", ["chat.py"])

    chat.main()

    out = capsys.readouterr().out
    assert "That step failed: Could not reach the Anthropic API." in out
    assert "Screening round 1" in out and "Goodbye." in out


def test_closing_the_input_stream_quits_cleanly(monkeypatch, capsys, use_agent):
    use_agent(ScriptedClaude())

    def closed(prompt=""):
        raise EOFError
    monkeypatch.setattr(chat.console, "input", closed)
    monkeypatch.setattr("sys.argv", ["chat.py"])

    chat.main()  # no traceback; the first "message" is quit

    assert "Agentic Profile Matching" in capsys.readouterr().out


def test_an_unexpected_error_is_shown_not_dumped(monkeypatch, tmp_path, capsys, use_agent):
    use_agent(ScriptedClaude())
    monkeypatch.setattr(MatchingAgent, "search_resumes", lambda self, state: {}["boom"])

    with pytest.raises(SystemExit):
        run_chat(monkeypatch, tmp_path, ["Find me candidates with React"])

    assert "Unexpected error (KeyError): 'boom'" in capsys.readouterr().out


def test_an_oversized_first_message_is_refused_before_anything_runs(monkeypatch, tmp_path, capsys, use_agent):
    claude = ScriptedClaude()
    use_agent(claude)

    run_chat(monkeypatch, tmp_path, ["x" * 9000, "Find me candidates with React and 3+ years experience"])

    out = capsys.readouterr().out
    assert "That message is 9,000 characters; the limit is 8,000" in out and "Screening round 1" in out


def test_a_session_is_saved_and_resumes_with_its_shortlist_in_a_new_process(monkeypatch, tmp_path, capsys,
                                                                             use_agent):
    """Durable state: quit leaves the graph paused in SQLite; a later chat.py --session carries on."""
    use_agent(ScriptedClaude())
    first = tmp_path / "first.txt"
    first.write_text("Find me candidates with React and 3+ years experience\nquit\n")
    monkeypatch.setattr("sys.argv", ["chat.py", "--script", str(first), "--session", "acme-react"])
    chat.main()
    assert "Resume it with: python chat.py --session acme-react" in capsys.readouterr().out

    claude = ScriptedClaude(replies=[reply("Ana Ruiz `ana_ruiz` still leads the shortlist.")])
    use_agent(claude)  # a fresh agent and graph, as in a new process: only the SQLite file carries over
    second = tmp_path / "second.txt"
    second.write_text("Who leads the shortlist?\nquit\n")
    monkeypatch.setattr("sys.argv", ["chat.py", "--script", str(second), "--session", "acme-react"])
    chat.main()

    out = capsys.readouterr().out
    assert "Resuming session acme-react" in out and "React Developer, screening round 1" in out
    assert "still leads the shortlist." in out  # rendered as Markdown, so without the backticks
    assert claude.judged == []  # nothing was re-screened: the shortlist came back from the checkpoint
    assert "ana_ruiz" in claude.conversations[0][-1]["content"][0]["text"]  # the restored screening state

    monkeypatch.setattr("sys.argv", ["chat.py", "--sessions"])
    chat.main()
    assert "acme-react" in capsys.readouterr().out
