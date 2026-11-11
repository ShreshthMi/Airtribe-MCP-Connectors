"""The agent with a real Corpus: every file operation goes to the real filesystem MCP server
process over stdio. Only Round 1's embedding search (no index in a temp folder) and Claude
(ScriptedClaude) are stand-ins, so this runs offline."""
import json
import os
import signal
import subprocess
import sys
from datetime import date

import pytest
from langgraph.types import Command

from corpus import Corpus
from matching_agent import MatchingAgent
from tests.fakes import RESUMES, FakeCorpus, ScriptedClaude, reply

TODAY = date(2026, 10, 31)
REACT_QUERY = "Find me candidates with React and 3+ years experience"


class OfflineSearchCorpus(Corpus):
    """The real Corpus, with FakeCorpus's word-overlap ranking in place of the embedding search."""

    def search(self, query, k=10, must_haves=()):
        return FakeCorpus().search(query, k, must_haves)


class RecordingClaude(ScriptedClaude):
    """ScriptedClaude that also keeps the tool schemas it was offered."""

    def converse(self, *, system, messages, tools):
        self.offered = tools
        return super().converse(system=system, messages=messages, tools=tools)


@pytest.fixture
def corpus(tmp_path):
    for folder in ("resumes", "job_descriptions", "reports"):
        (tmp_path / folder).mkdir()
    for cid, (_, text) in RESUMES.items():
        (tmp_path / "resumes" / f"{cid}.txt").write_text(text)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-not-a-real-key\n")
    db = tmp_path / "db"
    db.mkdir()
    (db / "candidates.json").write_text(json.dumps({"candidates": [
        {"candidate_name": name, "resume_path": f"resumes/{cid}.txt"} for cid, (name, _) in RESUMES.items()]}))
    opened = OfflineSearchCorpus(root=tmp_path, db_path=db)
    yield opened
    opened.close()


def screen(corpus, claude):
    app = MatchingAgent(claude, corpus, TODAY).build()
    config = {"configurable": {"thread_id": "t"}}
    app.invoke({"messages": [{"role": "user", "content": REACT_QUERY}], "jd_input": REACT_QUERY}, config)
    return app, config


def test_a_screening_through_mcp_gives_the_same_result_as_milestone3(corpus, tmp_path):
    app, config = screen(corpus, ScriptedClaude())
    via_mcp = app.get_state(config).values
    reference_app, reference_config = screen(FakeCorpus(), ScriptedClaude())  # Milestone 3's in-memory stand-in
    reference = reference_app.get_state(reference_config).values

    assert [(e["candidate_id"], e["fit_score"], e["recommendation"]) for e in via_mcp["shortlist"]] == \
        [(e["candidate_id"], e["fit_score"], e["recommendation"]) for e in reference["shortlist"]]
    assert via_mcp["report_path"] == reference["report_path"] == "reports/react-developer-2026-10-31-round-1.md"
    assert "## Shortlist" in (tmp_path / "reports" / "react-developer-2026-10-31-round-1.md").read_text()


def test_every_file_operation_the_agent_made_reached_the_server(corpus):
    screen(corpus, ScriptedClaude())

    served = json.loads(corpus.mcp.read_resource("stats://server"))["calls"]
    sent = corpus.mcp.calls
    assert served["read_file"] == sent["tools/call read_file"] >= 10  # Round 2 read the 10 resumes it judged
    assert served["write_file"] == sent["tools/call write_file"] == 2  # the Markdown and JSON reports
    assert served["list_files"] == sent["tools/call list_files"] >= 1  # checking report names are free


def test_claude_is_offered_the_servers_tools_and_can_use_one_the_agent_has_no_code_for(corpus, tmp_path):
    claude = RecordingClaude(replies=[
        reply(("read_file", {"filepath": "resumes/ana_ruiz.txt"}),
              ("batch_process", {"directory": "resumes", "operation": "search", "keyword": "React"}),
              ("read_file", {"filepath": "../.env"}),
              ("write_file", {"filepath": "reports/react-developer-2026-10-31-round-1.md", "content": "overwrite"})),
        reply("Ana's resume mentions React; 8 resumes do.")])
    app, config = screen(corpus, claude)

    app.invoke(Command(resume="Open Ana's resume and count who mentions React"), config)

    offered = {t["name"]: t for t in claude.offered}
    served = {t["name"]: t for t in corpus.mcp_tools()}
    assert set(served) <= set(offered) and {"search_resumes", "compare_candidates"} <= set(offered)
    assert offered["read_file"]["input_schema"] == served["read_file"]["input_schema"]  # straight from tools/list
    read, batch, secret, overwrite = claude.conversations[1][-1]["content"]
    assert not read["is_error"] and read["content"].startswith('<file path="resumes/ana_ruiz.txt">')
    assert not batch["is_error"] and batch["content"].startswith('<mcp_result tool="batch_process">')
    assert json.loads(batch["content"].split("\n", 1)[1].rsplit("\n", 1)[0])["ok"] == len(RESUMES)
    assert secret["is_error"] and "Access denied" in secret["content"] and "sk-ant" not in secret["content"]
    assert overwrite["is_error"] and "already exists" in overwrite["content"]
    assert (tmp_path / "reports" / "react-developer-2026-10-31-round-1.md").read_text() != "overwrite"


def _server_pid(root) -> int:
    found = subprocess.run(["pgrep", "-f", f"filesystem_mcp_server.py --root {root}"], capture_output=True, text=True)
    pids = [int(p) for p in found.stdout.split()]
    assert len(pids) == 1, found
    return pids[0]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc and uses pgrep")
def test_the_server_process_does_not_get_the_api_key(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-parent-only")
    corpus.read_file("resumes/ana_ruiz.txt")  # starts the server

    environ = open(f"/proc/{_server_pid(tmp_path.resolve())}/environ", "rb").read()

    assert b"ANTHROPIC" not in environ and b"PATH=" in environ


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses pgrep")
def test_a_dead_server_is_restarted_once(corpus, tmp_path):
    assert corpus.read_file("resumes/ana_ruiz.txt")["success"]
    first = _server_pid(tmp_path.resolve())

    os.kill(first, signal.SIGKILL)

    assert corpus.read_file("resumes/ben_ito.txt")["content"].startswith("Ben Ito")
    assert _server_pid(tmp_path.resolve()) != first
