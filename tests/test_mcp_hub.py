"""The multi-MCP hub: namespaced tools routed to the right server, and the agent using a
second MCP server (fetch) next to the filesystem one."""
import json
import shutil
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest
from langgraph.types import Command

import mcp_hub
from mcp_hub import HubError, MCPHub, server_parameters
from tests.fakes import BACKEND_ROLE, ScriptedClaude, reply
from tests.test_mcp_agent import TODAY, OfflineSearchCorpus, RecordingClaude, corpus  # noqa: F401 (fixture)
from matching_agent import MatchingAgent

POSTING = "Backend Engineer\nRequirements\n- Java\nNice to have\n- Kubernetes\n"


def two_folders(tmp_path):
    for name in ("alpha", "beta"):
        (tmp_path / name / "resumes").mkdir(parents=True)
        (tmp_path / name / "resumes" / "who.txt").write_text(f"I live in {name}\n")
    config = tmp_path / "servers.json"
    config.write_text(json.dumps({"mcpServers": {
        name: {"command": "${PYTHON}", "args": ["filesystem_mcp_server.py", "--root", str(tmp_path / name)]}
        for name in ("alpha", "beta")}}))
    return config


def test_tools_are_namespaced_and_each_call_goes_to_its_own_server(tmp_path):
    hub = MCPHub(two_folders(tmp_path))
    try:
        names = [t["name"] for t in hub.tools()]
        _, alpha = hub.call("alpha__read_file", {"filepath": "resumes/who.txt"})
        _, beta = hub.call("beta__read_file", {"filepath": "resumes/who.txt"})
        with pytest.raises(HubError, match="no MCP server for tool 'gamma__read_file'"):
            hub.call("gamma__read_file", {})
    finally:
        hub.close()

    assert "alpha__read_file" in names and "beta__batch_process" in names and len(names) == 16
    assert alpha["content"] == "I live in alpha\n" and beta["content"] == "I live in beta\n"
    assert hub.owns("alpha__list_files") and not hub.owns("read_file")


def test_server_settings_expand_variables_and_pass_only_the_listed_ones(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-parent-only")
    monkeypatch.setenv("SOME_PROXY", "http://proxy.local:3128")
    monkeypatch.delenv("UNSET_THING", raising=False)
    params = server_parameters({"command": "${PYTHON}", "args": ["x.py", "--flag=${SOME_PROXY}"],
                                "env": {"HTTPS_PROXY": "${SOME_PROXY}", "EMPTY": "${UNSET_THING}"}})

    assert params.command.endswith("python") or "python" in params.command
    assert params.args == ["x.py", "--flag=http://proxy.local:3128"]
    assert params.env["HTTPS_PROXY"] == "http://proxy.local:3128" and "EMPTY" not in params.env
    assert "ANTHROPIC_API_KEY" not in params.env


def test_bad_server_files_are_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"mcpServers": {"a__b": {"command": "x"}}}))
    with pytest.raises(HubError, match="without '__'"):
        mcp_hub.load_servers(bad)
    with pytest.raises(HubError, match="mcpServers format"):
        mcp_hub.load_servers(tmp_path / "missing.json")


class FakeFetchHub:
    """Stands in for an MCPHub with the fetch server, offline."""

    def __init__(self):
        self.fetched = []

    def tools(self):
        return [{"name": "fetch__fetch", "description": "[fetch MCP server] Fetches a URL",
                 "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}]

    def owns(self, name):
        return name == "fetch__fetch"

    def call(self, name, args):
        self.fetched.append(args["url"])
        return False, {"text": f"Contents of {args['url']}:\n{POSTING}"}

    def close(self):
        pass


def test_a_posting_given_by_url_is_fetched_by_one_server_and_the_report_saved_by_the_other(corpus, tmp_path):
    corpus.extra = hub = FakeFetchHub()
    claude = RecordingClaude(requirements=[BACKEND_ROLE], replies=[reply("Mo Sato meets the Java requirement.")])
    app = MatchingAgent(claude, corpus, TODAY).build()
    config = {"configurable": {"thread_id": "t"}}
    message = "Screen candidates for the posting at https://jobs.example.com/backend-engineer"

    app.invoke({"messages": [{"role": "user", "content": message}], "jd_input": message}, config)
    app.invoke(Command(resume="Who meets the Java requirement?"), config)
    state = app.get_state(config).values

    assert hub.fetched == ["https://jobs.example.com/backend-engineer"]
    assert "Backend Engineer" in state["job_description"] and state["requirements"]["role_title"] == "Backend Engineer"
    assert (tmp_path / "reports" / "backend-engineer-2026-10-31-round-1.md").is_file()  # via filesystem write_file
    assert "fetch__fetch" in {t["name"] for t in claude.offered}


@pytest.mark.skipif(shutil.which("uvx") is None, reason="needs uv's uvx to run the reference fetch server")
def test_the_real_fetch_server_reads_a_posting_from_a_local_web_server(tmp_path):
    """The reference fetch server, started by the hub as uvx mcp-server-fetch (first run downloads it)."""
    (tmp_path / "posting.txt").write_text(POSTING)
    web = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
    threading.Thread(target=web.serve_forever, daemon=True).start()
    fetch = mcp_hub.load_servers()["fetch"]  # the committed settings, plus loopback for this local test only
    config = tmp_path / "servers.json"
    config.write_text(json.dumps({"mcpServers": {"fetch": {**fetch, "args": [*fetch["args"], "--allow-private-ips"]}}}))
    hub = MCPHub(config)
    try:
        failed, page = hub.call("fetch__fetch", {"url": f"http://127.0.0.1:{web.server_port}/posting.txt"})
    finally:
        hub.close()
        web.shutdown()

    assert not failed and "Backend Engineer" in page["text"] and "Kubernetes" in page["text"]
