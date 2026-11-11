"""The filesystem MCP server through a real MCP client session (in memory: real protocol
messages and SDK code on both sides, no subprocess). Every row of the README's error
table has a case here."""
import json

import pytest
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session as connect

from filesystem_mcp_server import ERROR_CODES, classify
from tests.mcp_helpers import call, make_server

pytestmark = pytest.mark.anyio

ALL_TOOLS = {"list_files", "read_file", "search_in_file", "write_file", "watch_directory", "get_watch_events",
             "unwatch_directory", "batch_process"}


@pytest.fixture
def server(data_root, monkeypatch):
    return make_server(data_root, monkeypatch)


async def test_tools_list_has_the_four_milestone1_tools_and_the_new_ones(server):
    async with connect(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    assert set(tools) == ALL_TOOLS
    assert tools["read_file"].inputSchema["required"] == ["filepath"]
    assert tools["search_in_file"].inputSchema["properties"]["whole_word"]["default"] is False
    assert tools["read_file"].annotations.readOnlyHint is True
    assert tools["write_file"].annotations.readOnlyHint is False
    assert all(t.description for t in tools.values())


async def test_the_milestone1_tools_work_through_mcp(server, data_root):
    async with connect(server) as client:
        failed, read = await call(client, "read_file", filepath="resumes/ana_ruiz.md")
        assert not failed and read["content"].startswith("# Ana Ruiz")

        _, listing = await call(client, "list_files", directory="resumes")
        assert [f["path"] for f in listing["files"]] == ["resumes/ana_ruiz.md", "resumes/ben_ito.txt"]

        _, found = await call(client, "search_in_file", filepath="resumes/ana_ruiz.md", keyword="java", whole_word=True)
        assert found["match_count"] == 1

        failed, saved = await call(client, "write_file", filepath="reports/new.md", content="# New\n")
        assert not failed and saved["success"]
    assert (data_root / "reports" / "new.md").read_text() == "# New\n"


ERROR_CASES = [
    ("read_file", {"filepath": "../.env"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "resumes/../.env"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "/etc/passwd"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "~/.ssh/id_rsa"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "secret.txt"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "resumes/.hidden.txt"}, "ACCESS_DENIED"),
    ("read_file", {"filepath": "resumes/nobody.txt"}, "NOT_FOUND"),
    ("read_file", {"filepath": "resumes/photo.png"}, "UNSUPPORTED_FORMAT"),
    ("read_file", {"filepath": "resumes/broken.pdf"}, "READ_FAILED"),
    ("list_files", {"directory": "milestone2_rag"}, "ACCESS_DENIED"),
    ("list_files", {"directory": "reports/none"}, "NOT_FOUND"),
    ("search_in_file", {"filepath": "resumes/ana_ruiz.md", "keyword": ""}, "INVALID_ARGUMENT"),
    ("write_file", {"filepath": "reports/old.md", "content": "replaced"}, "ALREADY_EXISTS"),
    ("write_file", {"filepath": "resumes/fake_candidate.md", "content": "x"}, "ACCESS_DENIED"),
    ("write_file", {"filepath": "reports/run.py", "content": "print()"}, "UNSUPPORTED_FORMAT"),
    ("get_watch_events", {"watch_id": "w99"}, "NOT_FOUND"),
    ("batch_process", {"directory": "resumes", "operation": "delete"}, "INVALID_ARGUMENT"),
]


@pytest.mark.parametrize("tool, args, code", ERROR_CASES, ids=[f"{t}-{c}-{i}" for i, (t, _, c) in enumerate(ERROR_CASES)])
async def test_tool_failures_are_isError_results_with_a_stable_code(server, data_root, tool, args, code):
    (data_root / "resumes" / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    async with connect(server) as client:
        result = await client.call_tool(tool, args)

    assert result.isError is True
    body = json.loads(result.content[0].text)
    assert body == result.structuredContent
    assert body["error_code"] == code and body["error_code"] in ERROR_CODES and body["message"]
    assert str(data_root) not in result.content[0].text  # no host paths in messages
    if tool == "write_file":
        assert (data_root / "reports" / "old.md").read_text() == "an earlier report\n"


async def test_a_file_over_the_configured_size_limit_is_too_large(data_root, monkeypatch):
    (data_root / "resumes" / "long.txt").write_text("x" * 2000)
    server = make_server(data_root, monkeypatch, "--max-file-mb", "0.001")  # about 1 KB
    async with connect(server) as client:
        failed, body = await call(client, "read_file", filepath="resumes/long.txt")
        ok, small = await call(client, "read_file", filepath="resumes/ana_ruiz.md")

    assert failed and body["error_code"] == "TOO_LARGE"
    assert not ok and small["success"]


async def test_arguments_of_the_wrong_type_are_rejected_by_the_sdk_before_the_tool_runs(server):
    async with connect(server) as client:
        result = await client.call_tool("read_file", {"filepath": 5})
        missing = await client.call_tool("read_file", {})

    assert result.isError and "validation error" in result.content[0].text
    assert missing.isError and "filepath" in missing.content[0].text


async def test_resources_are_discoverable_and_readable(server, data_root):
    (data_root / "resumes" / "late.txt").write_text("Late Applicant\n")  # after start-up: only the template knows it
    async with connect(server) as client:
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        templates = {t.uriTemplate for t in (await client.list_resource_templates()).resourceTemplates}
        resume = (await client.read_resource("resume://ana_ruiz.md")).contents[0]
        late = (await client.read_resource("resume://late.txt")).contents[0]
        config = json.loads((await client.read_resource("config://server")).contents[0].text)

    assert {"resume://ana_ruiz.md", "resume://ben_ito.txt", "jd://fe.txt", "report://old.md",
            "config://server", "stats://server"} <= uris
    assert not any(".env" in u or ".hidden" in u or "secret" in u for u in uris)
    assert templates == {"resume://{filename}", "jd://{filename}", "report://{filename}"}
    assert resume.text.startswith("# Ana Ruiz") and resume.mimeType == "text/markdown"
    assert late.text == "Late Applicant\n"
    assert config["writable_folders"] == ["reports"] and config["sources"]["root"] == "cli"
    assert config["root"] == data_root.name  # by name only: no absolute host path


@pytest.mark.parametrize("uri, code, error_code", [
    ("resume://nobody.txt", -32002, "NOT_FOUND"),
    ("bogus://anything", -32002, "NOT_FOUND"),
    ("resume://.env", -32602, "ACCESS_DENIED"),
    ("resume://photo.png", -32602, "UNSUPPORTED_FORMAT"),
])
async def test_resource_failures_are_json_rpc_errors(server, uri, code, error_code):
    async with connect(server) as client:
        with pytest.raises(McpError) as caught:
            await client.read_resource(uri)

    assert caught.value.error.code == code
    assert caught.value.error.data == {"uri": uri, "error_code": error_code}


async def test_the_dotenv_file_cannot_be_reached_by_any_route(server):
    async with connect(server) as client:
        attempts = [await client.call_tool("read_file", {"filepath": p}) for p in ("../.env", ".env", "reports/../.env")]
        uris = [str(r.uri) for r in (await client.list_resources()).resources]
        with pytest.raises(McpError):
            await client.read_resource("resume://../.env")

    assert all(a.isError and "sk-ant" not in a.content[0].text for a in attempts)
    assert not any("env" in u for u in uris)


async def test_stats_count_every_tool_call(server):
    async with connect(server) as client:
        await call(client, "read_file", filepath="resumes/ana_ruiz.md")
        await call(client, "read_file", filepath="../.env")
        await call(client, "list_files", directory="resumes")
        await client.read_resource("resume://ben_ito.txt")
        stats = json.loads((await client.read_resource("stats://server")).contents[0].text)

    assert stats["calls"] == {"list_files": 1, "read_file": 2, "resources/read": 1}


@pytest.mark.parametrize("message, code", [
    ("File not found: 'resumes/x.txt'. Did you mean 'resumes/y.txt'?", "NOT_FOUND"),
    ("Access denied: '../x' is outside the allowed folder.", "ACCESS_DENIED"),
    ("Permission denied: cannot access 'resumes/x.txt'.", "ACCESS_DENIED"),
    ("Can't read 'a.xyz': '.xyz' files aren't supported. Supported types: ...", "UNSUPPORTED_FORMAT"),
    ("Refusing to write '.py' files: write_file only creates plain-text files (.md, .txt).", "UNSUPPORTED_FORMAT"),
    ("'a.pdf' is too large to read (25.0 MB; the limit is 20.0 MB).", "TOO_LARGE"),
    ("keyword is too long (300 characters; the limit is 200).", "INVALID_ARGUMENT"),
    ("'resumes' is a folder, not a file. Use list_files to see what it contains.", "INVALID_ARGUMENT"),
    ("Could not open the PDF - the file looks damaged (PdfStreamError).", "READ_FAILED"),
])
def test_milestone1_messages_map_to_stable_codes(message, code):
    assert classify(message) == code
