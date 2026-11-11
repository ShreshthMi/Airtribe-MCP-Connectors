"""The real server process, spoken to in raw JSON-RPC 2.0 lines over stdio: no SDK on the
client side, so this checks the wire format itself, ids and error codes included."""
import json
import select
import subprocess
import sys
from pathlib import Path

import pytest

from jsonrpc_stdio import check_line

ROOT = Path(__file__).resolve().parent.parent


class RawClient:
    def __init__(self, root: Path):
        self.process = subprocess.Popen(
            [sys.executable, str(ROOT / "filesystem_mcp_server.py"), "--root", str(root), "--log-level", "WARNING"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, cwd=ROOT)

    def send(self, message) -> None:
        self.process.stdin.write((message if isinstance(message, str) else json.dumps(message)) + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float = 15.0) -> dict | None:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        return json.loads(self.process.stdout.readline()) if ready else None

    def request(self, message):
        self.send(message)
        return self.receive()

    def close(self) -> None:
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


@pytest.fixture
def raw(data_root):
    client = RawClient(data_root)
    yield client
    client.close()


def test_a_full_session_in_raw_json_rpc(raw):
    init = raw.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "raw-test", "version": "0"}}})
    assert init["jsonrpc"] == "2.0" and init["id"] == 1
    assert init["result"]["protocolVersion"] == "2025-11-25"
    assert init["result"]["serverInfo"]["name"] == "airtribe-filesystem"
    assert {"tools", "resources"} <= set(init["result"]["capabilities"])

    raw.send({"jsonrpc": "2.0", "method": "notifications/initialized"})  # a notification: no reply

    tools = raw.request({"jsonrpc": "2.0", "id": "list-1", "method": "tools/list"})  # a string id
    assert tools["id"] == "list-1" and len(tools["result"]["tools"]) == 8

    ok = raw.request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "read_file", "arguments": {"filepath": "resumes/ben_ito.txt"}}})
    assert ok["id"] == 3 and ok["result"]["isError"] is False
    assert json.loads(ok["result"]["content"][0]["text"])["content"].startswith("Ben Ito")

    denied = raw.request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                          "params": {"name": "read_file", "arguments": {"filepath": "../.env"}}})
    assert denied["id"] == 4 and denied["result"]["isError"] is True  # a tool failure is a result, not an error
    assert denied["result"]["structuredContent"]["error_code"] == "ACCESS_DENIED"

    templates = raw.request({"jsonrpc": "2.0", "id": 5, "method": "resources/templates/list"})
    assert {t["uriTemplate"] for t in templates["result"]["resourceTemplates"]} == {
        "resume://{filename}", "jd://{filename}", "report://{filename}"}


def test_protocol_errors_get_json_rpc_error_codes(raw):
    raw.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "raw-test", "version": "0"}}})
    raw.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    cases = [
        ('{"jsonrpc": "2.0", "id": 9, "method": ', None, -32700),                                # not JSON
        ([{"jsonrpc": "2.0", "id": 10, "method": "ping"}], None, -32600),                       # a batch array
        ('"just a string"', None, -32600),                                                      # not an object
        ({"jsonrpc": "1.0", "id": 11, "method": "ping"}, 11, -32600),                           # wrong version
        ({"jsonrpc": "2.0", "id": None, "method": "ping"}, None, -32600),                       # null id
        ({"jsonrpc": "2.0", "id": "u-12", "method": "files/delete_everything"}, "u-12", -32601),  # unknown method
        ({"jsonrpc": "2.0", "id": 13, "method": "tools/call", "params": {"arguments": {}}}, 13, -32602),  # no name
        ({"jsonrpc": "2.0", "id": 14, "method": "resources/read", "params": {"uri": "resume://nobody.txt"}}, 14, -32002),
        ({"jsonrpc": "2.0", "id": 15, "method": "resources/read", "params": {"uri": "resume://.env"}}, 15, -32602),
    ]
    for message, expected_id, code in cases:
        reply = raw.request(message)
        assert reply is not None, f"no reply to {message!r}"
        assert reply["jsonrpc"] == "2.0" and reply["id"] == expected_id and reply["error"]["code"] == code, reply

    raw.send({"jsonrpc": "2.0", "method": "notifications/made_up"})  # unknown notification: silently dropped
    pong = raw.request({"jsonrpc": "2.0", "id": 16, "method": "ping"})  # and the session is still healthy
    assert pong == {"jsonrpc": "2.0", "id": 16, "result": {}}


def test_nothing_but_protocol_messages_reach_stdout(data_root):
    """Logs go to stderr; a stray print on stdout would corrupt the protocol stream."""
    client = RawClient(data_root)
    try:
        first = client.request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        assert first["id"] == 1
    finally:
        client.close()
    assert client.process.stdout.read() == ""


def test_the_line_checker_passes_good_messages_through():
    message = check_line('{"jsonrpc": "2.0", "id": 1, "method": "ping"}')
    assert message.root.method == "ping" and message.root.id == 1
    assert check_line('{"jsonrpc": "2.0", "method": "notifications/initialized"}').root.method == \
        "notifications/initialized"
