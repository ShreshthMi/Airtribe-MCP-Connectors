"""The same server over the streamable HTTP transport: HTTP status codes around JSON-RPC."""
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
ACCEPT = {"Accept": "application/json, text/event-stream"}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "http-test", "version": "0"}}}


@pytest.fixture
def url(data_root):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(ROOT / "filesystem_mcp_server.py"), "--root", str(data_root),
                                "--transport", "streamable-http", "--port", str(port), "--log-level", "WARNING"],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}/mcp"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            httpx.get(base, timeout=0.5)
            break
        except httpx.TransportError:
            time.sleep(0.2)
    yield base
    process.terminate()
    process.wait(timeout=10)


def test_http_status_codes(url):
    with httpx.Client(timeout=10, trust_env=False) as http:
        assert http.post(url, json=INIT).status_code == 406  # the client must accept JSON and SSE
        started = http.post(url, json=INIT, headers=ACCEPT)
        assert started.status_code == 200 and '"protocolVersion":"2025-11-25"' in started.text
        session = {**ACCEPT, "mcp-session-id": started.headers["mcp-session-id"]}
        unknown = http.post(url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                            headers={**ACCEPT, "mcp-session-id": "no-such-session"})
        assert unknown.status_code == 404
        garbled = http.post(url, content=b'{"jsonrpc": ', headers={**session, "Content-Type": "application/json"})
        assert garbled.status_code == 400 and garbled.json()["error"]["code"] == -32700
        assert http.post(url.replace("/mcp", "/elsewhere"), json=INIT, headers=ACCEPT).status_code == 404
