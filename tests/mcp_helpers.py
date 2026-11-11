"""Shared helpers for the MCP tests: build a server over a temporary data folder, call tools."""
from __future__ import annotations

import anyio

from filesystem_mcp_server import build_server
from server_config import load_config


def make_server(root, monkeypatch, *flags):
    """A server confined to `root`. FS_ASSISTANT_ROOT is process-global, so monkeypatch restores it."""
    monkeypatch.setenv("FS_ASSISTANT_ROOT", str(root))
    return build_server(load_config(["--root", str(root), *flags], env={}))


async def call(client, name: str, **args) -> tuple[bool, dict]:
    result = await client.call_tool(name, args)
    return result.isError, result.structuredContent


async def wait_for_events(client, watch_id: str, count: int = 1, timeout: float = 10.0) -> list[dict]:
    """Poll get_watch_events until `count` events arrived (or fail after `timeout` seconds)."""
    with anyio.fail_after(timeout):
        while True:
            _, data = await call(client, "get_watch_events", watch_id=watch_id)
            if len(data["events"]) >= count:
                return data["events"]
            await anyio.sleep(0.1)
