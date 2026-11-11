"""
mcp_hub.py: several MCP servers behind one tool list (the multi-MCP bonus).

The servers are listed in mcp_servers.json, in the same shape Claude Desktop uses:

  {"mcpServers": {"fetch": {"command": "uvx", "args": ["mcp-server-fetch"], "env": {...}}}}

Each one is started as a child process over stdio with its own MCPClient. Tool names are
namespaced as <server>__<tool> (fetch__fetch, filesystem__read_file), so two servers can
both have a tool called "read" without colliding, and a call is routed by its prefix.

In command, args and env values, ${NAME} is replaced by that environment variable (empty
if unset; an env entry that ends up empty is left out) and ${PYTHON} by this interpreter.
A server gets only the SDK's minimal environment plus the env entries listed for it, so
no server sees ANTHROPIC_API_KEY unless the file names it.

  python mcp_hub.py                  # start every server and list its namespaced tools
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

from mcp import StdioServerParameters
from mcp.client.stdio import get_default_environment

from mcp_client import PROJECT_ROOT, MCPClient

CONFIG_FILE = PROJECT_ROOT / "mcp_servers.json"
SEPARATOR = "__"
_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class HubError(Exception):
    """mcp_servers.json is missing or malformed, or a tool name has no server."""


def expand(text: str) -> str:
    return _VARIABLE.sub(lambda m: sys.executable if m.group(1) == "PYTHON" else os.environ.get(m.group(1), ""), text)


def load_servers(path: Path = CONFIG_FILE) -> dict[str, dict]:
    try:
        servers = json.loads(Path(path).read_text())["mcpServers"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HubError(f"{path} is missing or not in the mcpServers format ({exc})") from exc
    for name, spec in servers.items():
        if SEPARATOR in name or not isinstance(spec, dict) or not spec.get("command"):
            raise HubError(f"server {name!r}: needs a name without '{SEPARATOR}' and a command")
    return servers


def server_parameters(spec: dict) -> StdioServerParameters:
    extra = {key: expand(str(value)) for key, value in (spec.get("env") or {}).items()}
    return StdioServerParameters(command=expand(spec["command"]), args=[expand(str(a)) for a in spec.get("args", [])],
                                 env={**get_default_environment(), **{k: v for k, v in extra.items() if v}},
                                 cwd=str(PROJECT_ROOT))


class MCPHub:
    def __init__(self, path: Path = CONFIG_FILE, exclude: tuple[str, ...] = (), only: tuple[str, ...] | None = None,
                 on_call: Callable[[str, str, dict], None] | None = None):
        self.clients: dict[str, MCPClient] = {}
        try:
            for name, spec in load_servers(path).items():
                if name in exclude or (only is not None and name not in only):
                    continue
                self.clients[name] = MCPClient(server_parameters(spec), name=name, on_call=on_call)
        except Exception:
            self.close()
            raise

    def tools(self) -> list[dict]:
        """Every server's tools as Anthropic tool schemas, named <server>__<tool>."""
        return [{"name": f"{server}{SEPARATOR}{tool['name']}",
                 "description": f"[{server} MCP server] {tool['description']}", "input_schema": tool["input_schema"]}
                for server, client in self.clients.items() for tool in client.list_tools()]

    def owns(self, qualified: str) -> bool:
        return qualified.split(SEPARATOR, 1)[0] in self.clients and SEPARATOR in qualified

    def call(self, qualified: str, arguments: dict) -> tuple[bool, dict]:
        server, _, tool = qualified.partition(SEPARATOR)
        if server not in self.clients or not tool:
            raise HubError(f"no MCP server for tool {qualified!r}; servers: {', '.join(self.clients)}")
        return self.clients[server].call_tool(tool, arguments)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()


def main() -> None:
    hub = MCPHub()
    try:
        for name, client in hub.clients.items():
            print(f"{name}: {client.server_info}")
        for tool in hub.tools():
            print(f"  {tool['name']}")
    finally:
        hub.close()


if __name__ == "__main__":
    main()
