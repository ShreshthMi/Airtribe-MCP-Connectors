"""
mcp_client.py: a synchronous MCP client, so the (synchronous) LangGraph agent can use MCP servers.

The agent's graph, its Toolbox and the chat loop are synchronous, and Round 2 calls the
corpus from several threads. MCP clients are async. This class owns an event loop in a
background thread (an anyio blocking portal), keeps one ClientSession open on it, and
exposes plain blocking methods that are safe to call from any thread:

  list_tools()      the server's tools, as Anthropic-style tool schemas
  call_tool()       (is_error, data): structuredContent, else the JSON or plain text result
  list_resources()  read_resource()
  notifications     what the server pushed (list_changed, log messages), newest last
  close()

The server runs as a child process over stdio. It gets only a minimal environment (the
SDK's default: PATH, HOME and the like), so it never sees ANTHROPIC_API_KEY. Its stderr
log goes to logs/<name>.log. If the server dies, the next call restarts it once; if that
fails too, MCPClientError says so.
"""
from __future__ import annotations

import atexit
import itertools
import json
import sys
import threading
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Callable

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError

PROJECT_ROOT = Path(__file__).resolve().parent
FILESYSTEM_SERVER = PROJECT_ROOT / "filesystem_mcp_server.py"
LOG_DIR = PROJECT_ROOT / "logs"
REQUEST_TIMEOUT = timedelta(seconds=120)
_DEAD = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream, BrokenPipeError, ConnectionError)


class MCPClientError(Exception):
    """The MCP server could not be started, or stopped answering and could not be restarted."""


class MCPClient:
    def __init__(self, params: StdioServerParameters, name: str = "filesystem",
                 on_call: Callable[[str, str, dict], None] | None = None, log_file: Path | None = None):
        self.params, self.name, self.on_call = params, name, on_call
        self.log_file = Path(log_file) if log_file else LOG_DIR / f"{name}_mcp_server.log"
        self.calls: Counter = Counter()  # "tools/call read_file", "resources/read", ...
        self.notifications: deque[dict] = deque(maxlen=200)
        self.server_info: dict = {}
        self._lock = threading.Lock()
        self._restarts = itertools.count()
        self._portal_cm = self._portal = self._session_cm = self._session = self._errlog = None
        self._tools_cache: list[dict] | None = None
        self._start()
        atexit.register(self.close)

    @classmethod
    def filesystem(cls, root: Path | str = PROJECT_ROOT, *server_args: str, **kwargs) -> "MCPClient":
        """Spawn filesystem_mcp_server.py confined to `root` and connect to it."""
        params = StdioServerParameters(command=sys.executable,
                                       args=[str(FILESYSTEM_SERVER), "--root", str(Path(root).resolve()),
                                             *server_args],
                                       cwd=str(PROJECT_ROOT))
        return cls(params, name="filesystem", **kwargs)

    # ---- connection ----------------------------------------------------------- #
    @asynccontextmanager
    async def _connect(self):
        async with stdio_client(self.params, errlog=self._errlog) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=REQUEST_TIMEOUT,
                                     message_handler=self._on_message, logging_callback=self._on_log) as session:
                init = await session.initialize()
                yield session, init

    def _start(self) -> None:
        try:
            self.log_file.parent.mkdir(exist_ok=True)
            self._errlog = open(self.log_file, "a", encoding="utf-8")  # noqa: SIM115 - closed in close()
            self._portal_cm = start_blocking_portal(backend="asyncio")
            self._portal = self._portal_cm.__enter__()
            self._session_cm = self._portal.wrap_async_context_manager(self._connect())
            self._session, init = self._session_cm.__enter__()
        except Exception as exc:
            self._shutdown()
            raise MCPClientError(f"Could not start the {self.name} MCP server ({type(exc).__name__}: {exc}); "
                                 f"see {self.log_file}") from exc
        self.server_info = {"name": init.serverInfo.name, "version": init.serverInfo.version,
                            "protocol": init.protocolVersion}

    def _shutdown(self) -> None:
        for cm in (self._session_cm, self._portal_cm):
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:  # noqa: BLE001 - the process may already be gone
                    pass
        self._session_cm = self._portal_cm = self._session = self._portal = None
        if self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    def close(self) -> None:
        with self._lock:
            self._shutdown()

    @property
    def closed(self) -> bool:
        return self._session is None

    def _request(self, label: str, fn: Callable, *args):
        """Run session method `fn(*args)` on the portal; restart the server once if it died."""
        self.calls[label] += 1
        for attempt in range(2):
            if self._session is None:
                raise MCPClientError(f"The {self.name} MCP client is closed.")
            try:
                return self._portal.call(fn(self._session), *args)
            except _DEAD + (McpError,) as exc:
                if isinstance(exc, McpError) and "connection closed" not in str(exc).lower():
                    raise  # a protocol error from a live server: the caller handles it
                if attempt or next(self._restarts) >= 1:
                    raise MCPClientError(f"The {self.name} MCP server stopped answering ({exc}); see {self.log_file}") \
                        from exc
                with self._lock:
                    self._shutdown()
                    self._start()

    # ---- notifications ---------------------------------------------------------- #
    async def _on_message(self, message) -> None:
        if isinstance(message, Exception):
            self.notifications.append({"type": "error", "data": str(message)})
            return
        root = getattr(message, "root", None)
        if root is not None and type(root).__name__.endswith("Notification"):
            self.notifications.append({"type": getattr(root, "method", type(root).__name__), "data": None})

    async def _on_log(self, params) -> None:
        self.notifications.append({"type": "notifications/message", "logger": params.logger, "data": params.data})

    # ---- the MCP methods --------------------------------------------------------------- #
    def list_tools(self, refresh: bool = False) -> list[dict]:
        """The server's tools as Anthropic tool schemas: MCP's inputSchema is JSON Schema, as is
        Anthropic's input_schema, so only the key changes."""
        if self._tools_cache is None or refresh:
            result = self._request("tools/list", lambda s: s.list_tools)
            self._tools_cache = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                                 for t in result.tools]
        return self._tools_cache

    def call_tool(self, name: str, arguments: dict) -> tuple[bool, dict]:
        if self.on_call:
            self.on_call("tools/call", name, arguments)
        result = self._request(f"tools/call {name}", lambda s: s.call_tool, name, arguments)
        if result.structuredContent is not None:
            return result.isError, result.structuredContent
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        return result.isError, data if isinstance(data, dict) else {"text": text}

    def list_resources(self) -> list[dict]:
        result = self._request("resources/list", lambda s: s.list_resources)
        return [{"uri": str(r.uri), "name": r.name, "mimeType": r.mimeType} for r in result.resources]

    def list_resource_templates(self) -> list[str]:
        result = self._request("resources/templates/list", lambda s: s.list_resource_templates)
        return [t.uriTemplate for t in result.resourceTemplates]

    def read_resource(self, uri: str) -> str:
        if self.on_call:
            self.on_call("resources/read", uri, {})
        result = self._request("resources/read", lambda s: s.read_resource, uri)
        return "\n".join(getattr(c, "text", "") for c in result.contents)

    def drain_notifications(self) -> list[dict]:
        items = list(self.notifications)
        self.notifications.clear()
        return items

