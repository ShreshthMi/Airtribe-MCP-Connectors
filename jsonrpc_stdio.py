"""
A JSON-RPC 2.0 checking stdio transport for the MCP server.

The SDK's own stdio transport (mcp 1.30.0) hands any line it cannot parse to the session
as an exception. The session answers that with an error *log notification*, not a JSON-RPC
error response, and it answers an unknown method with -32602 instead of -32601. This
transport keeps the SDK for everything else (sessions, handlers, framing of good messages)
and checks each incoming line first, answering protocol errors the way JSON-RPC 2.0 says:

  -32700  Parse error       the line is not JSON                     id: null
  -32600  Invalid Request   a batch array (MCP removed batching in
                            spec revision 2025-06-18), not an object,
                            "jsonrpc" is not "2.0", or a bad id       id: the request's, if usable
  -32601  Method not found  a request for a method MCP does not define
  -32602  Invalid params    (from the SDK) params that fail its schema
  -32603  Internal error    (rewritten from the SDK's code 0) a handler crashed

Notifications (no id) never get a response, as JSON-RPC requires; an unknown one is dropped.
"""
from __future__ import annotations

import json
import sys
import typing
from contextlib import asynccontextmanager
from io import TextIOWrapper

import anyio
from mcp import types
from mcp.shared.message import SessionMessage

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INTERNAL_ERROR = -32700, -32600, -32601, -32603


def _methods(union) -> frozenset[str]:
    return frozenset(typing.get_args(cls.model_fields["method"].annotation)[0]
                     for cls in typing.get_args(union.model_fields["root"].annotation))


REQUEST_METHODS = _methods(types.ClientRequest)
NOTIFICATION_METHODS = _methods(types.ClientNotification)


def _error(request_id, code: int, message: str, data=None) -> dict:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def check_line(line: str) -> types.JSONRPCMessage | dict | None:
    """The parsed message, an error response to send back (a dict), or None to drop the line."""
    try:
        obj = json.loads(line)
    except ValueError as exc:
        return _error(None, PARSE_ERROR, "Parse error", str(exc))
    if isinstance(obj, list):
        return _error(None, INVALID_REQUEST, "Invalid Request: JSON-RPC batches are not supported by MCP "
                                             "(removed in protocol revision 2025-06-18); send one message per line")
    if not isinstance(obj, dict):
        return _error(None, INVALID_REQUEST, "Invalid Request: a message must be a JSON object")
    raw_id = obj.get("id")
    usable_id = raw_id if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else None
    if obj.get("jsonrpc") != "2.0":
        return _error(usable_id, INVALID_REQUEST, 'Invalid Request: "jsonrpc" must be exactly "2.0"')
    if "method" in obj:
        if not isinstance(obj["method"], str):
            return _error(usable_id, INVALID_REQUEST, 'Invalid Request: "method" must be a string')
        if "id" not in obj:  # a notification: never answered
            return None if obj["method"] not in NOTIFICATION_METHODS else _parse(obj, None)
        if usable_id is None:
            return _error(None, INVALID_REQUEST, 'Invalid Request: "id" must be a string or an integer')
        if obj["method"] not in REQUEST_METHODS:
            return _error(usable_id, METHOD_NOT_FOUND, f"Method not found: {obj['method']}")
    return _parse(obj, usable_id)


def _parse(obj: dict, request_id) -> types.JSONRPCMessage | dict | None:
    try:
        return types.JSONRPCMessage.model_validate(obj)
    except ValueError as exc:
        return None if request_id is None and "method" in obj else \
            _error(request_id, INVALID_REQUEST, "Invalid Request", str(exc).splitlines()[0])


@asynccontextmanager
async def jsonrpc_stdio_server(stdin=None, stdout=None):
    """Like mcp.server.stdio.stdio_server, plus the checks above."""
    stdin = stdin or anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace"))
    stdout = stdout or anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding="utf-8"))
    read_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_reader = anyio.create_memory_object_stream(0)
    lock = anyio.Lock()

    async def emit(text: str) -> None:
        async with lock:
            await stdout.write(text + "\n")
            await stdout.flush()

    async def stdin_reader() -> None:
        try:
            async with read_writer:
                async for line in stdin:
                    if not line.strip():
                        continue
                    checked = check_line(line)
                    if isinstance(checked, dict):
                        await emit(json.dumps(checked))
                    elif checked is not None:
                        await read_writer.send(SessionMessage(checked))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdout_writer() -> None:
        try:
            async with write_reader:
                async for session_message in write_reader:
                    root = session_message.message.root
                    if isinstance(root, types.JSONRPCError) and root.error.code == 0:
                        root.error.code = INTERNAL_ERROR  # the SDK's catch-all; JSON-RPC calls it -32603
                    await emit(session_message.message.model_dump_json(by_alias=True, exclude_none=True))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async with anyio.create_task_group() as group:
        group.start_soon(stdin_reader)
        group.start_soon(stdout_writer)
        yield read_stream, write_stream


async def serve(fastmcp) -> None:
    """Run a FastMCP server over the checking transport (what FastMCP.run('stdio') does otherwise)."""
    async with jsonrpc_stdio_server() as (read_stream, write_stream):
        await fastmcp._mcp_server.run(read_stream, write_stream, fastmcp._mcp_server.create_initialization_options())
