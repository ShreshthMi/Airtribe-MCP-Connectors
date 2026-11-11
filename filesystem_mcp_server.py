"""
filesystem_mcp_server.py: the Milestone 1 file tools as a Model Context Protocol server.

Run it:
  python filesystem_mcp_server.py                         # stdio (what the agent spawns)
  python filesystem_mcp_server.py --transport streamable-http --port 8765
  mcp dev filesystem_mcp_server.py                        # MCP Inspector (needs Node.js)

What it serves (protocol revision 2025-11-25, JSON-RPC 2.0 framing, via the official
`mcp` Python SDK, pinned at 1.30.0):

Tools (model-invoked actions)
  list_files, read_file, search_in_file, write_file     Milestone 1, unchanged underneath
  watch_directory, get_watch_events, unwatch_directory  new: tell clients about new resumes
  batch_process                                         new: one operation over many files

Resources (application-read data)
  resume://{filename}, jd://{filename}, report://{filename}   templates over the folders
  one concrete resource per file, re-announced (list_changed) when a watch sees a new one
  config://server   the effective configuration      stats://server   call counters

Milestone 1's fs_tools is imported from the vendored, hash-checked copy in
milestone2_rag/milestone1_tools; nothing of it is reimplemented here. This module adds
the folder policy the agent used to enforce itself (read resumes/, job_descriptions/,
reports/; write reports/ only, never replacing a file), machine-readable error codes,
configuration, and the two MCP-specific tools.

stdout is the protocol channel in stdio mode, so every log line goes to stderr.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.resources import FunctionResource
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, CallToolResult, ErrorData, TextContent, ToolAnnotations

import anyio

import jsonrpc_stdio
import mcp_batch
from mcp_watch import WatchEvent, WatchManager
from server_config import ConfigError, ServerConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parent
M1_TOOLS_DIR = PROJECT_ROOT / "milestone2_rag" / "milestone1_tools"
SERVER_NAME = "airtribe-filesystem"
SCHEMES = {"resumes": "resume", "job_descriptions": "jd", "reports": "report"}
MIME_TYPES = {".md": "text/markdown", ".json": "application/json", ".csv": "text/csv"}  # others: extracted text

# The stable error codes a tool result carries (isError: true), and what each means.
ERROR_CODES = {
    "NOT_FOUND": "the file or folder does not exist",
    "ACCESS_DENIED": "outside the sandbox, hidden, or outside the folders this operation may use",
    "UNSUPPORTED_FORMAT": "a file type the tools cannot read or write",
    "TOO_LARGE": "over the configured size limit",
    "ALREADY_EXISTS": "write_file never replaces an existing file",
    "INVALID_ARGUMENT": "an argument has the right type but an unusable value",
    "READ_FAILED": "the file exists but could not be read or parsed (e.g. a corrupt PDF)",
}

# JSON-RPC error codes for resources/read. -32002 is the MCP spec's "resource not found";
# the SDK on its own answers every resource failure with code 0, so FilesystemMCP maps them.
RESOURCE_NOT_FOUND = -32002
RESOURCE_ERROR_CODES = {"NOT_FOUND": RESOURCE_NOT_FOUND, "ACCESS_DENIED": INVALID_PARAMS,
                        "UNSUPPORTED_FORMAT": INVALID_PARAMS, "TOO_LARGE": INVALID_PARAMS,
                        "INVALID_ARGUMENT": INVALID_PARAMS, "READ_FAILED": INTERNAL_ERROR}

log = logging.getLogger("mcp_fs")
sys.path.insert(0, str(M1_TOOLS_DIR))
import fs_tools  # noqa: E402  (Milestone 1, vendored verbatim)


def classify(message: str) -> str:
    """Map a Milestone 1 error message to a stable error code."""
    m = message.lower()
    if "keyword is too long" in m or "keyword is empty" in m or "invalid extension filter" in m:
        return "INVALID_ARGUMENT"
    if "not found" in m or "doesn't exist" in m:
        return "NOT_FOUND"
    if m.startswith("access denied") or "permission denied" in m or "read-only" in m:
        return "ACCESS_DENIED"
    if "aren't supported" in m or "isn't supported" in m or "refusing to write" in m:
        return "UNSUPPORTED_FORMAT"
    if "too large" in m:
        return "TOO_LARGE"
    if "already exists" in m:
        return "ALREADY_EXISTS"
    if "must be" in m or "is empty" in m or "null byte" in m or "is a folder" in m or "not a folder" in m:
        return "INVALID_ARGUMENT"
    return "READ_FAILED"


class FileService:
    """Folder policy plus the Milestone 1 calls. Synchronous; batch_process runs it in threads.

    The sandbox root is set once, here, from the configuration: Milestone 1 reads
    $FS_ASSISTANT_ROOT on every call, and with one fixed root per process no lock is needed.
    """

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.root = cfg.root
        os.environ[fs_tools.ROOT_ENV_VAR] = str(self.root)
        self.calls: Counter = Counter()
        self.max_bytes = int(cfg.max_file_mb * 1024 * 1024)

    # ---- results --------------------------------------------------------------- #
    def error(self, code: str, message: str, filepath: Any = None) -> dict:
        message = str(message).replace(str(self.root), "<root>")  # never leak a host path
        return {"success": False, "error_code": code, "message": message,
                "filepath": None if filepath is None else str(filepath)}

    def _from_m1(self, result: dict, filepath: str) -> dict:
        if result.get("success"):
            return result
        message = result.get("error") or "unknown error"
        return self.error(classify(message), message, filepath)

    # ---- policy ---------------------------------------------------------------- #
    def refuse(self, filepath: Any, folders: tuple[str, ...], *, is_dir: bool = False) -> dict | None:
        """An error unless `filepath` is a relative path inside one of `folders`.

        Milestone 1 still applies its own sandbox afterwards ('..', symlinks, hidden files,
        anything resolving outside the root); this keeps clients to the data folders.
        """
        if not isinstance(filepath, str) or not filepath.strip():
            return self.error("INVALID_ARGUMENT", "filepath must be a non-empty string", filepath)
        text = filepath.strip().replace("\\", "/")
        parts = PurePosixPath(text).parts
        allowed = ", ".join(f"{f}/" for f in folders)
        if text.startswith(("/", "~")) or ".." in parts or not parts or parts[0] not in folders \
                or (len(parts) < 2 and not is_dir):
            return self.error("ACCESS_DENIED", f"Access denied: '{filepath}'. Access is limited to {allowed}", filepath)
        if any(p.startswith(".") for p in parts):
            return self.error("ACCESS_DENIED", f"Access denied: '{filepath}' is hidden", filepath)
        return None

    def _check_extension(self, filepath: str) -> dict | None:
        ext = PurePosixPath(filepath).suffix.lower()
        if ext not in self.cfg.allowed_extensions:
            return self.error("UNSUPPORTED_FORMAT", f"'{filepath}': {ext or 'no extension'} is not one of "
                              f"{', '.join(self.cfg.allowed_extensions)}", filepath)
        return None

    def _check_size(self, filepath: str) -> dict | None:
        try:
            size = os.stat(self.root / filepath).st_size
        except OSError:
            return None  # Milestone 1 reports the missing file in its own words
        if size > self.max_bytes:
            return self.error("TOO_LARGE", f"'{filepath}' is {size / 1048576:.2f} MB; the limit is "
                              f"{self.cfg.max_file_mb:g} MB", filepath)
        return None

    # ---- the four Milestone 1 operations ------------------------------------------ #
    def read_file(self, filepath: str, *, count: bool = True) -> dict:
        if count:
            self.calls["read_file"] += 1
        bad = (self.refuse(filepath, self.cfg.readable_folders) or self._check_extension(filepath)
               or self._check_size(filepath))
        return bad or self._from_m1(fs_tools.read_file(filepath), filepath)

    def list_files(self, directory: str, extension: str | None = None, *, count: bool = True) -> dict:
        if count:
            self.calls["list_files"] += 1
        bad = self.refuse(directory, self.cfg.readable_folders, is_dir=True)
        if bad:
            return bad
        entries = fs_tools.list_files(directory.strip().rstrip("/"), extension or None)
        if entries and "error" in entries[0]:
            return self.error(classify(entries[0]["error"]), entries[0]["error"], directory)
        return {"success": True, "directory": directory, "count": len(entries), "files": entries}

    def search_in_file(self, filepath: str, keyword: str, whole_word: bool = False, *, count: bool = True) -> dict:
        if count:
            self.calls["search_in_file"] += 1
        bad = (self.refuse(filepath, self.cfg.readable_folders) or self._check_extension(filepath)
               or self._check_size(filepath))
        return bad or self._from_m1(fs_tools.search_in_file(filepath, keyword, whole_word), filepath)

    def write_file(self, filepath: str, content: str) -> dict:
        self.calls["write_file"] += 1
        bad = self.refuse(filepath, self.cfg.writable_folders) or self._check_extension(filepath)
        if bad:
            return bad
        if os.path.lexists(self.root / filepath):
            return self.error("ALREADY_EXISTS", f"'{filepath}' already exists; choose a new name", filepath)
        (self.root / PurePosixPath(filepath).parts[0]).mkdir(exist_ok=True)
        return self._from_m1(fs_tools.write_file(filepath, content), filepath)

    def stat(self, filepath: str) -> dict:
        bad = self.refuse(filepath, self.cfg.readable_folders)
        if bad:
            return bad
        target = self.root / filepath
        real = target.resolve()
        if not real.is_relative_to(self.root) or not target.is_file():
            return self.error("NOT_FOUND", f"File not found: '{filepath}'", filepath)
        info = target.stat()
        return {"success": True, "size_bytes": info.st_size, "extension": target.suffix.lower(),
                "modified": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(timespec="seconds")}

    # ---- resources ------------------------------------------------------------- #
    def resource_text(self, folder: str, filename: str) -> str:
        self.calls["resources/read"] += 1
        if "/" in filename or "\\" in filename:
            raise ResourceError(f"ACCESS_DENIED: '{filename}' must be a file name inside {folder}/")
        result = self.read_file(f"{folder}/{filename}", count=False)  # counted as a resource read
        if not result.get("success"):
            raise ResourceError(f"{result['error_code']}: {result['message']}")
        return result["content"]

    def data_files(self) -> list[tuple[str, str]]:
        """(folder, filename) for every visible, allowed file in the readable folders."""
        found = []
        for folder in self.cfg.readable_folders:
            path = self.root / folder
            if path.is_dir():
                found += [(folder, p.name) for p in sorted(path.iterdir())
                          if p.is_file() and not p.name.startswith((".", "~$"))
                          and p.suffix.lower() in self.cfg.allowed_extensions]
        return found


class FilesystemMCP(FastMCP):
    """FastMCP, with resources/read failures sent as proper JSON-RPC error codes."""

    async def read_resource(self, uri):
        try:
            return await super().read_resource(uri)
        except (ResourceError, ValueError) as exc:  # the resource manager raises ValueError for unknown URIs
            message = str(exc).replace("Error creating resource from template: ", "")
            code = "NOT_FOUND" if message.startswith("Unknown resource") else \
                next((c for c in ERROR_CODES if message.startswith(f"{c}: ")), "READ_FAILED")
            message = message.removeprefix(f"{code}: ")
            raise McpError(ErrorData(code=RESOURCE_ERROR_CODES[code], message=message,
                                     data={"uri": str(uri), "error_code": code})) from exc


def _result(data: dict) -> CallToolResult:
    """A tool result: JSON text for any client, the same object as structuredContent, and
    isError set when the operation failed (so clients and models can tell without parsing)."""
    failed = data.get("success") is False
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, ensure_ascii=False))],
                          structuredContent=data, isError=failed)


def build_server(cfg: ServerConfig | None = None) -> FastMCP:
    cfg = cfg or load_config([])
    service = FileService(cfg)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    mcp: FastMCP  # assigned below; the watcher callback needs it

    def add_file_resource(folder: str, filename: str) -> str:
        scheme = SCHEMES.get(folder, folder)
        uri = f"{scheme}://{filename}"
        mcp.add_resource(FunctionResource.from_function(
            lambda: service.resource_text(folder, filename), uri=uri, name=f"{folder}/{filename}",
            description=f"{folder}/{filename} as text (PDF and DOCX are extracted by Milestone 1)",
            mime_type=MIME_TYPES.get(Path(filename).suffix.lower(), "text/plain")))
        return uri

    async def announce(event: WatchEvent, session) -> None:
        folder, filename = event.path.split("/", 1)
        uri = add_file_resource(folder, filename) if folder in SCHEMES else None
        if session is not None:
            await session.send_resource_list_changed()
            await session.send_log_message("info", {"event": "new_file", "path": event.path, "seq": event.seq,
                                                    "watch_id": event.watch_id, "resource": uri},
                                           logger="watch_directory")

    watches = WatchManager(cfg.root, cfg.allowed_extensions, cfg.watch_debounce_ms, cfg.watch_force_polling,
                           cfg.watch_buffer_size, announce)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield {"service": service, "watches": watches}
        finally:
            await watches.stop_all()

    mcp = FilesystemMCP(
        SERVER_NAME, lifespan=lifespan, host=cfg.host, port=cfg.port, log_level=cfg.log_level,
        instructions="Read-only access to resumes/, job_descriptions/ and reports/ through Milestone 1's "
                     "sandboxed file tools; new files may be written to reports/ only and never replace one. "
                     "Failed calls return isError with {error_code, message, filepath}.")
    read_only = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    def traced(name: str, args: dict, data: dict) -> CallToolResult:
        status = "ok" if data.get("success") is not False else data.get("error_code")
        log.info("tools/call %s %s -> %s", name, json.dumps(args, ensure_ascii=False)[:120], status)
        return _result(data)

    # ---- Milestone 1 tools ---------------------------------------------------------- #
    @mcp.tool(annotations=read_only)
    def list_files(directory: str, extension: str | None = None) -> CallToolResult:
        """List the files in a folder: resumes, job_descriptions or reports.

        Args:
            directory: The folder, e.g. "resumes".
            extension: Optional filter such as ".pdf"; omit or leave empty for all files.
        Returns {directory, count, files: [{name, path, extension, size_bytes, modified}]}.
        """
        return traced("list_files", {"directory": directory, "extension": extension},
                      service.list_files(directory, extension))

    @mcp.tool(annotations=read_only)
    def read_file(filepath: str) -> CallToolResult:
        """Read a .txt, .md, .pdf or .docx file under resumes/, job_descriptions/ or reports/ and return
        its text (PDF and DOCX are extracted). Returns {filepath, content, metadata, warning}."""
        return traced("read_file", {"filepath": filepath}, service.read_file(filepath))

    @mcp.tool(annotations=read_only)
    def search_in_file(filepath: str, keyword: str, whole_word: bool = False) -> CallToolResult:
        """Find a keyword (case-insensitive) in one file and return each match with its line and context.

        Args:
            filepath: e.g. "resumes/grace_liu.pdf".
            keyword: The word or phrase to look for.
            whole_word: Match whole words only (so "Java" does not match "JavaScript").
        """
        return traced("search_in_file", {"filepath": filepath, "keyword": keyword, "whole_word": whole_word},
                      service.search_in_file(filepath, keyword, whole_word))

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                          openWorldHint=False))
    def write_file(filepath: str, content: str) -> CallToolResult:
        """Save a new text or Markdown file under reports/. An existing file is never replaced
        (error ALREADY_EXISTS): pick a new name instead."""
        return traced("write_file", {"filepath": filepath, "content": f"<{len(content)} chars>"},
                      service.write_file(filepath, content))

    # ---- watch_directory ------------------------------------------------------------- #
    @mcp.tool()
    async def watch_directory(ctx: Context, directory: str = "resumes",
                              extensions: list[str] | None = None) -> CallToolResult:
        """Start watching a folder for new files (by default new resumes in resumes/).

        Files already there are not reported. Each new file, once fully written, is added as a
        resource; clients get notifications/resources/list_changed plus a log message, and the
        event is buffered for get_watch_events. Returns {watch_id, directory, ...}.
        """
        bad = service.refuse(directory, cfg.readable_folders, is_dir=True)
        if bad is None and "/" in directory.strip().strip("/"):
            bad = service.error("INVALID_ARGUMENT", "watch a top-level folder such as 'resumes'", directory)
        if bad is None and not (cfg.root / directory.strip().strip("/")).is_dir():
            bad = service.error("NOT_FOUND", f"Folder not found: '{directory}'", directory)
        if bad:
            return traced("watch_directory", {"directory": directory}, bad)
        service.calls["watch_directory"] += 1
        info = watches.start(directory.strip().strip("/"), tuple(extensions) if extensions else None, ctx.session)
        return traced("watch_directory", {"directory": directory}, {"success": True, **info})

    @mcp.tool(annotations=read_only)
    def get_watch_events(watch_id: str, since: int = 0) -> CallToolResult:
        """New files seen by a watch, oldest first, with sequence numbers above `since`.
        Pass the highest seq you have seen to get only newer events."""
        service.calls["get_watch_events"] += 1
        if not watches.exists(watch_id):
            data = service.error("NOT_FOUND", f"No active watch '{watch_id}'", None)
        else:
            events = watches.events_since(watch_id, since)
            data = {"success": True, "watch_id": watch_id, "events": events,
                    "last_seq": max([e["seq"] for e in events], default=since)}
        return traced("get_watch_events", {"watch_id": watch_id, "since": since}, data)

    @mcp.tool()
    async def unwatch_directory(watch_id: str) -> CallToolResult:
        """Stop a watch started by watch_directory."""
        service.calls["unwatch_directory"] += 1
        stopped = await watches.stop(watch_id)
        data = {"success": True, "watch_id": watch_id, "stopped": True} if stopped else \
            service.error("NOT_FOUND", f"No active watch '{watch_id}'", None)
        return traced("unwatch_directory", {"watch_id": watch_id}, data)

    # ---- batch_process ----------------------------------------------------------------- #
    @mcp.tool()
    async def batch_process(ctx: Context, paths: list[str] | None = None, directory: str | None = None,
                            operation: str = "read", keyword: str | None = None) -> CallToolResult:
        """Run one operation over many files concurrently, with per-file results and progress.

        Args:
            paths: Files to process, e.g. ["resumes/a.pdf", "resumes/b.md"]. Or use directory.
            directory: Process every allowed file in this folder (e.g. "resumes") instead of paths.
            operation: "read" (text length and a short preview), "search" (keyword match counts)
                or "stats" (size and modified time only).
            keyword: Required for operation "search".
        One failing file never fails the batch. Returns {total, ok, failed, elapsed_ms, results}.
        """
        args = {"paths": f"<{len(paths)} paths>" if paths else None, "directory": directory,
                "operation": operation, "keyword": keyword}
        service.calls["batch_process"] += 1
        if operation not in mcp_batch.OPERATIONS:
            return traced("batch_process", args, service.error(
                "INVALID_ARGUMENT", f"operation must be one of {', '.join(mcp_batch.OPERATIONS)}"))
        if (paths is None) == (directory is None):
            return traced("batch_process", args, service.error("INVALID_ARGUMENT", "give either paths or directory"))
        if operation == "search" and not (keyword or "").strip():
            return traced("batch_process", args, service.error("INVALID_ARGUMENT", "operation 'search' needs a keyword"))
        if directory is not None:
            listing = service.list_files(directory, count=False)
            if not listing["success"]:
                return traced("batch_process", args, listing)
            paths = [f["path"] for f in listing["files"] if Path(f["path"]).suffix.lower() in cfg.allowed_extensions]
        if len(paths) > cfg.batch_max_files:
            return traced("batch_process", args, service.error(
                "TOO_LARGE", f"{len(paths)} files; batch_max_files is {cfg.batch_max_files}"))

        preview = cfg.batch_preview_chars

        def worker(path: str) -> dict:
            if operation == "stats":
                r = service.stat(path)
                return {"ok": True, **{k: r[k] for k in ("size_bytes", "extension", "modified")}} if r["success"] \
                    else {"ok": False, "error_code": r["error_code"], "message": r["message"]}
            r = service.read_file(path, count=False) if operation == "read" else \
                service.search_in_file(path, keyword, count=False)  # counted once, as batch_process
            if not r.get("success"):
                return {"ok": False, "error_code": r["error_code"], "message": r["message"]}
            if operation == "read":
                text = r["content"]
                return {"ok": True, "chars": len(text), "preview": " ".join(text[:preview].split())}
            return {"ok": True, "match_count": r.get("match_count", 0)}

        async def progress(done: int, total: int, path: str) -> None:
            await ctx.report_progress(done, total, f"{operation} {path}")

        summary = await mcp_batch.run_batch(paths, operation, worker, cfg.batch_concurrency, progress)
        return traced("batch_process", args, {"success": True, **summary})

    # ---- resources ------------------------------------------------------------------- #
    for folder, scheme in SCHEMES.items():
        if folder not in cfg.readable_folders:
            continue

        mcp.resource(f"{scheme}://{{filename}}", name=f"{folder} file",
                     description=f"Any file in {folder}/ by name, as text", mime_type="text/plain")(
            _template_reader(service, folder, f"read_{scheme}"))

    for folder, filename in service.data_files():
        if folder in SCHEMES:
            add_file_resource(folder, filename)

    @mcp.resource("config://server", name="server configuration", mime_type="application/json",
                  description="The effective configuration and which layer each value came from")
    def config_resource() -> str:
        service.calls["resources/read"] += 1
        return json.dumps({"server": SERVER_NAME, **cfg.public_dict()}, indent=2)

    @mcp.resource("stats://server", name="server statistics", mime_type="application/json",
                  description="How many times each tool and resource has been used since start-up")
    def stats_resource() -> str:
        return json.dumps({"started_at": started, "calls": dict(sorted(service.calls.items())),
                           "active_watches": watches.active, "error_codes": ERROR_CODES}, indent=2)

    mcp.file_service = service  # for tests and the in-process tour
    return mcp


def _template_reader(service: FileService, folder: str, name: str):
    """A resource-template function for one folder (FastMCP matches its parameters to the URI's)."""
    def read(filename: str) -> str:
        return service.resource_text(folder, filename)
    read.__name__ = name
    return read


def main(argv: list[str] | None = None) -> None:
    try:
        cfg = load_config(sys.argv[1:] if argv is None else argv)
    except ConfigError as exc:
        print(f"filesystem_mcp_server: invalid configuration: {exc}", file=sys.stderr)
        raise SystemExit(2)
    logging.basicConfig(stream=sys.stderr, level=cfg.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)  # one line per request otherwise
    log.info("starting %s on %s (root %s/, %d files)", SERVER_NAME, cfg.transport, cfg.root.name,
             len(FileService(cfg).data_files()))
    t0 = time.perf_counter()
    server = build_server(cfg)
    log.info("ready in %.0f ms", (time.perf_counter() - t0) * 1000)
    if cfg.transport == "stdio":
        anyio.run(jsonrpc_stdio.serve, server)  # the SDK's stdio transport plus strict JSON-RPC 2.0 checks
    else:
        server.run(transport=cfg.transport)


if __name__ == "__main__":
    main()
