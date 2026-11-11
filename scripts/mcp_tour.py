"""
A guided tour of the filesystem MCP server in raw JSON-RPC 2.0, over stdio.

No SDK on this side: the script starts filesystem_mcp_server.py, writes JSON-RPC lines to
its stdin and prints what comes back on its stdout, so what you see is the wire protocol
itself. Needs no API key. It produces scenarios 07-10 and the server segments of the video.

    python scripts/mcp_tour.py                        # every part
    python scripts/mcp_tour.py --part discovery       # 07: handshake, tools, resources, templates
    python scripts/mcp_tour.py --part watch           # 08: a new resume arrives while watched
    python scripts/mcp_tour.py --part batch           # 09: batch_process over all 100 resumes
    python scripts/mcp_tour.py --part errors          # 10: tool errors and protocol errors
    python scripts/mcp_tour.py --pause 1.5            # slower, for the video

Lines starting with "## " say what the next step shows (the video uses them as captions).
The watch part copies demo/incoming/zoe_adeyemi.md into resumes/ and removes it afterwards.
"""
from __future__ import annotations

import argparse
import json
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCOMING = ROOT / "demo" / "incoming" / "zoe_adeyemi.md"
WIDTH = 112
COLOUR = sys.stdout.isatty()
PARTS = ("discovery", "watch", "batch", "errors")


def paint(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if COLOUR else text


def short(obj, limit: int = 2 * WIDTH) -> str:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit - 5] + " ...}"


class Wire:
    """A raw JSON-RPC client over the server's stdin/stdout."""

    def __init__(self, pause: float):
        self.pause, self.next_id = pause, 1
        self.log = open(ROOT / "logs" / "mcp_tour_server.log", "w")  # noqa: SIM115
        self.process = subprocess.Popen([sys.executable, str(ROOT / "filesystem_mcp_server.py"),
                                         "--watch-debounce-ms", "200"], cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)

    def caption(self, text: str) -> None:
        print(f"\n## {text}", flush=True)
        time.sleep(self.pause)

    def send(self, message, *, show: bool = True) -> None:
        line = message if isinstance(message, str) else json.dumps(message)
        if show:
            print(paint("→ ", "36;1") + paint(short(line), "36"), flush=True)
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def receive(self, timeout: float = 30.0) -> dict | None:
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        return json.loads(self.process.stdout.readline()) if ready else None

    def request(self, method: str, params: dict | None = None, *, show_reply=True, raw_id=None) -> dict:
        request_id = raw_id if raw_id is not None else self.next_id
        self.next_id += 1
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, **({"params": params} if params else {})}
        self.send(message)
        while True:
            reply = self.receive()
            if reply is None or reply.get("id") == request_id or "method" not in reply:
                break
            self.show_notification(reply)
        if show_reply and reply is not None:
            self.show_reply(reply)
        time.sleep(self.pause / 2)
        return reply

    def show_reply(self, reply: dict, summary: str | None = None) -> None:
        if "error" in reply:
            print(paint("← ", "31;1") + paint(short(reply), "31"), flush=True)
        else:
            print(paint("← ", "32;1") + (summary or short(reply)), flush=True)

    def show_notification(self, note: dict) -> None:
        print(paint("⇠ ", "33;1") + paint(short(note), "33"), flush=True)

    def tool(self, name: str, arguments: dict, summary=None, meta: dict | None = None) -> dict:
        params = {"name": name, "arguments": arguments, **({"_meta": meta} if meta else {})}
        reply = self.request("tools/call", params, show_reply=summary is None)
        if summary is not None and reply is not None:
            self.show_reply(reply, summary(reply["result"]))
        return reply

    def close(self) -> None:
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


def handshake(wire: Wire) -> None:
    wire.caption("Handshake: the client offers protocol revision 2025-11-25; the server answers with its "
                 "name, version and capabilities.")
    wire.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                "clientInfo": {"name": "mcp_tour", "version": "1.0"}})
    wire.send({"jsonrpc": "2.0", "method": "notifications/initialized"})


def discovery(wire: Wire) -> None:
    wire.caption("tools/list: the four Milestone 1 tools plus watch_directory, get_watch_events, "
                 "unwatch_directory and batch_process, each with a JSON Schema for its arguments.")
    reply = wire.request("tools/list", show_reply=False)
    for tool in reply["result"]["tools"]:
        args = ", ".join(f"{k}{'' if k in tool['inputSchema'].get('required', []) else '?'}"
                         for k in tool["inputSchema"]["properties"])
        hint = " (read-only)" if (tool.get("annotations") or {}).get("readOnlyHint") else ""
        print(paint("← ", "32;1") + f"{tool['name']}({args}){hint}")
    time.sleep(wire.pause)

    wire.caption("resources/list: every resume, job description and report is also a resource, plus "
                 "config://server and stats://server.")
    resources = wire.request("resources/list", show_reply=False)["result"]["resources"]
    by_scheme: dict[str, int] = {}
    for r in resources:
        by_scheme[r["uri"].split("://")[0]] = by_scheme.get(r["uri"].split("://")[0], 0) + 1
    wire.show_reply({}, f"{len(resources)} resources: " + ", ".join(f"{n} {s}://" for s, n in by_scheme.items()))
    wire.show_reply({}, "e.g. " + ", ".join(f"{r['uri']} ({r['mimeType']})" for r in resources[:3]))
    templates = wire.request("resources/templates/list", show_reply=False)["result"]["resourceTemplates"]
    wire.show_reply({}, "templates: " + ", ".join(t["uriTemplate"] for t in templates))

    wire.caption("resources/read: a resume by URI (PDFs and DOCX come back as extracted text), and the "
                 "effective configuration with the layer each value came from.")
    resume = wire.request("resources/read", {"uri": "resume://clara_jensen.pdf"}, show_reply=False)
    content = resume["result"]["contents"][0]
    wire.show_reply({}, f"{content['uri']} {content['mimeType']}: " + short(" ".join(content["text"].split()), 150))
    config = json.loads(wire.request("resources/read", {"uri": "config://server"},
                                     show_reply=False)["result"]["contents"][0]["text"])
    wire.show_reply({}, short({k: config[k] for k in ("root", "readable_folders", "writable_folders", "max_file_mb",
                                                     "batch_concurrency", "transport")}))

    wire.caption("tools/call read_file: a Milestone 1 tool through MCP. The result carries JSON text and the "
                 "same object as structuredContent; isError is false.")
    wire.tool("read_file", {"filepath": "job_descriptions/backend_engineer.txt"},
              lambda r: f"isError={str(r['isError']).lower()} " + short(
                  {k: r["structuredContent"][k] for k in ("filepath", "content")}, 200))


def watch(wire: Wire) -> None:
    target = ROOT / "resumes" / INCOMING.name
    wire.caption("watch_directory starts watching resumes/. Files already there are not reported.")
    started = wire.tool("watch_directory", {"directory": "resumes"},
                        lambda r: short({k: r["structuredContent"][k] for k in ("watch_id", "directory")}))
    watch_id = started["result"]["structuredContent"]["watch_id"]
    time.sleep(1.0)
    try:
        wire.caption(f"A new resume lands in resumes/ ({INCOMING.name}). Once its size is stable, the server "
                     "pushes notifications/resources/list_changed and a log message.")
        print(paint(f"$ cp demo/incoming/{INCOMING.name} resumes/", "35"), flush=True)
        shutil.copy(INCOMING, target)
        deadline = time.monotonic() + 15
        seen = 0
        while seen < 2 and time.monotonic() < deadline:
            note = wire.receive(timeout=1.0)
            if note:
                wire.show_notification(note)
                seen += 1
        time.sleep(wire.pause)

        wire.caption("Clients that ignore notifications (or a model) can poll get_watch_events; the new file "
                     "is now a resource too.")
        wire.tool("get_watch_events", {"watch_id": watch_id}, lambda r: short(r["structuredContent"]["events"]))
        text = wire.request("resources/read", {"uri": f"resume://{INCOMING.name}"},
                            show_reply=False)["result"]["contents"][0]["text"]
        wire.show_reply({}, f"resume://{INCOMING.name}: " + short(" ".join(text.split()), 150))
        wire.tool("unwatch_directory", {"watch_id": watch_id})
    finally:
        target.unlink(missing_ok=True)
        print(paint(f"$ rm resumes/{INCOMING.name}   # tidy up", "35"), flush=True)


def batch(wire: Wire) -> None:
    wire.caption("batch_process reads all 100 resumes concurrently (2 at a time), sending "
                 "notifications/progress as each file finishes.")
    message = {"jsonrpc": "2.0", "id": 900, "method": "tools/call", "params": {
        "name": "batch_process", "arguments": {"directory": "resumes", "operation": "read"},
        "_meta": {"progressToken": "batch-1"}}}
    wire.send(message)
    progress = 0
    while True:
        reply = wire.receive()
        if reply is None or reply.get("id") == 900:
            break
        progress += 1
        params = reply.get("params", {})
        if progress in (1, 2) or params.get("progress") in (25, 50, 75, 100):
            wire.show_notification(reply)
        elif progress == 3:
            print(paint("⇠ ...", "33"), flush=True)
    summary = reply["result"]["structuredContent"]
    wire.show_reply({}, f"{progress} progress notifications; " + short(
        {k: summary[k] for k in ("total", "ok", "failed", "concurrency", "elapsed_ms")}))
    wire.show_reply({}, "results[0] = " + short(summary["results"][0], 200))
    time.sleep(wire.pause)

    wire.caption("One bad file never fails the batch: each file gets its own result and error code.")
    paths = ["resumes/ahmed_karim.md", "resumes/nobody.pdf", "../.env", "job_descriptions/devops_sre.txt"]
    reply = wire.tool("batch_process", {"paths": paths, "operation": "search", "keyword": "Kubernetes"},
                      lambda r: short({k: r["structuredContent"][k] for k in ("total", "ok", "failed")}))
    for item in reply["result"]["structuredContent"]["results"]:
        outcome = f"{item['match_count']} matches" if item["ok"] else item["error_code"]
        print(paint("   ", "0") + f"{item['path']:<40} {outcome}", flush=True)


def errors(wire: Wire) -> None:
    wire.caption("Tool errors are results with isError: true and a stable error_code. Here the model asks "
                 "for the project's .env; the server refuses, so the API key cannot leak through MCP.")
    for args in ({"filepath": "../.env"}, {"filepath": "/etc/passwd"}, {"filepath": "resumes/nobody.pdf"},
                 {"filepath": "resumes/photo.png"}):
        wire.tool("read_file", args, lambda r: f"isError={str(r['isError']).lower()} " + short(
            {k: r["structuredContent"][k] for k in ("error_code", "message")}, 170))
    wire.tool("write_file", {"filepath": "reports/ml_hiring_summary.md", "content": "rewritten"},
              lambda r: f"isError={str(r['isError']).lower()} " + short(r["structuredContent"]["error_code"]))

    wire.caption("Protocol errors are JSON-RPC error objects: -32700 parse error, -32600 invalid request "
                 "(including batch arrays, which MCP removed), -32601 unknown method, -32602 bad params, "
                 "-32002 unknown resource.")
    for message in ('{"jsonrpc": "2.0", "id": 50, "method": ',
                    [{"jsonrpc": "2.0", "id": 51, "method": "ping"}, {"jsonrpc": "2.0", "id": 52, "method": "ping"}],
                    {"jsonrpc": "1.0", "id": 53, "method": "ping"},
                    {"jsonrpc": "2.0", "id": "x-54", "method": "files/delete_everything"},
                    {"jsonrpc": "2.0", "id": 55, "method": "tools/call", "params": {"arguments": {}}},
                    {"jsonrpc": "2.0", "id": 56, "method": "resources/read", "params": {"uri": "resume://nobody.txt"}}):
        wire.send(message)
        wire.show_reply(wire.receive())
        time.sleep(wire.pause / 3)
    wire.request("ping", raw_id=57)
    stats = json.loads(wire.request("resources/read", {"uri": "stats://server"},
                                    show_reply=False)["result"]["contents"][0]["text"])
    wire.show_reply({}, "stats://server calls: " + short(stats["calls"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw JSON-RPC tour of the filesystem MCP server.")
    parser.add_argument("--part", default=",".join(PARTS), help=f"comma-separated: {', '.join(PARTS)}")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait at each step (for the video)")
    parser.add_argument("--transcripts", action="store_true",
                        help="run each part on its own and save it as scenarios/transcripts/07-10_*.txt")
    args = parser.parse_args()
    if args.transcripts:
        for number, part in zip((7, 8, 9, 10), ("discovery", "watch", "batch", "errors")):
            name = f"{number:02d}_mcp_{part}"
            result = subprocess.run([sys.executable, __file__, "--part", part], cwd=ROOT, capture_output=True,
                                    text=True, timeout=120)
            (ROOT / "scenarios" / "transcripts" / f"{name}.txt").write_text(
                f"# Transcript of scenario {name}: python scripts/mcp_tour.py --part {part}\n"
                f"# Raw JSON-RPC 2.0 over stdio with filesystem_mcp_server.py; run on {time.strftime('%Y-%m-%d')} "
                f"(exit code {result.returncode}). '→' sent, '←' response, '⇠' server notification.\n"
                + result.stdout + result.stderr)
            print(f"{'ok  ' if result.returncode == 0 else 'FAIL'} {name}")
        return
    parts = [p.strip() for p in args.part.split(",") if p.strip()]
    if unknown := [p for p in parts if p not in PARTS]:
        parser.error(f"unknown part(s): {unknown}")
    (ROOT / "logs").mkdir(exist_ok=True)
    wire = Wire(args.pause)
    try:
        handshake(wire)
        for part in parts:
            globals()[part](wire)
    finally:
        wire.close()


if __name__ == "__main__":
    main()
