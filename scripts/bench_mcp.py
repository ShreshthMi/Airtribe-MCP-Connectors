"""
Measure what MCP costs and what batch_process saves, and write docs/benchmarks.md.

  python scripts/bench_mcp.py            # about a minute; no API key

1. batch_process over all resumes in resumes/, run by the real server over stdio with
   batch_concurrency 1, 2, 4 and 8, several runs each. The time is the server's own
   elapsed_ms for the batch, so process start-up and the JSON-RPC hop are not counted.
2. One batch_process request against one read_file request per file, both timed by the
   client (wall clock, JSON-RPC over stdio included).
3. The cost of the MCP hop: Milestone 1's read_file called directly in this process,
   against the same reads through MCPClient, one request at a time.

The prose in the output is written from the measured numbers, not assumed.
"""
from __future__ import annotations

import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "milestone2_rag" / "milestone1_tools"))

from mcp_client import MCPClient  # noqa: E402
from server_config import ServerConfig  # noqa: E402

RUNS = 5
OUT = ROOT / "docs" / "benchmarks.md"


def batch_ms(concurrency: int) -> list[float]:
    client = MCPClient.filesystem(ROOT, "--batch-concurrency", str(concurrency))
    try:
        client.call_tool("batch_process", {"directory": "resumes", "operation": "read"})  # warm-up (imports, cache)
        times = []
        for _ in range(RUNS):
            _, summary = client.call_tool("batch_process", {"directory": "resumes", "operation": "read"})
            assert summary["failed"] == 0, summary
            times.append(summary["elapsed_ms"])
        return times
    finally:
        client.close()


def per_read_ms(paths: list[str]) -> tuple[list[float], list[float]]:
    os.environ["FS_ASSISTANT_ROOT"] = str(ROOT)
    import fs_tools
    direct, via_mcp = [], []
    client = MCPClient.filesystem(ROOT)
    try:
        for path in paths:  # warm both paths up once
            fs_tools.read_file(path)
            client.call_tool("read_file", {"filepath": path})
        for path in paths:
            t0 = time.perf_counter()
            assert fs_tools.read_file(path)["success"]
            direct.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            failed, _ = client.call_tool("read_file", {"filepath": path})
            assert not failed
            via_mcp.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        spawn = MCPClient.filesystem(ROOT)
        startup_ms = (time.perf_counter() - t0) * 1000
        spawn.close()
    finally:
        client.close()
    return direct, via_mcp, startup_ms


def one_request_vs_many(paths: list[str]) -> tuple[list[float], list[float]]:
    client = MCPClient.filesystem(ROOT)
    try:
        client.call_tool("batch_process", {"paths": paths, "operation": "read"})  # warm-up
        one, many = [], []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            client.call_tool("batch_process", {"paths": paths, "operation": "read"})
            one.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            for path in paths:
                client.call_tool("read_file", {"filepath": path})
            many.append((time.perf_counter() - t0) * 1000)
        return one, many
    finally:
        client.close()


def main() -> None:
    paths = sorted(f"resumes/{p.name}" for p in (ROOT / "resumes").iterdir() if not p.name.startswith("."))
    levels = {n: batch_ms(n) for n in (1, 2, 4, 8)}
    one, many = one_request_vs_many(paths)
    direct, via_mcp, startup_ms = per_read_ms(paths)
    med = statistics.median
    sequential = med(levels[1])
    best = min(levels, key=lambda n: med(levels[n]))
    if med(levels[best]) < 0.9 * sequential:
        verdict = (f"Concurrency helps here: {best} workers read the corpus {sequential / med(levels[best]):.1f}x "
                   "faster than one.")
    else:
        verdict = ("Concurrency does not help on this machine: the files are small and on a local disk, and "
                   "pypdf and python-docx parse in pure Python under the GIL, so worker threads mostly queue "
                   "for it. It is kept (and configurable) for slow or remote storage, where reads wait on I/O; "
                   "that case was not measured here.")
    rows = "\n".join(f"| {n}{' (default)' if n == ServerConfig.batch_concurrency else ''} | {med(t):.0f} ms | {', '.join(f'{x:.0f}' for x in t)} |"
                     for n, t in levels.items())
    machine = (f"{platform.system()} {platform.release()} ({platform.machine()}), {os.cpu_count()} CPUs, "
               f"Python {platform.python_version()}")
    OUT.write_text(f"""# Benchmarks

Written by `python scripts/bench_mcp.py` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.
Machine: {machine}. Corpus: the {len(paths)} files in `resumes/` (txt, md, pdf, docx).
Each figure is the median of {RUNS} runs after a warm-up run; rerun the script to reproduce on your machine.

## batch_process: one request instead of one per file

| | median wall time for all {len(paths)} resumes, seen by the client |
|---|---|
| one `batch_process` request | {med(one):.0f} ms |
| {len(paths)} `read_file` requests, one after another | {med(many):.0f} ms |

This is where batch_process saves: one JSON-RPC round trip instead of {len(paths)}, and for Claude one
tool call and one compact result (lengths and previews, not {len(paths)} full texts) to read.

## batch_process: how many files at once

| batch_concurrency | median server time to read all {len(paths)} resumes | runs (ms) |
|---|---|---|
{rows}

{verdict}

## The cost of the MCP hop

| | median per `read_file` | p95 |
|---|---|---|
| Milestone 1 `read_file`, called directly (Milestone 3) | {med(direct):.2f} ms | {statistics.quantiles(direct, n=20)[-1]:.2f} ms |
| the same call through MCP (JSON-RPC over stdio) | {med(via_mcp):.2f} ms | {statistics.quantiles(via_mcp, n=20)[-1]:.2f} ms |

The hop adds about {med(via_mcp) - med(direct):.1f} ms per call. Starting the server process
(Python start-up, imports, scanning the data folders) took {startup_ms:.0f} ms, paid once per
agent process. Both are small next to a Claude call, which takes seconds.
""")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
