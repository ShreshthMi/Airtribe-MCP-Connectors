# Benchmarks

Written by `python scripts/bench_mcp.py` on 2026-11-11 14:16 UTC.
Machine: Linux 6.18.44-fc-v378 (x86_64), 4 CPUs, Python 3.12.3. Corpus: the 100 files in `resumes/` (txt, md, pdf, docx).
Each figure is the median of 5 runs after a warm-up run; rerun the script to reproduce on your machine.

## batch_process: one request instead of one per file

| | median wall time for all 100 resumes, seen by the client |
|---|---|
| one `batch_process` request | 212 ms |
| 100 `read_file` requests, one after another | 411 ms |

This is where batch_process saves: one JSON-RPC round trip instead of 100, and for Claude one
tool call and one compact result (lengths and previews, not 100 full texts) to read.

## batch_process: how many files at once

| batch_concurrency | median server time to read all 100 resumes | runs (ms) |
|---|---|---|
| 1 | 159 ms | 162, 153, 155, 213, 159 |
| 2 | 150 ms | 169, 149, 149, 214, 150 |
| 4 (default) | 200 ms | 200, 183, 192, 228, 205 |
| 8 | 221 ms | 242, 221, 203, 239, 200 |

Concurrency does not help on this machine: the files are small and on a local disk, and pypdf and python-docx parse in pure Python under the GIL, so worker threads mostly queue for it. It is kept (and configurable) for slow or remote storage, where reads wait on I/O; that case was not measured here.

## The cost of the MCP hop

| | median per `read_file` | p95 |
|---|---|---|
| Milestone 1 `read_file`, called directly (Milestone 3) | 0.33 ms | 8.94 ms |
| the same call through MCP (JSON-RPC over stdio) | 3.00 ms | 12.37 ms |

The hop adds about 2.7 ms per call. Starting the server process
(Python start-up, imports, scanning the data folders) took 703 ms, paid once per
agent process. Both are small next to a Claude call, which takes seconds.
