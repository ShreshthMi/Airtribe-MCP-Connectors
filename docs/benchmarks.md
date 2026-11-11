# Benchmarks

Written by `python scripts/bench_mcp.py` on 2026-11-11 14:41 UTC.
Machine: Linux 6.18.44-fc-v378 (x86_64), 4 CPUs, Python 3.12.3. Corpus: the 100 files in `resumes/` (txt, md, pdf, docx).
Each figure is the median of 15 runs after a warm-up run; rerun the script to reproduce on your machine.

## batch_process: one request instead of one per file

| | median wall time for all 100 resumes, seen by the client |
|---|---|
| one `batch_process` request | 156 ms |
| 100 `read_file` requests, one after another | 370 ms |

This is where batch_process saves: one JSON-RPC round trip instead of 100, and for Claude one
tool call and one compact result (lengths and previews, not 100 full texts) to read.

## batch_process: how many files at once

| batch_concurrency | median server time to read all 100 resumes | range of runs, ms |
|---|---|---|
| 1 | 167 ms | 147-200 (middle half 152-189) |
| 2 (default) | 172 ms | 150-196 (middle half 158-181) |
| 4 | 183 ms | 156-236 (middle half 169-194) |
| 8 | 207 ms | 170-252 (middle half 200-215) |

No clear gain from concurrency on this machine: the differences between levels are within the run-to-run variation (the last column). The files are small and on a local disk, and pypdf and python-docx parse in pure Python under the GIL, so worker threads mostly queue for it. Concurrency is kept (and configurable) for slow or remote storage, where reads wait on I/O; that case was not measured here.

## The cost of the MCP hop

| | median per `read_file` | p95 |
|---|---|---|
| Milestone 1 `read_file`, called directly (Milestone 3) | 0.33 ms | 8.70 ms |
| the same call through MCP (JSON-RPC over stdio) | 2.73 ms | 12.46 ms |

The hop adds about 2.4 ms per call. Starting the server process
(Python start-up, imports, scanning the data folders) took 639 ms, paid once per
agent process. Both are small next to a Claude call, which takes seconds.
