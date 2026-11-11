# Agentic Profile Matching over MCP (Milestone 4)

Milestone 3's LangGraph screening agent, rebuilt on the **Model Context Protocol**. The
Milestone 1 file tools now live in a standalone MCP server,
[`filesystem_mcp_server.py`](filesystem_mcp_server.py). It speaks JSON-RPC 2.0 over stdio
(or streamable HTTP), exposes the files as MCP resources, and adds `watch_directory()` and
`batch_process()`. The agent in [`matching_agent.py`](matching_agent.py) no longer imports
the file tools: it spawns the server and reaches every resume, job description and report
through an MCP client. It can also connect other MCP servers, such as the reference *fetch*
server, so a job posting can be given by URL.

```bash
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-dev.txt
cp .env.example .env                              # then add your ANTHROPIC_API_KEY
.venv/bin/python scripts/build_index.py           # the Round 1 index, once
.venv/bin/python -m pytest                        # the offline suite, no API key
.venv/bin/python scripts/mcp_tour.py              # the server in raw JSON-RPC, no API key
.venv/bin/python chat.py --show-mcp               # the agent, with its MCP calls shown
```

**What changed since Milestone 3:** everything in
[`f538978...main`](https://github.com/ShreshthMi/Airtribe-MCP-Connectors/compare/f538978...main)
(`git diff f538978..HEAD`). This repository carries Milestone 3's history, so that diff is
exactly the Milestone 4 work. Milestone 3's own README, with the screening design and its
human evaluation, is at
[f538978/README.md](https://github.com/ShreshthMi/Airtribe-MCP-Connectors/blob/f538978/README.md).

## Deliverables

| The brief asks for | Where it is |
|---|---|
| MCP filesystem server, `filesystem_mcp_server.py` | [`filesystem_mcp_server.py`](filesystem_mcp_server.py) with [`jsonrpc_stdio.py`](jsonrpc_stdio.py), [`server_config.py`](server_config.py), [`mcp_watch.py`](mcp_watch.py), [`mcp_batch.py`](mcp_batch.py) |
| Milestone 1 tools converted to MCP, exposed as resources too | Tools `list_files`, `read_file`, `search_in_file`, `write_file`; resources `resume://`, `jd://`, `report://` ([below](#tools-and-resources)) |
| JSON-RPC 2.0 compliant interface | [`jsonrpc_stdio.py`](jsonrpc_stdio.py), tested on the wire by [`tests/test_mcp_stdio.py`](tests/test_mcp_stdio.py) |
| Proper error handling and status codes | [Error handling](#error-handling), one test per row |
| Resource discovery endpoints | `resources/list`, `resources/templates/list`, `config://server`, `stats://server` |
| Configuration management | [`server_config.py`](server_config.py) and [`mcp_server.toml`](mcp_server.toml): flags > `MCP_FS_*` > TOML > defaults |
| `watch_directory()` and `batch_process()` | [`mcp_watch.py`](mcp_watch.py), [`mcp_batch.py`](mcp_batch.py), [`tests/test_mcp_watch_batch.py`](tests/test_mcp_watch_batch.py) |
| Refactored LangGraph agent with an MCP client | [`matching_agent.py`](matching_agent.py), [`corpus.py`](corpus.py), [`mcp_client.py`](mcp_client.py), [`agent_tools.py`](agent_tools.py); [`tests/test_architecture.py`](tests/test_architecture.py) |
| Multi-MCP integration (bonus) | [`mcp_servers.json`](mcp_servers.json), [`mcp_hub.py`](mcp_hub.py), `chat.py --mcp-servers` |
| State machine / workflow diagram of agent ↔ MCP | [`docs/mcp_architecture.md`](docs/mcp_architecture.md): components, a message-by-message sequence, the state machine annotated with MCP calls |
| Test scenarios: MCP resource usage and agent workflow | [`tests/`](tests/) and scenarios 07-12 in [`scenarios/`](#tests-and-scenarios) |
| Demo video, 5-6 minutes | [`demo/mcp_demo.mp4`](demo/mcp_demo.mp4) ([below](#the-demo-video)) |

## How to run it

**The server on its own**

```bash
.venv/bin/python filesystem_mcp_server.py                     # stdio: what an MCP client spawns
.venv/bin/python filesystem_mcp_server.py --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
.venv/bin/python filesystem_mcp_server.py --help              # every configuration flag
```

**In the MCP Inspector** (needs Node.js):

```bash
npx @modelcontextprotocol/inspector .venv/bin/python filesystem_mcp_server.py
```

`mcp dev filesystem_mcp_server.py` works too: the module exposes a FastMCP object named
`mcp`. That route runs the SDK's own stdio transport, though, without the JSON-RPC checks
described below. The Inspector's CLI mode is a quick check:
`npx @modelcontextprotocol/inspector --cli .venv/bin/python filesystem_mcp_server.py --method tools/list`.

**The agent**

```bash
.venv/bin/python chat.py --show-mcp                        # each MCP request is shown in magenta
.venv/bin/python chat.py --session acme-backend            # a named, saved session; rerun to resume it
.venv/bin/python chat.py --sessions                        # list saved sessions
.venv/bin/python chat.py --watch                           # watch resumes/ and tell the agent about new files
.venv/bin/python chat.py --mcp-servers --show-mcp          # also connect the servers in mcp_servers.json (fetch)
```

**Without an API key:** `scripts/mcp_tour.py` (the server in raw JSON-RPC: scenarios 07-10),
`scripts/bench_mcp.py` (writes [`docs/benchmarks.md`](docs/benchmarks.md)) and the test suite.

## Part A: the filesystem MCP server

The server is built on the official `mcp` Python SDK, pinned at **1.30.0** (FastMCP API,
protocol revision **2025-11-25**). The current 2.x line renamed that API, and
`langchain-mcp-adapters` still needs `mcp<2`. The agent needs neither, because it calls the
Anthropic SDK directly. Milestone 1's `fs_tools.py` is imported from the vendored,
hash-checked copy in [`milestone2_rag/milestone1_tools/`](milestone2_rag/milestone1_tools/);
none of its logic is reimplemented. The server sets its sandbox root
(`$FS_ASSISTANT_ROOT`) once, at start-up, from the configuration.

### Tools and resources

| Tool | What it does |
|---|---|
| `list_files(directory, extension?)` | Milestone 1, limited to `resumes/`, `job_descriptions/`, `reports/` |
| `read_file(filepath)` | Milestone 1: text of a .txt, .md, .pdf or .docx file (PDF and DOCX extracted) |
| `search_in_file(filepath, keyword, whole_word?)` | Milestone 1: each match with its line and context |
| `write_file(filepath, content)` | Milestone 1, limited to `reports/`; **never replaces an existing file** |
| `watch_directory(directory?, extensions?)` | Starts a watch, by default on `resumes/`. Each new file, once its size is stable, becomes a resource, and connected clients get `notifications/resources/list_changed` plus a log message |
| `get_watch_events(watch_id, since?)` | The buffered events, for clients (or a model) that do not handle notifications |
| `unwatch_directory(watch_id)` | Stops a watch |
| `batch_process(paths? \| directory?, operation, keyword?)` | `read`, `search` or `stats` over many files at once: per-file results, `notifications/progress`, capped at `batch_max_files` |

| Resource | What it is |
|---|---|
| `resume://{filename}`, `jd://{filename}`, `report://{filename}` | Templates: any file in that folder, as text |
| one concrete resource per file | Listed by `resources/list`; a file the watcher sees is added and announced |
| `config://server` | The effective configuration and which layer each value came from |
| `stats://server` | How often each tool and resource has been used since start-up |

### How the brief's wording is read

- **"Expose all Milestone 1 tools as MCP resources."** In MCP, *tools* are actions the model
  chooses to call, and *resources* are data the application reads. The server does both:
  the four Milestone 1 functions are tools, and the files they work on are resources. The
  agent uses each half as intended. Claude opens files with the tools; the graph's own
  reads (Round 2 judging, suggestions) are `resources/read resume://…`.
- **"JSON-RPC 2.0 compliant."** Every MCP message is a JSON-RPC 2.0 message. On its own, the
  SDK's stdio transport answers a malformed line with a log notification rather than an
  error response, and an unknown method with -32602. [`jsonrpc_stdio.py`](jsonrpc_stdio.py)
  checks each line before the SDK sees it, and answers -32700, -32600 and -32601 as
  JSON-RPC prescribes. **JSON-RPC batches are not supported, deliberately.** MCP removed
  batching in revision 2025-06-18 ("Remove support for JSON-RPC batching", PR #416, in that
  revision's changelog), and 2025-11-25 did not bring it back. A batch array gets a single
  -32600 error naming the reason. MCP also forbids null ids, which get -32600 as well.
- **"Proper error handling and status codes."** There are three layers, below: JSON-RPC
  error objects for protocol errors, `isError` results with a stable code for tool
  failures, and HTTP status codes on the HTTP transport.

### Error handling

**Tool failures** are results, not protocol errors, as the 2025-11-25 tools page specifies.
They have `isError: true`, and both the text content and `structuredContent` are
`{"success": false, "error_code", "message", "filepath"}`. Messages never contain an
absolute host path.

| `error_code` | When | Example |
|---|---|---|
| `ACCESS_DENIED` | outside the sandbox, hidden, or outside the folders the operation may use | `read_file("../.env")`, `read_file("/etc/passwd")`, `write_file("resumes/x.md")` |
| `NOT_FOUND` | no such file or folder | `read_file("resumes/nobody.pdf")` |
| `UNSUPPORTED_FORMAT` | not an allowed extension, or not a text format for writing | `read_file("resumes/photo.png")`, `write_file("reports/run.py")` |
| `TOO_LARGE` | over `max_file_mb`, or more files than `batch_max_files` | a 2 KB file with `--max-file-mb 0.001` |
| `ALREADY_EXISTS` | `write_file` onto an existing file | `write_file("reports/ml_hiring_summary.md")` |
| `INVALID_ARGUMENT` | a usable type but an unusable value | an empty keyword, `operation: "delete"` |
| `READ_FAILED` | the file exists but cannot be parsed | a corrupt PDF |

**Protocol errors** are JSON-RPC error objects:

| Code | When |
|---|---|
| -32700 Parse error | the line is not JSON (`id: null`) |
| -32600 Invalid Request | a batch array, not an object, `"jsonrpc"` not `"2.0"`, a null or non-scalar id |
| -32601 Method not found | a method MCP does not define, e.g. `files/delete_everything` |
| -32602 Invalid params | params failing the request schema; a `tools/call` for a tool that does not exist; a resource that is hidden or not allowed |
| -32002 Resource not found | `resources/read` of an unknown URI or a missing file (with `data.uri` and `data.error_code`) |
| -32603 Internal error | a handler crashed, or a resource exists but cannot be read |

Arguments of the wrong type (`read_file(filepath=5)`) are rejected by the SDK before the
tool runs. The spec classes these as tool-execution errors, so they come back as an
`isError` result naming the field. **Over HTTP** (`--transport streamable-http`), the SDK's
transport adds status codes: 406 without `Accept: application/json, text/event-stream`, 404
for an unknown `mcp-session-id` or path, and 400 for a body that is not JSON. Tests:
[`test_mcp_server.py`](tests/test_mcp_server.py) (every tool code),
[`test_mcp_stdio.py`](tests/test_mcp_stdio.py) (every protocol code, on the wire),
[`test_mcp_http.py`](tests/test_mcp_http.py).

**Security point.** The model cannot read the API key through MCP: `read_file("../.env")`,
`read_file(".env")` and `resume://../.env` are all refused, and `.env` never appears in
`resources/list`. The server process is started with the SDK's minimal environment, so it
does not have `ANTHROPIC_API_KEY` at all. A test reads its `/proc/<pid>/environ` to check.

### Configuration

[`server_config.py`](server_config.py) builds one frozen configuration from four layers: command-line
flags, then `MCP_FS_*` environment variables, then [`mcp_server.toml`](mcp_server.toml) (or
`--config` / `MCP_FS_CONFIG`), then the defaults. It is validated at start-up: a bad value
stops the server with exit code 2 and a message naming the field and its layer, e.g.
`invalid configuration: batch_concurrency (cli): must be between 1 and 64`. The effective
configuration is published read-only as `config://server`.

| Setting | Default | Meaning |
|---|---|---|
| `root` | the project folder | the sandbox root for Milestone 1 (`--root`, `MCP_FS_ROOT`) |
| `readable_folders` / `writable_folders` | `resumes, job_descriptions, reports` / `reports` | the folder policy |
| `allowed_extensions` | `.txt .md .pdf .docx .json .csv` | what may be read or written |
| `max_file_mb` | 20 | size limit (Milestone 1's own read limit is the ceiling) |
| `batch_max_files`, `batch_concurrency`, `batch_preview_chars` | 200, 2, 160 | `batch_process` limits |
| `watch_debounce_ms`, `watch_force_polling`, `watch_buffer_size` | 300, false, 500 | `watch_directory` |
| `transport`, `host`, `port` | stdio, 127.0.0.1, 8765 | `stdio`, `streamable-http` or `sse` |
| `log_level` | INFO | logs go to stderr only; stdout is the protocol channel |

### `watch_directory()` and `batch_process()`

- **The watcher** runs `watchfiles.awatch` in a background task. It ignores hidden files,
  Office lock files (`~$…`) and other extensions. It announces a file only when its size is
  non-zero and the same across two checks, because PDF and DOCX writes are not atomic, and
  it re-checks a half-written file on a timer. Each file is announced once, with an
  increasing sequence number, into a bounded buffer. Watches stop on `unwatch_directory` and
  at shutdown.
- **`batch_process`** runs Milestone 1 calls in worker threads under a semaphore. Each file
  gets `{path, ok, error_code?, chars, preview, ms}`, so one corrupt file never fails the
  batch. Progress goes out after each file. **What "efficiently" means, measured**
  ([`docs/benchmarks.md`](docs/benchmarks.md), `scripts/bench_mcp.py`): one `batch_process`
  over the 100 resumes took about half the time of 100 separate `read_file` requests. For
  Claude it is one tool call with a compact result, not 100. More worker threads did *not*
  speed up reading this small, local corpus, because parsing is pure Python under the GIL:
  over 15 runs per setting, no setting beat sequential by more than the run-to-run
  variation, and 8 threads were clearly slower. The default is therefore 2, kept for slower
  storage, and the benchmark page says so.

## Part B: the agent, through MCP

**Same functionality, cleaner architecture.** The graph itself did not change: the same seven
nodes, edges and state ([`docs/state_machine.md`](docs/state_machine.md) regenerates
identically). What changed is underneath:

- **[`mcp_client.py`](mcp_client.py)** is a synchronous facade over the async MCP client: an
  anyio blocking portal holding one `ClientSession` over stdio. It is safe to call from
  Round 2's worker threads, receives the server's notifications, restarts a dead server
  once, and writes the server's log to `logs/`.
- **[`corpus.py`](corpus.py)** no longer imports `fs_tools` or Milestone 2's
  `resume_loader`. Its lock and its folder checks are gone, because the server enforces the
  policy in one place. The four file methods call the server and return the same shapes as
  before, so none of their callers changed.
- **[`agent_tools.py`](agent_tools.py)** builds Claude's file-tool schemas from the server's
  `tools/list` (MCP's `inputSchema` is the same JSON Schema as Anthropic's
  `input_schema`). Any server tool without a local wrapper is passed straight through. So
  `batch_process` and the watch tools reached Claude with no agent code for them, and a
  tool added to the server later will too. The four Milestone 1 tools keep a thin wrapper
  that marks file text as data, Milestone 3's prompt-injection defence.

**Evidence:**
- [`tests/test_architecture.py`](tests/test_architecture.py) parses `matching_agent.py`,
  `agent_tools.py`, `corpus.py` and `chat.py` and fails if any imports `fs_tools`,
  `resume_loader` or `extractors`. It includes cases showing the check catches each import
  style, and it was seen to fail when `import fs_tools` was added to `corpus.py`.
- [`tests/test_mcp_agent.py`](tests/test_mcp_agent.py) runs a screening against the real
  server process and checks:
  - the shortlist matches Milestone 3's in-memory run;
  - every file operation the agent made was counted by the server (`stats://server`);
  - Claude was offered the server's schemas and could call `batch_process`;
  - the server's environment has no API key;
  - a killed server is restarted.

  Milestone 3's own tests run unchanged, with `corpus.py`'s file tests now going through the
  real server.
- **Cost of the hop** ([`docs/benchmarks.md`](docs/benchmarks.md)): about 3 ms per
  `read_file` over stdio, against a fraction of a millisecond in-process. The server starts
  in well under a second, once per agent process. Both are small next to a Claude call.

**Durable sessions (from the Milestone 3 review).** `chat.py` now checkpoints every
conversation to SQLite (`.state/checkpoints.sqlite`, LangGraph's `SqliteSaver`) under a
session name. `quit`, Ctrl-D, a closed terminal or a crash leave the graph paused at
`human_feedback`. `chat.py --session NAME` resumes it in a new process with its
requirements, shortlist, verdicts and history, without re-screening; `--sessions` lists
them. Tested in [`tests/test_chat.py`](tests/test_chat.py).

## Multi-MCP (bonus)

[`mcp_servers.json`](mcp_servers.json) lists MCP servers in the same shape Claude Desktop
uses: this project's filesystem server, and the reference
[fetch server](https://pypi.org/project/mcp-server-fetch/) run with `uvx mcp-server-fetch`.
The fetch server needs `mcp>=2.3`, so it runs in its own environment, which is also the
point of MCP. [`mcp_hub.py`](mcp_hub.py) starts each server, names its tools
`<server>__<tool>` (`fetch__fetch`, `filesystem__read_file`) so tools from different
servers cannot collide, and routes each call by its prefix. Each server gets only the
SDK's minimal environment plus the variables its entry lists, with `${NAME}` expansion.

With `chat.py --mcp-servers`, Claude is offered `fetch__fetch` next to the filesystem tools.
`parse_jd` also fetches a posting given by URL ("Screen candidates for the posting at
https://…") through the fetch server, then screens as usual and writes the report through
the filesystem server. The fetch server refuses private and loopback addresses by default,
which is kept. Tests: [`tests/test_mcp_hub.py`](tests/test_mcp_hub.py), including the real
fetch server against a local web server (with `--allow-private-ips` for that test only).

## Tests and scenarios

`.venv/bin/python -m pytest`: 312 tests, offline, no API key, about half a minute. That is
Milestone 3's suite plus the MCP tests:

| File | What it checks |
|---|---|
| `test_mcp_server.py` | every tool, every tool error code, resources, templates, resource error codes, stats (in-memory MCP sessions) |
| `test_mcp_stdio.py` | the real server process in raw JSON-RPC lines: integer and string ids, every protocol error code, nothing but protocol on stdout |
| `test_mcp_http.py` | the streamable HTTP transport's status codes |
| `test_server_config.py` | precedence of the four layers, fail-fast validation, exit code 2 |
| `test_mcp_watch_batch.py` | new, ignored and half-written files under watch; `list_changed`; batch errors, progress, limits, concurrency |
| `test_mcp_agent.py`, `test_architecture.py` | Part B, as above |
| `test_mcp_hub.py` | namespacing and routing across two servers, variable expansion, a URL posting through the graph, the real fetch server |
| `test_chat.py` | adds a session saved, then resumed in a fresh agent |

| Scenario | What it shows | Transcript |
|---|---|---|
| 07 MCP discovery | handshake, `tools/list`, `resources/list`, templates, `config://server`, a tool call | [07](scenarios/transcripts/07_mcp_discovery.txt) |
| 08 watch a new resume | a resume arrives; `list_changed` and a log message are pushed; it becomes a resource | [08](scenarios/transcripts/08_mcp_watch.txt) |
| 09 batch over 100 resumes | progress notifications; per-file errors in a mixed batch | [09](scenarios/transcripts/09_mcp_batch.txt) |
| 10 errors | `../.env`, an absolute path, a missing file, an unsupported format, an overwrite; every JSON-RPC error code | [10](scenarios/transcripts/10_mcp_errors.txt) |
| 11 agent via MCP | a live screening with `--show-mcp`: every file operation visible as an MCP request | [`scenarios/11_agent_via_mcp.txt`](scenarios/11_agent_via_mcp.txt) |
| 12 multi-MCP | a posting fetched by URL through the fetch server, the report written through the filesystem server | [`scenarios/12_multi_mcp.txt`](scenarios/12_multi_mcp.txt) |

Scenarios 07-10 come from `python scripts/mcp_tour.py --transcripts` (raw JSON-RPC, no
key). Scenarios 11-12 are chat scripts for `python scripts/run_scenarios.py 11_agent_via_mcp
12_multi_mcp`, which runs them against the real model and needs an API key. Milestone 3's
scenarios 01-06 are unchanged.

## The demo video

[`demo/mcp_demo.mp4`](demo/mcp_demo.mp4) is rendered by `python scripts/record_mcp_demo.py`
from terminal sessions recorded under a pseudo-terminal and replayed at their real speed,
with captions. It is not a screen capture and has no voice-over. It has four segments:

1. the server in raw JSON-RPC: handshake, discovery and errors;
2. `watch_directory` catching a new resume, and `batch_process` over 100 resumes;
3. the agent live through MCP, with a posting fetched by URL;
4. the test suite.

In the live chat, waits longer than 4 seconds with no output are shortened to 4 seconds,
and the title card says so. The recorded sessions are in `demo/segments/`, and `--replay`
re-renders them. Milestone 3's video is still at [`demo/demo.mp4`](demo/demo.mp4).

## What the Milestone 3 review asked for

| Review point | What was done |
|---|---|
| Replace the in-memory checkpointer and per-run thread ids with durable state, so conversations resume across sessions | Done: SQLite checkpoints and named sessions ([above](#part-b-the-agent-through-mcp)) |
| Expand the evaluation with independent labellers and broader resume coverage, then calibrate ranking and hire thresholds | **Not done in this milestone.** It needs people other than the author to label postings, which could not be arranged for this deadline. Milestone 3's evaluation tooling (`scripts/human_eval.py`, `scripts/evaluate.py`) is unchanged and ready for that. Its caveats still apply: one non-independent labeller, three real postings, Round 2 re-ranking not shown to help. |

## Limitations

- **The HTTP transport has no authentication.** It binds to 127.0.0.1 by default; anything
  beyond local use would need the SDK's auth support configured. The agent uses stdio.
- **New resumes are not indexed automatically.** `watch_directory` makes a new resume
  readable through MCP at once, and `chat.py --watch` tells the agent about it. Round 1's
  embedding search finds it only after `scripts/build_index.py` is run again.
- **`batch_process` threads do not speed up local parsing** (see above). The win is fewer
  round trips and tool calls.
- **One server process per agent process.** The MCP hop costs a few milliseconds per call.
  A crashed server is restarted once, then the error is reported.
- **Matching is not hiring.** Everything in Milestone 3's "Limitations and responsible use"
  still holds: recommendations support a human decision, protected characteristics are
  never ranked on, resumes are untrusted input, and the evaluation is internal.

## How this was built

Milestone 4 was built by an AI coding assistant (Claude Code, Anthropic's coding agent),
working in a cloud session on 11 November 2026 from a handoff brief the author prepared and
approved. The author made the decisions recorded there:
- the scope, building on Milestone 3;
- the repository and branch;
- how the video is produced;
- attempting the bonus;
- disclosing AI use in this README rather than in commit trailers.

The code, tests, documents and video in the Milestone 4 commits were written or produced
by the assistant. The author reviewed them and is responsible for the submission.
Milestones 1-3 are unchanged except where this README says so.

## Repository layout

```
filesystem_mcp_server.py   the MCP server: tools, resources, error codes (Part A)
jsonrpc_stdio.py           stdio transport with JSON-RPC 2.0 checks
server_config.py           configuration: flags > MCP_FS_* > mcp_server.toml > defaults
mcp_watch.py, mcp_batch.py watch_directory() and batch_process()
mcp_client.py              synchronous MCP client used by the agent (Part B)
mcp_hub.py                 several MCP servers, namespaced tools (bonus); mcp_servers.json
corpus.py                  the agent's files (through MCP) and Round 1 search (Milestone 2)
matching_agent.py          the LangGraph graph (unchanged structure)
agent_tools.py             Claude's tools: local ones plus the server's tools/list
chat.py                    terminal chat: --show-mcp, --session, --watch, --mcp-servers
screening.py, claude_tasks.py, llm.py, embedders.py   Milestone 3, unchanged in role
milestone2_rag/            Milestone 2 (and Milestone 1's tools inside it), vendored verbatim
docs/                      mcp_architecture.md, state_machine.md, benchmarks.md, walkthrough.md
scripts/                   mcp_tour.py, bench_mcp.py, record_mcp_demo.py, run_scenarios.py, ...
scenarios/                 chat scenarios 01-06 and 11-12; transcripts 01-10 (and 11-12 when run live)
demo/                      mcp_demo.mp4 and its segments; Milestone 3's demo.mp4
tests/                     the pytest suite
resumes/ job_descriptions/ reports/   the data the server serves
```
