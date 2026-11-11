"""
Chat with the matching agent in the terminal.

  python chat.py                                               # type messages yourself
  python chat.py --jd job_descriptions/frontend_engineer.txt   # open with a job description file
  python chat.py --script scenarios/01_react_search.txt        # replay a scripted conversation
  python chat.py --session acme-frontend                       # start, or resume, a named session
  python chat.py --sessions                                    # list the saved sessions
  python chat.py --show-mcp --watch                            # show MCP calls; announce new resumes
  python chat.py --mcp-servers                                 # also connect mcp_servers.json (fetch)

The graph runs until it reaches human_feedback, where interrupt() pauses it. This loop
then reads the next message and resumes the graph with Command(resume=message). While a
node runs, its progress lines and tool calls are streamed so you can watch it work.

Every conversation is checkpointed to SQLite (.state/checkpoints.sqlite) under its session
name, so leaving - quit, Ctrl-D, a closed terminal, a crash - keeps it paused at
human_feedback, and `--session NAME` picks it up later with its shortlist and history.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Quieten the embedding model's start-up chatter before anything imports it.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # the hub sets its own log level from this on import
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from dotenv import load_dotenv  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402

from corpus import CorpusError  # noqa: E402
from llm import LLMError  # noqa: E402
from matching_agent import MAX_MESSAGE_CHARS, QUIT_WORDS, MatchingAgent  # noqa: E402

NODE_LABELS = {
    "parse_jd": "Parse JD",
    "extract_requirements": "Extract requirements",
    "search_resumes": "Search resumes - Round 1",
    "rank_candidates": "Rank candidates - Round 2",
    "generate_report": "Generate report - Round 3",
    "agent_turn": "Agent answered",
}
GREETING = ("Describe who you are looking for in one line, or give a job description file such as "
            "job_descriptions/frontend_engineer.txt. Type quit to leave.")

ROOT = Path(__file__).resolve().parent
STATE_FILE_ENV = "CHAT_STATE_FILE"  # where sessions are saved; tests point it at a temporary file

console = Console()


def state_file() -> Path:
    return Path(os.environ.get(STATE_FILE_ENV) or ROOT / ".state" / "checkpoints.sqlite")


def open_checkpointer() -> SqliteSaver:
    """Durable conversation state: LangGraph's SQLite checkpointer, one thread per session name."""
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))  # Round 2 runs nodes' work in threads


def saved_sessions(saver: SqliteSaver) -> list[tuple[str, str]]:
    """(session name, one-line summary) for every saved session, most recent first."""
    latest: dict[str, tuple] = {}
    for item in saver.list(None):
        name = item.config["configurable"]["thread_id"]
        latest.setdefault(name, (item.checkpoint.get("ts", ""), item.checkpoint.get("channel_values", {})))
    rows = []
    for name, (ts, values) in sorted(latest.items(), key=lambda kv: kv[1][0], reverse=True):
        role = (values.get("requirements") or {}).get("role_title") or "no job screened yet"
        rows.append((name, f"{ts[:16].replace('T', ' ')}  {role}, round {values.get('screening_round', 0)}, "
                           f"{len(values.get('messages', []))} messages"))
    return rows


def show_mcp_call(method: str, name: str, arguments: dict) -> None:
    """--show-mcp: one dim line per request the agent sends to the MCP server."""
    shown = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 60 else v) for k, v in arguments.items()}
    console.print(f"  [magenta]↳ MCP {method}[/magenta] [dim]{name} {json.dumps(shown, ensure_ascii=False)}[/dim]")


def new_files(agent) -> list[str]:
    """--watch: files the server announced since the last look (from its watch_directory log messages)."""
    corpus = agent.corpus
    if getattr(corpus, "_mcp", None) is None:
        return []
    found = []
    for note in corpus.mcp.drain_notifications():
        data = note.get("data")
        if isinstance(data, dict) and data.get("event") == "new_file":
            found.append(data["path"])
    return found


def stream(app, graph_input, config) -> None:
    """Run the graph until it pauses or ends, printing progress and new assistant messages."""
    for mode, chunk in app.stream(graph_input, config, stream_mode=["updates", "custom"]):
        if mode == "custom":
            console.print(f"  [dim]{chunk['progress']}[/dim]")
            continue
        for node, update in chunk.items():
            if node not in NODE_LABELS:  # human_feedback, and the "__interrupt__" marker when the graph pauses
                continue
            console.print(f"[cyan]✓ {NODE_LABELS[node]}[/cyan]")
            for message in (update or {}).get("messages", []):
                if message["role"] == "assistant":
                    console.print(Panel(Markdown(message["content"]), title="Agent", border_style="green"))


def read_script(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def ask(prompt: str) -> str:
    try:
        return console.input(prompt).strip()
    except (EOFError, KeyboardInterrupt):  # Ctrl-D, Ctrl-C or a closed input stream
        return "quit"


def next_message(script: list[str] | None, pause: float) -> str:
    if script is None:
        return ask("[bold]You:[/bold] ")
    if not script:
        return "quit"
    time.sleep(pause)
    message = script.pop(0)
    console.print(f"[bold]You:[/bold] {message}")
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the resume matching agent.")
    parser.add_argument("--jd", help="Open with this job description (text or a path under job_descriptions/)")
    parser.add_argument("--script", type=Path, help="Replay user messages from a file, one per line")
    parser.add_argument("--pause", type=float, default=0.0, help="Seconds to wait before each scripted message")
    parser.add_argument("--session", help="Session name: resumes it if saved, else starts it (default: a new name)")
    parser.add_argument("--sessions", action="store_true", help="List the saved sessions and exit")
    parser.add_argument("--show-mcp", action="store_true", help="Show each request the agent sends to the MCP server")
    parser.add_argument("--watch", action="store_true", help="Watch resumes/ through the MCP server and tell the "
                                                              "agent about new files")
    parser.add_argument("--mcp-servers", nargs="?", const=str(ROOT / "mcp_servers.json"), metavar="JSON",
                        help="Also connect the other MCP servers in this file (default mcp_servers.json), "
                             "e.g. fetch, so a posting can be given by URL")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    saver = open_checkpointer()
    if args.sessions:
        for name, summary in saved_sessions(saver) or [("(none)", f"no sessions saved in {state_file()}")]:
            console.print(f"[bold]{name}[/bold]  [dim]{summary}[/dim]")
        return
    script = read_script(args.script) if args.script else None
    agent = MatchingAgent()
    if args.show_mcp:
        agent.mcp_observer = show_mcp_call
    if args.mcp_servers:
        from mcp_hub import MCPHub  # starts each server listed there except filesystem, which the corpus runs
        agent.extra_mcp = MCPHub(Path(args.mcp_servers), exclude=("filesystem",), on_call=agent.mcp_observer)
        console.print(f"[dim]Also connected: {', '.join(agent.extra_mcp.clients) or 'no other MCP servers'}.[/dim]")
    try:
        converse(agent, agent.build(checkpointer=saver), args, script)
    finally:
        agent.close()


def converse(agent, app, args, script: list[str] | None) -> None:
    session = args.session or datetime.now().strftime("chat-%Y%m%d-%H%M%S")
    config = {"configurable": {"thread_id": session}}
    console.print(Panel(GREETING, title="Agentic Profile Matching", border_style="cyan"))
    if args.watch:
        failed, watch = agent.corpus.call_mcp_tool("watch_directory", {"directory": "resumes"})
        console.print(f"[dim]Watching resumes/ through the MCP server ({watch.get('watch_id') or watch}).[/dim]")

    saved = app.get_state(config)
    if saved.next:  # a saved session, paused at human_feedback: carry on where it stopped
        values = saved.values
        console.print(f"[cyan]Resuming session {session}[/cyan] [dim]({(values.get('requirements') or {}).get('role_title')}"
                      f", screening round {values.get('screening_round', 0)}, {len(values.get('messages', []))} "
                      "messages so far)[/dim]")
        if values.get("messages"):
            console.print(Panel(Markdown(values["messages"][-1]["content"]), title="Agent (last reply)",
                                border_style="green"))
        message = next_message(script, args.pause)
        if message.lower() in QUIT_WORDS:
            goodbye(agent, session, saved=True)
            return
        graph_input = Command(resume=message)
    else:
        first = args.jd or next_message(script, args.pause)
        while len(first) > MAX_MESSAGE_CHARS:
            console.print(f"[red]That message is {len(first):,} characters; the limit is {MAX_MESSAGE_CHARS:,}. "
                          "Save a long job description to a file in job_descriptions/ and give its path.[/red]")
            first = next_message(script, args.pause)
        if not first or first.lower() in QUIT_WORDS:
            return
        graph_input = {"messages": [{"role": "user", "content": first}], "jd_input": first}
    while True:
        try:
            stream(app, graph_input, config)
            failure = None
        except (LLMError, CorpusError) as exc:
            failure = f"That step failed: {exc}"
        except Exception as exc:  # a bug: show it, but keep the conversation so the user can retry
            failure = f"Unexpected error ({type(exc).__name__}): {exc}"
        if failure:
            console.print(f"[red]{failure}[/red]")
            if script is not None:  # nobody is at the keyboard to decide whether to retry
                sys.exit(1)
            if ask("Press Enter to retry, or type quit: ").lower() in QUIT_WORDS:
                return
            graph_input = None  # None resumes the graph at the step that failed
            continue
        if not app.get_state(config).next:  # the graph reached END
            goodbye(agent, session, saved=False)
            return
        message = next_message(script, args.pause)
        if message.lower() in QUIT_WORDS:  # leave the graph paused at human_feedback, so the session can resume
            goodbye(agent, session, saved=True)
            return
        if args.watch and (arrived := new_files(agent)):
            console.print(f"[yellow]New resume(s) from the MCP watch: {', '.join(arrived)}[/yellow]")
            message += f"\n\n[watch_directory: new file(s) arrived since my last message: {', '.join(arrived)}]"
        graph_input = Command(resume=message)


def goodbye(agent, session: str, saved: bool) -> None:
    u = agent.claude.usage
    console.print(f"[cyan]Goodbye.[/cyan] [dim]Claude usage: {u['calls']} calls, {u['input_tokens']:,} input "
                  f"tokens (+{u['cache_read_input_tokens']:,} from cache), {u['output_tokens']:,} output "
                  f"tokens on {agent.claude.model}.[/dim]")
    if saved:
        console.print(f"[dim]Session saved. Resume it with: python chat.py --session {session}[/dim]")


if __name__ == "__main__":
    main()
