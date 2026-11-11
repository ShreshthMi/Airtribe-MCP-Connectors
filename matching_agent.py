"""
The LangGraph matching agent.

    START -> parse_jd -> extract_requirements -> search_resumes -> rank_candidates
          -> generate_report -> human_feedback <-> agent_turn
                                      |                 |-- requirements changed -> search_resumes
                                      |                 '-- a different job      -> parse_jd
                                      '-- user says quit -> END

search_resumes, rank_candidates and generate_report are the three screening rounds:
  Round 1  Milestone 2's hybrid RAG ranks all 100 resumes; the top 10 go forward
  Round 2  Claude judges each of the 10 against every requirement, quoting the resume
  Round 3  hire / borderline / no-hire, improvement suggestions, and a Markdown report

Every node is a method of MatchingAgent that takes the state and returns only the keys it
changes. human_feedback pauses the graph with LangGraph's interrupt() until the user
replies; agent_turn then lets Claude answer with tools, and its result decides whether the
graph waits again, re-screens, or starts over on a new job.

Regenerate the state machine diagram from the compiled graph with:
  python matching_agent.py --diagram
"""
from __future__ import annotations

import argparse
import json
import operator
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import claude_tasks
from agent_tools import SYSTEM_PROMPT, Toolbox, brief_args, state_snapshot, tools_for
from corpus import ROUND1_SIZE, Corpus
from llm import Claude, LLMError
from screening import (LABEL, current_verdicts, describe_change, interview_checks, rank_by_fit, ranking_changes,
                       render_report, requirement_key, suggestion_key, weighted_requirements)

REPORT_FIELDS = ("rank", "candidate_id", "name", "resume_path", "rag_score", "fit_score", "recommendation",
                 "recommendation_reason", "must_haves_met")
SHORTLIST_SIZE = 10      # candidates kept after Round 2
MAX_TOOL_ROUNDS = 8      # tool calls Claude may chain in one chat turn
SUMMARY_TOP_N = 5        # candidates listed in the chat after each round
MAX_MESSAGE_CHARS = 8000  # longer messages are refused; a job description that long belongs in a file
MAX_HISTORY_MESSAGES = 40  # how much of the conversation Claude is shown each turn
QUIT_WORDS = {"quit", "exit", "bye", "q"}
ASK_FOR_A_JOB = ("Describe who you are looking for in one line, or give a job description file such as "
                 "`job_descriptions/frontend_engineer.txt`.")
DOCUMENT_SUFFIXES = (".txt", ".md", ".pdf", ".docx")
DIAGRAM_FILE = Path(__file__).resolve().parent / "docs" / "state_machine.md"


class ScreeningState(TypedDict, total=False):
    # Conversation history, as {"role": "user" | "assistant", "content": str}.
    # The operator.add reducer means a node's messages are appended, never replaced.
    messages: Annotated[list[dict], operator.add]
    # The job and its requirements
    jd_input: str                   # what the user gave: text or a file path
    job_description: str            # the text that was screened
    requirements: dict              # role_title, must_have, nice_to_have, notes
    previous_requirements: dict | None
    # The shortlist and the reasoning behind it
    round1: list[dict]              # Round 1 top 10 (+ the previous shortlist on a re-screen), with RAG scores
    verdicts: dict[str, dict]       # candidate_id -> requirement key -> Round 2 verdict
    unjudged: list[dict]            # Round 1 candidates Claude could not judge this round, with the reason
    screened: list[dict]            # everyone judged this round, ranked by fit, with Round 3 recommendations
    shortlist: list[dict]           # the top SHORTLIST_SIZE of screened
    previous_shortlist: list[dict]
    ranking_changes: list[dict]
    suggestions: dict[str, list[str]]  # Round 3 suggestions for borderline candidates
    screening_round: int
    report_path: str
    # Control
    next_step: str                  # which way the graph goes after the current node
    tool_trace: list[str]           # tools called in the latest chat turn


class MatchingAgent:
    def __init__(self, claude: Claude | None = None, corpus: Corpus | None = None, today: date | None = None):
        self._claude, self._corpus = claude, corpus
        self.today = today or date.today()
        self.mcp_observer = None  # called with (method, name, arguments) before each MCP request, e.g. by chat.py

    # Created on first use, so drawing the diagram needs neither an API key nor an index.
    @property
    def claude(self) -> Claude:
        if self._claude is None:
            self._claude = Claude()
        return self._claude

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            self._corpus = Corpus(on_mcp_call=self.mcp_observer)
        return self._corpus

    def close(self) -> None:
        """Stop the MCP server the corpus started, if any."""
        if self._corpus is not None and hasattr(self._corpus, "close"):
            self._corpus.close()

    def build(self, checkpointer=None):
        graph = StateGraph(ScreeningState)
        graph.add_node("parse_jd", self.parse_jd)
        graph.add_node("extract_requirements", self.extract_requirements)
        graph.add_node("search_resumes", self.search_resumes)
        graph.add_node("rank_candidates", self.rank_candidates)
        graph.add_node("generate_report", self.generate_report)
        graph.add_node("human_feedback", self.human_feedback)
        graph.add_node("agent_turn", self.agent_turn)
        graph.add_edge(START, "parse_jd")
        graph.add_conditional_edges("parse_jd", route_after_parse, ["extract_requirements", "human_feedback"])
        graph.add_conditional_edges("extract_requirements", route_after_requirements,
                                    ["search_resumes", "human_feedback"])
        graph.add_edge("search_resumes", "rank_candidates")
        graph.add_edge("rank_candidates", "generate_report")
        graph.add_edge("generate_report", "human_feedback")
        graph.add_conditional_edges("human_feedback", route_after_feedback, ["agent_turn", "human_feedback", END])
        graph.add_conditional_edges("agent_turn", route_after_agent, ["human_feedback", "search_resumes", "parse_jd"])
        # The checkpointer is what lets interrupt() pause the graph and resume it later.
        return graph.compile(checkpointer=checkpointer or InMemorySaver())

    # ---- Parse JD -------------------------------------------------------------- #
    def parse_jd(self, state: ScreeningState) -> dict:
        """Get the job description text: pasted, or read from a file with read_file on the MCP server."""
        text = (state.get("jd_input") or "").strip()
        if "\n" not in text and text.lower().endswith(DOCUMENT_SUFFIXES):
            result = self.corpus.read_file(text)
            if not result.get("success"):
                return _say(f"I couldn't open `{text}`: {result.get('error')}", next_step="wait")
            text = result["content"].strip()
        if not text:
            return _say(ASK_FOR_A_JOB, next_step="wait")
        return {"jd_input": text, "next_step": "continue"}

    # ---- Extract Requirements -------------------------------------------------- #
    def extract_requirements(self, state: ScreeningState) -> dict:
        _progress("Reading the job description for must-haves and nice-to-haves")
        reqs = claude_tasks.extract_requirements(self.claude, state["jd_input"])
        if not weighted_requirements(reqs):
            return _say(f"I didn't find a job to screen for in that. I rank all {self.corpus.size} resumes against "
                        f"the job you describe, judge the top {ROUND1_SIZE} against each requirement, and then answer "
                        f"questions about them.\n\n{ASK_FOR_A_JOB}", next_step="wait")
        # A new job: the previous shortlist is about something else, so nothing carries over
        # except the verdict cache, which is keyed by requirement text and stays valid.
        return {"job_description": state["jd_input"], "requirements": reqs, "previous_requirements": None,
                "screened": [], "shortlist": [], "previous_shortlist": [], "ranking_changes": [],
                "screening_round": 0, "next_step": "continue"}

    # ---- Search Resumes (Round 1) ---------------------------------------------- #
    def search_resumes(self, state: ScreeningState) -> dict:
        reqs = state["requirements"]
        _progress(f"Round 1: ranking all {self.corpus.size} resumes with the Milestone 2 hybrid RAG search")
        ranking = self.corpus.search(search_query(state["job_description"], reqs), k=self.corpus.size,
                                     must_haves=reqs["must_have"])
        top = ranking[:ROUND1_SIZE]
        _progress(f"Round 1: top {ROUND1_SIZE} - " + ", ".join(c["name"] for c in top))
        # On a re-screen, keep the previous shortlist in play: a change in the search query
        # alone must not drop someone before Round 2 has judged them on the new requirements.
        kept = {e["candidate_id"] for e in state.get("shortlist", [])}
        carried = [c for c in ranking[ROUND1_SIZE:] if c["candidate_id"] in kept]
        if carried:
            _progress("Round 1: also carrying forward from the last shortlist - "
                      + ", ".join(c["name"] for c in carried))
        return {"round1": top + carried}

    # ---- Rank Candidates (Round 2) --------------------------------------------- #
    def rank_candidates(self, state: ScreeningState) -> dict:
        reqs = state["requirements"]
        texts = [text for text, _, _ in weighted_requirements(reqs)]
        resumes = {c["candidate_id"]: (c["name"], self.corpus.resume_text(c["candidate_id"])) for c in state["round1"]}
        _progress(f"Round 2: Claude judges {len(resumes)} resumes against {len(texts)} requirements "
                  "(only judgements not already made)")
        cached = state.get("verdicts", {})
        needed = {cid for cid in resumes if any(requirement_key(t) not in cached.get(cid, {}) for t in texts)}
        verdicts, failures = claude_tasks.judge_candidates(
            self.claude, texts, resumes, cached, self.today,
            on_done=lambda cid, n: _progress(f"  judged {resumes[cid][0]} on {n} requirement(s)"))
        # Someone Claude could not judge is left out of this round rather than guessed at;
        # nothing is cached for them, so the next re-screen tries again.
        if needed and set(failures) == needed:
            # Nobody could be judged: an outage or a bad key, not one awkward resume. Raise, so
            # the chat offers a retry and the graph resumes here, instead of reporting an empty round.
            raise LLMError(next(iter(failures.values())))
        unjudged = [{"candidate_id": cid, "name": resumes[cid][0], "reason": reason} for cid, reason in failures.items()]
        for entry in unjudged:
            _progress(f"  could not judge {entry['name']}: {entry['reason']}")
        screened = rank_by_fit([c for c in state["round1"] if c["candidate_id"] not in failures], reqs, verdicts)
        return {"verdicts": verdicts, "unjudged": unjudged, "screened": screened,
                "shortlist": screened[:SHORTLIST_SIZE], "previous_shortlist": state.get("shortlist", []),
                "screening_round": state.get("screening_round", 0) + 1}

    # ---- Generate Report (Round 3) --------------------------------------------- #
    def generate_report(self, state: ScreeningState) -> dict:
        reqs, shortlist, verdicts = state["requirements"], state["shortlist"], state["verdicts"]
        suggestions = self._suggest_for_borderline(reqs, shortlist, verdicts, state.get("suggestions", {}))
        changes = (ranking_changes(state["previous_shortlist"], state["screened"], state.get("previous_requirements"),
                                   reqs, verdicts, SHORTLIST_SIZE) if state.get("previous_shortlist") else [])
        was_listed = {e["candidate_id"]: e for e in state.get("previous_shortlist", [])}
        changes += [{"candidate_id": u["candidate_id"], "name": u["name"], "new_rank": None, "ranked_at": None,
                     "old_rank": was_listed[u["candidate_id"]]["rank"],
                     "reasons": [f"could not be judged this round: {u['reason']}"]}
                    for u in state.get("unjudged", []) if u["candidate_id"] in was_listed]
        round_no = state["screening_round"]
        keys = {e["candidate_id"]: suggestion_key(e["candidate_id"], reqs) for e in shortlist}
        markdown = render_report(
            requirements=reqs, shortlist=shortlist, verdicts_by_cid=verdicts, changes=changes,
            suggestions={cid: suggestions[key] for cid, key in keys.items() if key in suggestions},
            screening_round=round_no, corpus_size=self.corpus.size, today=self.today,
            also_judged=state["screened"][SHORTLIST_SIZE:])
        path = self._save_report(reqs, shortlist, verdicts, markdown, round_no)
        summary = _round_summary(state, changes, self.corpus.size, path)
        return {"messages": [{"role": "assistant", "content": summary}], "ranking_changes": changes,
                "suggestions": suggestions, "report_path": path, "previous_requirements": reqs}

    def _save_report(self, reqs: dict, shortlist: list[dict], verdicts: dict, markdown: str, round_no: int) -> str:
        """Save the round as Markdown for people and JSON for scripts/human_eval.py; return the .md path.

        Reports are never overwritten: a second job with the same title on the same day, or
        a second session, gets the next free name (-2, -3, ...).
        """
        base = f"reports/{_slug(reqs.get('role_title'))}-{self.today.isoformat()}-round-{round_no}"
        stem, n = base, 1
        while self.corpus.exists(f"{stem}.md") or self.corpus.exists(f"{stem}.json"):
            n += 1
            stem = f"{base}-{n}"
        record = {"screening_round": round_no, "date": self.today.isoformat(), "requirements": reqs,
                  "shortlist": [{**{k: e[k] for k in REPORT_FIELDS},
                                 "verdicts": list(current_verdicts(reqs, verdicts[e["candidate_id"]]).values())}
                                for e in shortlist]}
        saved = self.corpus.write_file(f"{stem}.md", markdown)
        if not saved.get("success"):
            _progress(f"Could not save the report: {saved.get('error')}")
            return ""
        saved = self.corpus.write_file(f"{stem}.json", json.dumps(record, indent=2, ensure_ascii=False))
        if not saved.get("success"):
            _progress(f"Saved {stem}.md, but not its JSON copy: {saved.get('error')}")
        return f"{stem}.md"

    def _suggest_for_borderline(self, reqs: dict, shortlist: list[dict], verdicts: dict, cached: dict) -> dict:
        """Round 3 suggestions for borderline candidates. Best effort: a failed call skips that
        candidate's suggestions instead of losing the report."""
        if not reqs.get("must_have"):  # without must-haves everyone is borderline; suggestions would be noise
            return cached
        todo = [e for e in shortlist
                if e["recommendation"] == "borderline" and suggestion_key(e["candidate_id"], reqs) not in cached]
        if not todo:
            return cached
        _progress(f"Round 3: improvement suggestions for {len(todo)} borderline candidate(s)")
        # Read the resumes here, not in the threads: the file tools are not thread-safe.
        texts = {e["candidate_id"]: self.corpus.resume_text(e["candidate_id"]) for e in todo}

        def work(entry: dict) -> tuple[str, list[str] | None]:
            cid = entry["candidate_id"]
            try:
                return suggestion_key(cid, reqs), claude_tasks.suggest_improvements(
                    self.claude, entry["name"], current_verdicts(reqs, verdicts[cid]), texts[cid])
            except LLMError as exc:
                _progress(f"  no suggestions for {entry['name']}: {exc}")
                return suggestion_key(cid, reqs), None

        with ThreadPoolExecutor(max_workers=claude_tasks.PARALLEL_CALLS) as pool:
            made = {key: tips for key, tips in pool.map(work, todo) if tips is not None}
        return {**cached, **made}

    # ---- Human Feedback Loop --------------------------------------------------- #
    def human_feedback(self, state: ScreeningState) -> dict:
        """Pause until the user replies. The chat interface resumes the graph with their message."""
        reply = str(interrupt("waiting for the user")).strip()
        if reply.lower() in QUIT_WORDS:
            return {"next_step": "end"}
        if not reply:
            return {"next_step": "wait"}
        if len(reply) > MAX_MESSAGE_CHARS:  # kept out of the history, where it would ride along on every call
            return _say(f"That message is {len(reply):,} characters; I can take up to {MAX_MESSAGE_CHARS:,}. "
                        "Save a long job description to a file in job_descriptions/ and give me its path.",
                        next_step="wait")
        return {"messages": [{"role": "user", "content": reply}], "next_step": "agent"}

    def agent_turn(self, state: ScreeningState) -> dict:
        """Claude answers the user's message, calling tools as it needs them."""
        toolbox = Toolbox(self.corpus, self.claude, state, self.today)
        messages = _history_for_claude(state["messages"], state_snapshot(state))
        tools = tools_for(self.corpus)  # the file tools' schemas come from the MCP server's tools/list
        reply = ""
        for _ in range(MAX_TOOL_ROUNDS):
            response = self.claude.converse(system=SYSTEM_PROMPT, messages=messages, tools=tools)
            # Append the whole response, thinking blocks included: the API needs them back
            # unchanged for the rest of this tool loop.
            messages.append({"role": "assistant", "content": response.content})
            reply = "\n".join(b.text for b in response.content if b.type == "text").strip() or reply
            calls = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not calls:
                break
            results = []
            for call in calls:
                _progress(f"tool: {call.name}({brief_args(call.input)})")
                results.append(toolbox.run(call.name, call.input, call.id))
            messages.append({"role": "user", "content": results})  # all results in one message
        else:
            reply = (reply + "\n\n" if reply else "") + (
                f"I stopped after {MAX_TOOL_ROUNDS} rounds of tool calls without finishing. "
                "Try asking for one thing at a time.")
        return {**toolbox.updates, "messages": [{"role": "assistant", "content": reply or "(no reply)"}],
                "next_step": toolbox.next_step or "wait", "tool_trace": toolbox.trace}


# ---- routing: each function reads next_step, set by the node before it -------- #
def route_after_parse(state: ScreeningState) -> str:
    return "extract_requirements" if state.get("next_step") == "continue" else "human_feedback"


def route_after_requirements(state: ScreeningState) -> str:
    return "search_resumes" if state.get("next_step") == "continue" else "human_feedback"


def route_after_feedback(state: ScreeningState) -> str:
    return {"end": END, "wait": "human_feedback"}.get(state.get("next_step"), "agent_turn")


def route_after_agent(state: ScreeningState) -> str:
    return {"rerank": "search_resumes", "new_jd": "parse_jd"}.get(state.get("next_step"), "human_feedback")


# ---- helpers ------------------------------------------------------------------- #
def _progress(text: str) -> None:
    """Stream a progress line to the chat interface (a no-op when nobody is streaming)."""
    get_stream_writer()({"progress": text})


def _say(text: str, **updates) -> dict:
    return {"messages": [{"role": "assistant", "content": text}], **updates}


def search_query(job_description: str, reqs: dict) -> str:
    """The Round 1 query: the job text plus the current requirements, which may have been refined since."""
    lines = [job_description, "", "Requirements", *[f"- {r}" for r in reqs["must_have"]]]
    if reqs["nice_to_have"]:
        lines += ["", "Nice to have", *[f"- {r}" for r in reqs["nice_to_have"]]]
    return "\n".join(lines)


def _slug(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "role").lower()).strip("-")[:60] or "role"


def _history_for_claude(messages: list[dict], snapshot: str) -> list[dict]:
    """The conversation as Claude sees it: earlier turns as plain text, the newest user
    message with the current screening state attached.

    Earlier turns' thinking blocks are not replayed. That is allowed (the API accepts
    thinking blocks dropped from the front of a conversation) and it means the snapshot,
    which changes every turn, never has to be edited into old messages. Only the latest
    MAX_HISTORY_MESSAGES are sent; the snapshot carries the state that matters.
    """
    history = [dict(m) for m in messages[-MAX_HISTORY_MESSAGES:]]
    while history and history[0]["role"] != "user":  # a conversation must open with the user
        history.pop(0)
    latest = history[-1]["content"]
    history[-1] = {"role": "user", "content": [{"type": "text", "text": snapshot}, {"type": "text", "text": latest}]}
    return history


def _round_summary(state: ScreeningState, changes: list[dict], corpus_size: int, report_path: str) -> str:
    reqs, shortlist = state["requirements"], state["shortlist"]
    counts = Counter(e["recommendation"] for e in shortlist)
    lines = [f"**Screening round {state['screening_round']}: {reqs.get('role_title')}**", "",
             f"Round 1 ranked {corpus_size} resumes and kept {len(shortlist)}. Round 2 judged them against "
             f"{len(weighted_requirements(reqs))} requirements. Round 3: {counts['hire']} hire, "
             f"{counts['borderline']} borderline, {counts['no_hire']} no hire."]
    if state["screening_round"] == 1:
        lines += ["", "**Requirements**", *[f"- {t} ({k}-have)" for t, k, _ in weighted_requirements(reqs)]]
        if interview_checks(reqs):
            lines += ["", "**For the interview** (a resume cannot show these, so they are not screened)",
                      *[f"- {t}" for t in interview_checks(reqs)]]
        if reqs.get("notes"):
            lines += ["", f"_How I read the request: {reqs['notes']}_"]
    if changes:
        lines += ["", "**What changed**", *[f"- {describe_change(c)}" for c in changes]]
    lines += ["", "**Top candidates**",
              *[f"{e['rank']}. {e['name']} `{e['candidate_id']}` - {LABEL[e['recommendation']]}, fit "
                f"{e['fit_score']}. {e['recommendation_reason']}" for e in shortlist[:SUMMARY_TOP_N]]]
    if state.get("unjudged"):
        lines += ["", "Could not be judged this round (ask me to re-screen to try again): "
                  + "; ".join(f"{e['name']} ({e['reason']})" for e in state["unjudged"])]
    if report_path:
        lines += ["", f"Full report with the evidence for all {len(shortlist)}: `{report_path}`"]
    return "\n".join(lines)


def write_diagram(path: Path = DIAGRAM_FILE) -> Path:
    """Write the state machine, drawn from the compiled graph itself, as a Mermaid diagram."""
    mermaid = MatchingAgent().build().get_graph().draw_mermaid()
    path.parent.mkdir(exist_ok=True)
    path.write_text("# State machine\n\nGenerated from the compiled LangGraph graph by "
                    "`python matching_agent.py --diagram`; do not edit by hand. Dashed edges are "
                    "conditional: the node's result decides which one is taken.\n\n"
                    f"```mermaid\n{mermaid}```\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="The LangGraph matching agent. Chat with it via chat.py.")
    parser.add_argument("--diagram", action="store_true", help=f"write the state machine diagram to {DIAGRAM_FILE}")
    args = parser.parse_args()
    if args.diagram:
        print(f"wrote {write_diagram()}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
