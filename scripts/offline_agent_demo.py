"""
Run chat.py with a rule-based stand-in for Claude, for showing the agent <-> MCP plumbing
when no Anthropic API key is available.

Everything except the model is real: the LangGraph graph, the filesystem MCP server process
(and the fetch server with --mcp-servers), Round 1's ChromaDB + BM25 search, the reports.
The stand-in is NOT Claude and its judgements are crude keyword matches: a requirement is
"met" when every keyword in it appears in the resume, quoting the first line that has one.
It answers the chat messages of demo/mcp_demo_script.txt with fixed tool plans (search a
resume over MCP, batch_process, write_file). Use it to demonstrate the integration, never
to judge candidates.

    python scripts/offline_agent_demo.py --show-mcp --mcp-servers --script demo/mcp_demo_script.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anthropic.types.beta import BetaMessage  # noqa: E402

import chat  # noqa: E402
from claude_tasks import (CandidateAssessment, ExtractedRequirements, InterviewQuestion,  # noqa: E402
                          InterviewQuestions, RequirementVerdict, Suggestions)
from matching_agent import MatchingAgent  # noqa: E402

STOPWORDS = {"years", "year", "experience", "of", "and", "in", "with", "on", "the", "a", "an", "or", "for", "to",
             "development", "design", "system", "strong", "good", "plus"}


def keywords(requirement: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*", requirement)
    return [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]


def message(*parts) -> BetaMessage:
    content, tool = [], False
    for i, part in enumerate(parts):
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        else:
            tool = True
            content.append({"type": "tool_use", "id": f"toolu_offline_{i}", "name": part[0], "input": part[1]})
    return BetaMessage.model_validate({"id": "msg_offline", "type": "message", "role": "assistant",
                                       "model": "rule-based-stand-in", "content": content,
                                       "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None,
                                       "usage": {"input_tokens": 0, "output_tokens": 0}})


class RuleBasedStandIn:
    """Same two methods as llm.Claude. Deterministic, offline, and deliberately simple."""

    model = "rule-based stand-in (not Claude)"

    def __init__(self):
        self.agent: MatchingAgent | None = None
        self.usage = {"calls": 0, "input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}

    # ---- structured(): the graph's own model calls ------------------------------------ #
    def structured(self, *, system: str, prompt: str, schema):
        self.usage["calls"] += 1
        if schema is ExtractedRequirements:
            return self._requirements(re.search(r"<request>\n(.*)\n</request>", prompt, re.S).group(1))
        if schema is CandidateAssessment:
            return self._judge(prompt)
        if schema is Suggestions:
            return Suggestions(suggestions=["Ask for a recent project that shows each partly met requirement.",
                                            "Check the years of hands-on use in a short technical screen."])
        if schema is InterviewQuestions:
            count = int(re.search(r"Write (\d+) questions", prompt).group(1))
            return InterviewQuestions(questions=[InterviewQuestion(
                question=f"Walk me through a project where you used this skill ({i}).", requirement="see report",
                why="Keyword evidence only.") for i in range(1, count + 1)])
        raise ValueError(f"the stand-in does not handle {schema.__name__}")

    def _requirements(self, text: str) -> ExtractedRequirements:
        lines = [line.strip() for line in text.splitlines()]
        title = next((line for line in lines if line and not line.lower().startswith(("contents of", "content type"))),
                     "Role")
        must, nice, section = [], [], None
        for line in lines:
            low = line.lower()
            if low.startswith(("must have", "requirements", "required")):
                section = must
            elif low.startswith(("preferred", "nice to have", "bonus")):
                section = nice
            elif line.startswith(("- ", "* ")) and section is not None:
                section.append(line[2:].strip())
        return ExtractedRequirements(role_title=title.lstrip("# "), must_have=must, nice_to_have=nice,
                                     interview_only=[], notes="Extracted by a rule-based stand-in, not Claude.")

    def _judge(self, prompt: str) -> CandidateAssessment:
        block = re.search(r"<requirements>\n(.*?)\n</requirements>", prompt, re.S).group(1)
        requirements = [re.sub(r"^\d+\. ", "", line) for line in block.splitlines()]
        resume = re.search(r"<resume>\n(.*)\n</resume>", prompt, re.S).group(1)
        lines = [line.strip(" -*•\t") for line in resume.splitlines() if len(line.split()) >= 3]
        verdicts = []
        for number, requirement in enumerate(requirements, start=1):
            words = keywords(requirement)
            found = [w for w in words if re.search(rf"(?<![A-Za-z]){re.escape(w)}(?![A-Za-z])", resume, re.I)]
            verdict = "met" if words and len(found) == len(words) else "partial" if found else "missing"
            evidence = next((line for line in lines if found and re.search(re.escape(found[0]), line, re.I)), "")
            verdicts.append(RequirementVerdict(
                requirement_number=number, verdict=verdict, evidence=" ".join(evidence.split()[:30]),
                reasoning=f"Keyword match: {len(found)} of {len(words)} keywords found (rule-based stand-in)."))
        return CandidateAssessment(verdicts=verdicts)

    # ---- converse(): the chat turn, with fixed tool plans ---------------------------- #
    def converse(self, *, system: str, messages: list[dict], tools: list[dict]) -> BetaMessage:
        self.usage["calls"] += 1
        last = messages[-1]["content"]
        if isinstance(last, list) and last and isinstance(last[0], dict) and last[0].get("type") == "tool_result":
            return message(self._summarise(last))
        text = " ".join(b["text"] for b in last if b.get("type") == "text") if isinstance(last, list) else str(last)
        request = text.rsplit("</screening_state>", 1)[-1].lower()
        top = re.search(r"1\. (.+?) `([a-z0-9_]+)`", text)
        if "kafka" in request and top:
            path = self.agent.corpus.candidates[top.group(2)]["resume_path"]
            return message(f"Searching {top.group(1)}'s resume over MCP.",
                           ("search_in_file", {"filepath": path, "keyword": "Kafka"}),
                           ("search_in_file", {"filepath": path, "keyword": "Redis"}))
        if "batch_process" in request or "how many" in request:
            keyword = next((w for w in ("Kubernetes", "Terraform", "Kafka", "React") if w.lower() in request), "Python")
            return message(("batch_process", {"directory": "resumes", "operation": "search", "keyword": keyword}))
        if "save" in request:
            names = re.findall(r"\d+\. (.+?) `", text)[:3]
            role = re.search(r"Role: (.+?) \(", text)
            summary = (f"# Shortlist summary{': ' + role.group(1) if role else ''}\n\n"
                       f"Top three by fit: {', '.join(names)}.\n"
                       "Judged by a rule-based stand-in (keyword matches), not Claude: verify before use.\n")
            return message(("write_file", {"filepath": "reports/backend-shortlist-summary-offline.md",
                                           "content": summary}))
        return message("The rule-based stand-in only answers the demo script's questions.")

    def _summarise(self, results: list[dict]) -> str:
        lines = []
        for result in results:
            content = result.get("content", "")
            if result.get("is_error"):
                lines.append(f"The server refused: {content}")
                continue
            body = content.split("\n", 1)[1].rsplit("\n", 1)[0] if "\n" in content else content
            try:
                data = json.loads(body)
            except ValueError:
                lines.append(content)
                continue
            if "match_count" in data:
                lines.append(f"`{data['keyword']}`: {data['match_count']} match(es) in {data['filepath']}.")
            elif "results" in data:
                hits = [r for r in data["results"] if r.get("match_count")]
                lines.append(f"{len(hits)} of {data['total']} resumes mention it (batch_process, "
                             f"{data['elapsed_ms']:.0f} ms on the server).")
            else:
                lines.append(content[:200])
        return "\n".join(f"- {line}" for line in lines)


def main() -> None:
    stand_in = RuleBasedStandIn()

    def build() -> MatchingAgent:
        agent = MatchingAgent(claude=stand_in)
        stand_in.agent = agent
        return agent

    chat.MatchingAgent = build
    chat.console.print("[bold yellow]Offline run: the model is a rule-based stand-in, not Claude "
                       "(no API key). Everything else - MCP servers, search, graph, reports - is real.[/bold yellow]")
    chat.main()


if __name__ == "__main__":
    main()
