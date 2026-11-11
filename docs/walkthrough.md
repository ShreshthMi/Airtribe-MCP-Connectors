# Walkthrough: one request, end to end

Milestone 1's review asked for a short trace of a user request from the tool schema through
the file operations to the model's final answer. This is that trace for the agent, in two
parts: a question answered with a tool, and a change of requirements that re-runs the
screening. File and function names are the real ones; the data is from a live run of
Milestone 3. Since Milestone 4 every file operation goes through the filesystem MCP server;
[mcp_architecture.md](mcp_architecture.md) shows those messages.

## Before the conversation: the three screening rounds

`python chat.py`, and the user types *Find me candidates with React and 3+ years
experience*. `chat.py` starts the graph with that text as `jd_input`, and LangGraph runs
the nodes in `matching_agent.py` in order:

| Node | What happens | Where |
|---|---|---|
| `parse_jd` | Text, not a file path, so it is used as it is | `matching_agent.py` |
| `extract_requirements` | Claude returns `{"must_have": ["3+ years of React"], ...}` validated by a Pydantic schema | `claude_tasks.extract_requirements` |
| `search_resumes` | **Round 1.** Milestone 2's `JobMatcher.match()` ranks all 100 resumes; the top 10 go on | `corpus.Corpus.search` |
| `rank_candidates` | **Round 2.** Each resume is read from the filesystem MCP server as a resource (`resources/read resume://<file>`, Milestone 1's `read_file` underneath); Claude judges it against each requirement in parallel; quotes are checked against the text | `corpus.Corpus.resume_text`, `claude_tasks.judge_candidates`, `screening.evidence_found` |
| `generate_report` | **Round 3.** Hire / borderline / no hire; suggestions for borderline candidates; the report is saved with `write_file` on the MCP server, which never replaces an existing file | `screening.render_report`, `corpus.Corpus.write_file` |
| `human_feedback` | `interrupt()` pauses the graph; the checkpointer (SQLite, opened by `chat.py`) keeps the state | `matching_agent.py`, `chat.open_checkpointer` |

## Part 1: "Compare the top 3 matches side by side"

**1. The chat loop resumes the graph.** `chat.py` reads the line and calls
`app.stream(Command(resume="Compare the top 3 matches side by side"), config)`. Inside
`human_feedback`, `interrupt()` now returns that string; the node appends it to
`messages` and sets `next_step="agent"`, so `route_after_feedback` picks `agent_turn`.

**2. The agent turn builds Claude's view.** `agent_turn` replays the conversation as plain
text and attaches a snapshot of the state to the newest message
(`_history_for_claude`, `agent_tools.state_snapshot`):

```text
<screening_state>
Role: React Developer (screening round 1)
Requirements:
- 3+ years of React (must-have)
Shortlist (fit = weighted share of requirements met; rag = Round 1 search score):
1. Grace Liu `grace_liu`: fit 100, rag 86, Hire, must-haves met 1/1
2. Hana Sato `hana_sato`: fit 100, rag 85, Hire, must-haves met 1/1
...
</screening_state>
Compare the top 3 matches side by side
```

**3. Claude is offered the tools.** `llm.Claude.converse()` sends one Messages API request
with `SYSTEM_PROMPT` and the tool schemas from `agent_tools.tools_for(corpus)`: the eight
local tools, plus every tool the filesystem MCP server lists in `tools/list` (the four
Milestone 1 tools, `batch_process` and the watch tools). The one that matters here:

```json
{"name": "compare_candidates", "strict": true,
 "description": "Side-by-side table of 2-5 candidates against every requirement. ...",
 "input_schema": {"type": "object",
                  "properties": {"candidate_ids": {"type": "array", "items": {"type": "string"}}},
                  "required": ["candidate_ids"], "additionalProperties": false}}
```

**4. Claude asks for a tool.** The response stops with `stop_reason: "tool_use"` and a block:

```json
{"type": "tool_use", "name": "compare_candidates",
 "input": {"candidate_ids": ["grace_liu", "hana_sato", "noah_fischer"]}}
```

**5. The tool runs.** `Toolbox.run()` looks up `tool_compare_candidates`, resolves each id
against the shortlist, and calls `screening.comparison_table()`. No Claude call is needed:
the three were judged in Round 2 and the verdicts are cached in the state. A candidate
off the shortlist would be read from the MCP server and judged on the spot.
The result goes back as:

```json
{"type": "tool_result", "tool_use_id": "toolu_...", "is_error": false,
 "content": "| | Grace Liu (#1) | Hana Sato (#2) | Noah Fischer (#3) |\n|---|---|---|---|\n| Fit score | 100 | 100 | 100 |\n..."}
```

Had the tool failed (an unknown name, say), `is_error` would be `true` and the message
would go to Claude rather than crash the turn, so Claude can recover or explain.

**6. Claude answers.** `agent_turn` appends Claude's response (thinking blocks included, as
the API requires within a tool loop) and the tool results, and calls `converse()` again.
This time `stop_reason` is `"end_turn"`: the text is the reply. The node returns

```python
{"messages": [{"role": "assistant", "content": "Here are the top 3 side by side: ..."}],
 "next_step": "wait", "tool_trace": ['compare_candidates(candidate_ids=["grace_liu", ...])']}
```

LangGraph appends the message (the `operator.add` reducer on `messages`),
`route_after_agent` returns `human_feedback`, the graph pauses again, and `chat.py` prints
the reply in a panel.

## Part 2: "TypeScript is a must-have too. Re-rank."

Steps 1-3 are the same. This time Claude calls

```json
{"name": "update_requirements",
 "input": {"must_have": ["3+ years of React", "TypeScript"], "nice_to_have": ["Next.js"],
           "change_summary": "User added TypeScript as a must-have and Next.js as a nice-to-have"}}
```

`tool_update_requirements` does not touch the graph state directly. It records
`requirements` and `previous_requirements` in `Toolbox.updates` and sets
`next_step="rerank"`. Claude replies in a sentence, `agent_turn` returns those updates,
and `route_after_agent` sends the graph to `search_resumes` instead of back to the user:

1. **Round 1** searches again with the new requirements, and carries the previous
   shortlist forward so nobody drops out because the search query changed.
2. **Round 2** judges only what is new. The verdict cache is keyed by requirement text,
   so candidates already judged on "3+ years of React" are asked about TypeScript and
   Next.js only. Their React verdicts stay exactly as they were, which is what makes the
   ranking change attributable to the user's change.
3. **Round 3** compares the old and new rankings (`screening.ranking_changes`) and the
   reply says who moved and why. From the live run in
   [`scenarios/transcripts/02_refine_requirements.txt`](../scenarios/transcripts/02_refine_requirements.txt):

```text
- Hana Sato: #2 -> #4 (fit 100 -> 90; changed requirements: TypeScript met, Next.js partial)
- Chloe Martin: #6 -> #10 (fit 50 -> 70; changed requirements: TypeScript met, Next.js partial)
```

Then `human_feedback` pauses for the next message. When the user types `quit`, `chat.py`
stops reading but leaves the graph paused in its SQLite checkpoint, so
`python chat.py --session <name>` picks the conversation up later.

## Where to look next

- `screening.py`: every scoring and explanation rule, as plain functions.
- `tests/test_conversation_flows.py`: these flows and others, run through the real graph
  with a scripted Claude.
- `docs/state_machine.md`: the graph, drawn from the compiled LangGraph object.
