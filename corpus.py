"""
The resume corpus, seen through the earlier milestones.

* Milestone 1's file tools - read_file, list_files, search_in_file, write_file - are now
  reached only through the filesystem MCP server (filesystem_mcp_server.py), which this
  class starts as a child process and talks to over stdio with mcp_client.MCPClient. The
  server owns the sandbox and the folder policy: read resumes/, job_descriptions/ and
  reports/; write only new files into reports/.
* Milestone 2's hybrid RAG search (JobMatcher) ranks every resume against a query using
  embeddings in ChromaDB, BM25 and exact skill matching. It is unchanged.

The four file methods keep Milestone 3's return shapes ({"success", "content", "error"} and
a list of entries), so the agent code that calls them did not change.

Round 1 can use any embedding provider in embedders.py. EMBEDDING_PROVIDER in .env picks
which index is opened (chroma_db for Milestone 2's MiniLM, chroma_db_<provider> for the
others); the index itself records the model it was built with.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
MILESTONE2_DIR = PROJECT_ROOT / "milestone2_rag"
ROUND1_SIZE = 10

# The vendored modules import each other by top-level name (`from resume_rag import ...`),
# so their folders go on sys.path instead of being edited into a package.
sys.path.insert(0, str(MILESTONE2_DIR))
import job_matcher  # noqa: E402
import resume_rag  # noqa: E402
from job_matcher import JobMatcher, parse_requirement  # noqa: E402
from resume_rag import extract_skills  # noqa: E402

import embedders  # noqa: E402
from mcp_client import MCPClient, MCPClientError  # noqa: E402

embedders.install(resume_rag, job_matcher)  # Milestone 2 now builds its embedder through embedders.py
MILESTONE2_BAND = (job_matcher.SIM_FLOOR, job_matcher.SIM_CEIL)  # hand-set for MiniLM in Milestone 2
# The tools Toolbox wraps itself (to mark file text as data); every other server tool is passed through.
MILESTONE1_TOOLS = ("list_files", "read_file", "search_in_file", "write_file")


class CorpusError(Exception):
    """The resume index is missing, or a resume could not be read."""


def index_dir(provider: str, root: Path = PROJECT_ROOT) -> Path:
    """Where the index for an embedding provider lives; MiniLM keeps Milestone 2's folder."""
    if provider not in embedders.PROVIDERS:
        raise CorpusError(f"Unknown EMBEDDING_PROVIDER {provider!r}; use one of {', '.join(embedders.PROVIDERS)}.")
    return Path(root) / ("chroma_db" if provider == "huggingface" else f"chroma_db_{provider}")


def provider_from_env() -> str:
    return (os.environ.get("EMBEDDING_PROVIDER") or "huggingface").strip().lower()


class Corpus:
    def __init__(self, root: Path = PROJECT_ROOT, db_path: Path | None = None, mcp: MCPClient | None = None,
                 on_mcp_call: Callable[[str, str, dict], None] | None = None, extra=None):
        self.root = Path(root)
        self._mcp, self._on_mcp_call = mcp, on_mcp_call
        self.extra = extra  # an mcp_hub.MCPHub of other MCP servers (e.g. fetch), tools named server__tool
        self._mcp_lock = threading.Lock()
        self.db_path = Path(db_path) if db_path else index_dir(provider_from_env(), self.root)
        registry_file = self.db_path / "candidates.json"
        if not registry_file.is_file():
            raise CorpusError(f"No resume index at {self.db_path}. Build it first:\n"
                              f"  python scripts/build_index.py --provider {provider_from_env()}")
        try:
            entries = json.loads(registry_file.read_text())["candidates"]
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise CorpusError(f"The resume index at {registry_file} is unreadable ({exc}); rebuild it.") from exc
        # Candidate ids are resume file names without the extension ("grace_liu"):
        # readable in a chat, and unlike names (two people here share one) unique.
        stems = Counter(Path(c["resume_path"]).stem for c in entries)
        if clashes := sorted(stem for stem, n in stems.items() if n > 1):
            raise CorpusError(f"Several resumes share a file name, so their ids would collide: {clashes}")
        self.candidates = {Path(c["resume_path"]).stem: c for c in entries}
        self.band = self._similarity_band()
        self._matcher = None

    def _similarity_band(self) -> tuple[float, float]:
        """The index's measured similarity band (embedders.calibrate), else Milestone 2's."""
        calibration = self.db_path / embedders.CALIBRATION_FILE
        if not calibration.is_file():
            return MILESTONE2_BAND
        try:
            band = json.loads(calibration.read_text())
            return float(band["floor"]), float(band["ceil"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
            raise CorpusError(f"{calibration} is unreadable ({exc}); rebuild the index.") from exc

    # ---- who is in the corpus -------------------------------------------- #
    @property
    def size(self) -> int:
        return len(self.candidates)

    def name(self, candidate_id: str) -> str:
        return self.candidates[candidate_id]["candidate_name"]

    def find(self, name_or_id: str) -> list[str]:
        """Candidate ids matching an id, a full name, or a unique part of a name."""
        wanted = name_or_id.strip().lower()
        if wanted in self.candidates:
            return [wanted]
        exact = [cid for cid, c in self.candidates.items() if c["candidate_name"].lower() == wanted]
        return exact or [cid for cid, c in self.candidates.items() if wanted in c["candidate_name"].lower()]

    def resume_text(self, candidate_id: str) -> str:
        result = self.read_file(self.candidates[candidate_id]["resume_path"])
        if not result.get("success"):
            raise CorpusError(f"Could not read the resume for {candidate_id}: {result.get('error')}")
        return result["content"]

    # ---- Milestone 2: RAG search (Round 1) --------------------------------- #
    @property
    def matcher(self) -> JobMatcher:
        if self._matcher is None:  # loading the embedding model takes a few seconds, so only once
            try:
                self._matcher = JobMatcher(self.db_path)
            except embedders.EmbeddingError as exc:
                raise CorpusError(str(exc)) from exc
        return self._matcher

    def search(self, query: str, k: int = ROUND1_SIZE, must_haves: list[str] = ()) -> list[dict]:
        """Rank every resume against `query` and return the top `k`.

        Must-haves are passed to Milestone 2 only to annotate each candidate with its rule
        checks; nobody is filtered out, because those checks rest on a fixed skill
        vocabulary and date arithmetic that Round 2 exists to improve on.
        """
        checkable = [m for m in must_haves if parse_requirement(m)]
        critical = list(dict.fromkeys(s for m in must_haves for s in extract_skills(m))) or None
        matcher = self.matcher
        # Milestone 2 reads its similarity band from module globals; set this index's band
        # before every search, since another index (another model) may have set its own.
        job_matcher.SIM_FLOOR, job_matcher.SIM_CEIL = self.band
        try:
            result = matcher.match(query, k=k, must_haves=checkable, auto_requirements=False,
                                   critical_skills=critical, apply_filters=False)
        except embedders.EmbeddingError as exc:
            raise CorpusError(str(exc)) from exc
        return [{"candidate_id": Path(m["resume_path"]).stem, "name": m["candidate_name"],
                 "resume_path": m["resume_path"], "rag_score": m["match_score"],
                 "experience_years": m["experience_years"], "matched_skills": m["matched_skills"],
                 "rule_check_failures": m["failed_requirements"]} for m in result["top_matches"]]

    # ---- Milestone 1 file tools, through the MCP server ---------------------- #
    @property
    def mcp(self) -> MCPClient:
        """The filesystem MCP server's client, started on first use (Round 2 calls from threads)."""
        with self._mcp_lock:
            if self._mcp is None:
                try:
                    self._mcp = MCPClient.filesystem(self.root, on_call=self._on_mcp_call)
                except MCPClientError as exc:
                    raise CorpusError(str(exc)) from exc
            return self._mcp

    def _call(self, tool: str, args: dict) -> tuple[bool, dict]:
        try:
            return self.mcp.call_tool(tool, args)
        except MCPClientError as exc:
            raise CorpusError(str(exc)) from exc

    @staticmethod
    def _as_m1(failed: bool, data: dict, filepath: str) -> dict:
        """Milestone 1's result shape, which the agent code expects, from an MCP tool result."""
        if not failed:
            return data
        return {"success": False, "filepath": data.get("filepath") or filepath, "error_code": data.get("error_code"),
                "error": data.get("message") or data.get("text") or "the MCP server reported an error"}

    def read_file(self, filepath: str) -> dict:
        return self._as_m1(*self._call("read_file", {"filepath": filepath}), filepath)

    def list_files(self, directory: str, extension: str | None = None) -> list[dict]:
        failed, data = self._call("list_files", {"directory": directory, "extension": extension or None})
        return [{"error": self._as_m1(failed, data, directory)["error"]}] if failed else data["files"]

    def search_in_file(self, filepath: str, keyword: str, whole_word: bool = False) -> dict:
        return self._as_m1(*self._call("search_in_file", {"filepath": filepath, "keyword": keyword,
                                                          "whole_word": whole_word}), filepath)

    def write_file(self, filepath: str, content: str) -> dict:
        """Save a new file into reports/. The server never replaces an existing file, so neither a
        tool call nor a later session can rewrite a screening report after the fact."""
        return self._as_m1(*self._call("write_file", {"filepath": filepath, "content": content}), filepath)

    def exists(self, filepath: str) -> bool:
        parent = str(PurePosixPath(str(filepath)).parent)
        entries = self.list_files(parent) if parent not in (".", "") else []
        return any(e.get("path") == filepath for e in entries)

    # ---- every other tool the MCP server offers ----------------------------- #
    def mcp_tools(self) -> list[dict]:
        """The filesystem server's tools (from tools/list), plus any other servers' as server__tool,
        all as Anthropic tool schemas."""
        try:
            return self.mcp.list_tools() + (self.extra.tools() if self.extra else [])
        except MCPClientError as exc:
            raise CorpusError(str(exc)) from exc

    def call_mcp_tool(self, name: str, args: dict) -> tuple[bool, dict]:
        if self.extra is not None and self.extra.owns(name):
            try:
                return self.extra.call(name, args)
            except MCPClientError as exc:
                raise CorpusError(str(exc)) from exc
        return self._call(name, args)

    def fetch_url(self, url: str) -> str | None:
        """A web page's text through the fetch MCP server, or None when no fetch server is connected."""
        if self.extra is None or not self.extra.owns("fetch__fetch"):
            return None
        failed, data = self.call_mcp_tool("fetch__fetch", {"url": url, "max_length": 20000})
        if failed:
            raise CorpusError(f"Could not fetch {url}: {data.get('text') or data.get('message') or data}")
        return data.get("text") or data.get("content") or json.dumps(data)

    def close(self) -> None:
        if self._mcp is not None:
            self._mcp.close()
        if self.extra is not None:
            self.extra.close()
