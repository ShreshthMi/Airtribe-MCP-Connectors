"""The agent reaches files only through MCP: none of its modules imports the Milestone 1 tools."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENT_MODULES = ("matching_agent.py", "agent_tools.py", "corpus.py", "chat.py")
FORBIDDEN = {"fs_tools", "resume_loader", "extractors"}  # Milestone 1's tools and Milestone 2's loader over them


def imported_modules(source: str) -> set[str]:
    """Top-level names of every module imported anywhere in `source` (functions included)."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) in (
                "__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
            names.add(str(node.args[0].value).split(".")[0])
    return names


@pytest.mark.parametrize("module", AGENT_MODULES)
def test_agent_modules_do_not_import_the_file_tools(module):
    assert not imported_modules((ROOT / module).read_text()) & FORBIDDEN


@pytest.mark.parametrize("source", [
    "import fs_tools",
    "from resume_loader import sandbox_root",
    "def later():\n    import extractors",
    "import importlib\nmod = importlib.import_module('fs_tools')",
    "fs = __import__('fs_tools')",
])
def test_the_check_catches_each_way_of_importing_them(source):
    """The test above would fail on any of these; this proves it can."""
    assert imported_modules(source) & FORBIDDEN


def test_the_corpus_talks_to_the_mcp_client_instead():
    assert "mcp_client" in imported_modules((ROOT / "corpus.py").read_text())
    assert "fs_tools" in imported_modules((ROOT / "filesystem_mcp_server.py").read_text())
