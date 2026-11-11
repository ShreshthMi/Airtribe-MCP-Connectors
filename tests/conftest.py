"""Make the project root importable from the tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pytest  # noqa: E402


@pytest.fixture
def anyio_backend():
    """The MCP tests use anyio's pytest plugin (installed with mcp) on asyncio."""
    return "asyncio"


@pytest.fixture
def data_root(tmp_path):
    """A throwaway project folder with the three data folders the MCP server serves."""
    for folder in ("resumes", "job_descriptions", "reports"):
        (tmp_path / folder).mkdir()
    (tmp_path / "resumes" / "ana_ruiz.md").write_text("# Ana Ruiz\nBuilt React apps for 6 years.\nJava, too.\n")
    (tmp_path / "resumes" / "ben_ito.txt").write_text("Ben Ito\nBuilt React Native apps.\n")
    (tmp_path / "job_descriptions" / "fe.txt").write_text("Frontend Engineer\nRequirements\n- React\n")
    (tmp_path / "reports" / "old.md").write_text("an earlier report\n")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-not-a-real-key\n")
    (tmp_path / "resumes" / ".hidden.txt").write_text("hidden\n")
    (tmp_path / "secret.txt").write_text("not for clients\n")
    return tmp_path
