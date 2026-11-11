"""
Run the conversation scenarios in scenarios/ against the real model and save transcripts.

    python scripts/run_scenarios.py                       # every scenario
    python scripts/run_scenarios.py 02_refine_requirements # just one

Each scenario is replayed through chat.py exactly as a user would type it, and the full
output - progress lines, tool calls, answers and the token usage - is saved to
scenarios/transcripts/<name>.txt. This calls the Anthropic API: each scenario costs what
its screening rounds and chat turns cost (see the README).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "scenarios"
TRANSCRIPTS = SCENARIOS / "transcripts"
WIDTH = 110  # transcript line width


def chat_flags(script: Path) -> list[str]:
    """Extra chat.py flags from a '# flags: ...' line in the scenario, e.g. --show-mcp --mcp-servers."""
    for line in script.read_text().splitlines():
        if line.startswith("# flags:"):
            return line.removeprefix("# flags:").split()
    return []


def run(script: Path) -> tuple[int, float]:
    started = time.monotonic()
    flags = chat_flags(script)
    result = subprocess.run([sys.executable, "chat.py", "--script", str(script), *flags], cwd=ROOT,
                            capture_output=True, text=True,
                            env={**os.environ, "COLUMNS": str(WIDTH), "NO_COLOR": "1"})
    header = (f"# Transcript of {script.relative_to(ROOT)}, run live on {time.strftime('%Y-%m-%d')}\n"
              f"# (chat.py --script {' '.join(flags)}; exit code {result.returncode}, "
              f"{time.monotonic() - started:.0f}s)\n\n")
    (TRANSCRIPTS / f"{script.stem}.txt").write_text(header + result.stdout + result.stderr)
    return result.returncode, time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conversation scenarios against the real model.")
    parser.add_argument("names", nargs="*", help="scenario file names without .txt (default: all)")
    args = parser.parse_args()
    scripts = [SCENARIOS / f"{n}.txt" for n in args.names] if args.names else sorted(SCENARIOS.glob("*.txt"))
    missing = [s.name for s in scripts if not s.is_file()]
    if missing:
        parser.error(f"no such scenario: {', '.join(missing)}")
    TRANSCRIPTS.mkdir(exist_ok=True)
    failures = 0
    for script in scripts:
        code, seconds = run(script)
        failures += code != 0
        print(f"{'ok  ' if code == 0 else 'FAIL'} {script.stem} ({seconds:.0f}s)", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
