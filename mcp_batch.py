"""
batch_process(): run one Milestone 1 operation over many files at once.

* Concurrency is bounded by a semaphore (batch_concurrency). The Milestone 1 calls are
  blocking (PDF and DOCX parsing), so each runs in a worker thread.
* One bad file never fails the batch: every file gets its own {path, ok, error_code, ...}.
* Progress goes to the client after each file (notifications/progress), when it asked.
* The number of files is capped (batch_max_files); a longer list is refused up front.

Operations:
  read    extract the text: per file the length and a short preview (full text: read_file)
  search  count keyword matches per file
  stats   metadata only (size, extension, modified time); nothing is parsed
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable

import anyio

OPERATIONS = ("read", "search", "stats")


async def run_batch(paths: list[str], operation: str, worker: Callable[[str], dict], concurrency: int,
                    progress: Callable[[int, int, str], Awaitable[None]] | None = None) -> dict:
    """Apply `worker` (a blocking function: path -> per-file result dict) to every path."""
    started = time.perf_counter()
    results: list[dict | None] = [None] * len(paths)
    limiter = anyio.Semaphore(concurrency)
    done = 0

    async def one(index: int, path: str) -> None:
        nonlocal done
        async with limiter:
            t0 = time.perf_counter()
            try:
                result = await anyio.to_thread.run_sync(worker, path)
            except Exception as exc:  # noqa: BLE001 - per-file isolation is the point
                result = {"ok": False, "error_code": "READ_FAILED", "message": f"{type(exc).__name__}: {exc}"}
            results[index] = {"path": path, **result, "ms": round((time.perf_counter() - t0) * 1000, 1)}
        done += 1
        if progress is not None:
            await progress(done, len(paths), path)

    async with anyio.create_task_group() as group:
        for i, p in enumerate(paths):
            group.start_soon(one, i, p)

    ok = sum(1 for r in results if r and r["ok"])
    return {"operation": operation, "total": len(paths), "ok": ok, "failed": len(paths) - ok, "skipped": 0,
            "concurrency": concurrency, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            "results": results}
