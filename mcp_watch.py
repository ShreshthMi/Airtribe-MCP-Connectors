"""
watch_directory(): tell MCP clients when a new resume lands in a watched folder.

Each watch runs watchfiles.awatch in a background task. A new file is announced once:
  * it is not hidden, not an Office lock file (~$...) and has an allowed extension;
  * its size is the same across two checks, because PDF and DOCX writes are not atomic;
  * it has not been announced before (files present when the watch starts never are).

Every announcement gets an increasing sequence number and goes into a bounded ring
buffer, so clients that ignore notifications (or an LLM) can poll get_watch_events().
Clients that do handle notifications also get notifications/resources/list_changed and
a log message from the server; that part is done by the on_new callback the server passes in.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from watchfiles import Change, awatch

log = logging.getLogger("mcp_fs.watch")
STABLE_CHECK_SECONDS = 0.2


@dataclass(frozen=True)
class WatchEvent:
    seq: int
    watch_id: str
    path: str        # relative to the server root, e.g. "resumes/new_person.pdf"
    event: str       # "added"
    size_bytes: int
    at: str          # ISO 8601, UTC


class WatchManager:
    def __init__(self, root: Path, extensions: tuple[str, ...], debounce_ms: int, force_polling: bool,
                 buffer_size: int, on_new: Callable[[WatchEvent, object], Awaitable[None]]):
        self.root, self.extensions = root, tuple(e.lower() for e in extensions)
        self.debounce_ms, self.force_polling = debounce_ms, force_polling
        self.on_new = on_new
        self.events: deque[WatchEvent] = deque(maxlen=buffer_size)
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self._watches: dict[str, dict] = {}

    # ---- lifecycle ---------------------------------------------------------- #
    def start(self, folder: str, extensions: tuple[str, ...] | None, session) -> dict:
        exts = tuple(e.lower() for e in (extensions or self.extensions))
        path = self.root / folder
        watch_id = f"w{next(self._ids)}"
        seen = {p.name for p in path.iterdir() if p.is_file()} if path.is_dir() else set()
        stop = asyncio.Event()
        watch = {"watch_id": watch_id, "directory": folder, "extensions": list(exts), "stop": stop,
                 "seen": seen, "session": session, "started_at": _now()}
        watch["task"] = asyncio.get_running_loop().create_task(self._run(watch, path))
        self._watches[watch_id] = watch
        log.info("watch_directory %s started on %s/ (%d existing files ignored)", watch_id, folder, len(seen))
        return self.describe(watch_id)

    async def stop(self, watch_id: str) -> bool:
        watch = self._watches.pop(watch_id, None)
        if watch is None:
            return False
        watch["stop"].set()
        watch["task"].cancel()
        try:
            await watch["task"]
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - the watcher is going away either way
            pass
        log.info("watch_directory %s stopped", watch_id)
        return True

    async def stop_all(self) -> None:
        for watch_id in list(self._watches):
            await self.stop(watch_id)

    # ---- queries ------------------------------------------------------------- #
    def exists(self, watch_id: str) -> bool:
        return watch_id in self._watches

    def describe(self, watch_id: str) -> dict:
        w = self._watches[watch_id]
        return {"watch_id": watch_id, "directory": w["directory"], "extensions": w["extensions"],
                "started_at": w["started_at"], "force_polling": self.force_polling,
                "events_so_far": sum(1 for e in self.events if e.watch_id == watch_id)}

    def events_since(self, watch_id: str, since: int) -> list[dict]:
        return [asdict(e) for e in self.events if e.watch_id == watch_id and e.seq > since]

    @property
    def active(self) -> list[str]:
        return list(self._watches)

    # ---- the watcher itself ---------------------------------------------------- #
    def _accept(self, name: str, exts: list[str]) -> bool:
        return not name.startswith((".", "~$")) and name.lower().endswith(tuple(exts))

    async def _run(self, watch: dict, path: Path) -> None:
        try:
            async for changes in awatch(path, stop_event=watch["stop"], debounce=self.debounce_ms,
                                        force_polling=self.force_polling, poll_delay_ms=100, recursive=False):
                for change, raw in sorted(changes, key=lambda c: c[1]):
                    name = Path(raw).name
                    if change == Change.deleted or name in watch["seen"] or not self._accept(name, watch["extensions"]):
                        continue
                    size = await _stable_size(Path(raw))
                    if size is None:
                        continue  # still being written; a later change event will bring it back
                    watch["seen"].add(name)
                    event = WatchEvent(next(self._seq), watch["watch_id"], f"{watch['directory']}/{name}", "added",
                                       size, _now())
                    self.events.append(event)
                    log.info("watch_directory %s: new file %s (%d bytes)", event.watch_id, event.path, size)
                    try:
                        await self.on_new(event, watch["session"])
                    except Exception as exc:  # noqa: BLE001 - a closed client must not kill the watcher
                        log.warning("could not notify the client about %s: %s", event.path, exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("watch %s failed: %s", watch["watch_id"], exc)


async def _stable_size(path: Path) -> int | None:
    """The file's size if it is the same across two checks STABLE_CHECK_SECONDS apart, else None."""
    try:
        first = os.stat(path).st_size
        await asyncio.sleep(STABLE_CHECK_SECONDS)
        second = os.stat(path).st_size
    except OSError:
        return None
    return second if first == second and second > 0 else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
