"""watch_directory() and batch_process() through a real MCP client session."""
import time

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session as connect

import mcp_batch
from tests.mcp_helpers import call, make_server, wait_for_events

pytestmark = pytest.mark.anyio


@pytest.fixture
def watched(data_root, monkeypatch):
    return make_server(data_root, monkeypatch, "--watch-force-polling", "true", "--watch-debounce-ms", "50")


async def test_a_new_resume_is_announced_once_with_notifications_and_becomes_a_resource(watched, data_root):
    notifications = []

    async def on_message(message):
        notifications.append(type(getattr(message, "root", message)).__name__)

    async with connect(watched, message_handler=on_message, logging_callback=lambda p: on_message(p)) as client:
        failed, info = await call(client, "watch_directory", directory="resumes")
        assert not failed and info["watch_id"] and info["directory"] == "resumes"
        await anyio.sleep(0.3)  # let the watcher take its first look

        (data_root / "resumes" / "new_person.md").write_text("# New Person\nGo developer\n")
        (data_root / "resumes" / ".draft.md").write_text("hidden")         # ignored: hidden
        (data_root / "resumes" / "~$lock.docx").write_text("lock")         # ignored: Office lock file
        (data_root / "resumes" / "photo.png").write_bytes(b"\x89PNG")      # ignored: not a resume format
        events = await wait_for_events(client, info["watch_id"])
        await anyio.sleep(0.6)  # time for any duplicate or ignored file to show up (none should)

        _, again = await call(client, "get_watch_events", watch_id=info["watch_id"])
        uris = {str(r.uri) for r in (await client.list_resources()).resources}
        text = (await client.read_resource("resume://new_person.md")).contents[0].text
        _, newer = await call(client, "get_watch_events", watch_id=info["watch_id"], since=events[-1]["seq"])
        _, stopped = await call(client, "unwatch_directory", watch_id=info["watch_id"])
        failed, gone = await call(client, "get_watch_events", watch_id=info["watch_id"])

    assert [e["path"] for e in again["events"]] == ["resumes/new_person.md"]
    assert events[0]["event"] == "added" and events[0]["size_bytes"] > 0 and events[0]["seq"] >= 1
    assert "resume://new_person.md" in uris and text.startswith("# New Person")
    assert newer["events"] == []
    assert "ResourceListChangedNotification" in notifications
    assert stopped["stopped"] is True and failed and gone["error_code"] == "NOT_FOUND"


async def test_files_present_before_the_watch_and_half_written_files_are_not_announced_early(watched, data_root):
    async with connect(watched) as client:
        _, info = await call(client, "watch_directory", directory="resumes")
        await anyio.sleep(0.3)
        (data_root / "resumes" / "slow.txt").write_text("")  # created empty, filled in later
        await anyio.sleep(0.8)
        _, early = await call(client, "get_watch_events", watch_id=info["watch_id"])
        (data_root / "resumes" / "slow.txt").write_text("Slow Writer\nPython\n")
        events = await wait_for_events(client, info["watch_id"])

    assert early["events"] == []  # existing files (ana_ruiz, ben_ito) and the empty file were not announced
    assert [(e["path"], e["size_bytes"]) for e in events] == [("resumes/slow.txt", 19)]


async def test_watch_directory_refuses_folders_outside_the_data_folders(watched):
    async with connect(watched) as client:
        results = [await call(client, "watch_directory", directory=d) for d in ("..", "milestone2_rag", "/tmp")]
    assert all(failed and body["error_code"] == "ACCESS_DENIED" for failed, body in results)


async def test_batch_reads_every_file_and_one_bad_file_does_not_fail_the_batch(data_root, monkeypatch):
    (data_root / "resumes" / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    server = make_server(data_root, monkeypatch)
    progress = []

    async def on_progress(done, total, message):
        progress.append((done, total))

    async with connect(server) as client:
        result = await client.call_tool("batch_process", {"directory": "resumes", "operation": "read"},
                                        progress_callback=on_progress)
    summary = result.structuredContent

    assert result.isError is False
    assert (summary["total"], summary["ok"], summary["failed"]) == (3, 2, 1)
    by_path = {r["path"]: r for r in summary["results"]}
    assert by_path["resumes/broken.pdf"]["ok"] is False and by_path["resumes/broken.pdf"]["error_code"] == "READ_FAILED"
    assert by_path["resumes/ana_ruiz.md"]["chars"] > 0 and by_path["resumes/ana_ruiz.md"]["preview"].startswith("# Ana")
    assert [d for d, _ in progress] == [1, 2, 3] and {t for _, t in progress} == {3}


async def test_batch_search_stats_and_per_file_policy(data_root, monkeypatch):
    server = make_server(data_root, monkeypatch)
    async with connect(server) as client:
        _, search = await call(client, "batch_process", paths=["resumes/ana_ruiz.md", "resumes/ben_ito.txt",
                                                               "../.env"], operation="search", keyword="react")
        _, stats = await call(client, "batch_process", directory="resumes", operation="stats")

    counts = {r["path"]: r.get("match_count", r.get("error_code")) for r in search["results"]}
    assert counts == {"resumes/ana_ruiz.md": 1, "resumes/ben_ito.txt": 1, "../.env": "ACCESS_DENIED"}
    assert all(r["ok"] and r["size_bytes"] > 0 and "content" not in r for r in stats["results"])


@pytest.mark.parametrize("args, flags", [
    ({"directory": "resumes"}, ["--batch-max-files", "1"]),                        # over the cap: TOO_LARGE
    ({"directory": "resumes", "paths": ["resumes/ana_ruiz.md"]}, []),              # both: INVALID_ARGUMENT
    ({"directory": "resumes", "operation": "search"}, []),                         # search without a keyword
])
async def test_batch_arguments_are_checked_up_front(data_root, monkeypatch, args, flags):
    server = make_server(data_root, monkeypatch, *flags)
    async with connect(server) as client:
        failed, body = await call(client, "batch_process", **args)
    assert failed and body["error_code"] == ("TOO_LARGE" if flags else "INVALID_ARGUMENT")


async def test_batch_runs_files_concurrently_up_to_the_limit():
    running, peak = 0, 0

    def slow_worker(path):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        time.sleep(0.2)
        running -= 1
        return {"ok": True}

    started = time.perf_counter()
    summary = await mcp_batch.run_batch([f"f{i}" for i in range(8)], "read", slow_worker, concurrency=4)

    assert peak == 4 and time.perf_counter() - started < 8 * 0.2 * 0.75  # well under sequential time
    assert [r["path"] for r in summary["results"]] == [f"f{i}" for i in range(8)]  # input order kept
