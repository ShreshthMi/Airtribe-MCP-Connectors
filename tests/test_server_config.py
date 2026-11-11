"""server_config.py: flags over MCP_FS_* variables over the TOML file over defaults."""
import subprocess
import sys
from pathlib import Path

import pytest

from server_config import ConfigError, load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def toml(tmp_path):
    path = tmp_path / "server.toml"
    path.write_text('[server]\nbatch_concurrency = 3\nmax_file_mb = 5\nroot = "."\n')
    return path


def test_each_layer_overrides_the_ones_below_it(toml, tmp_path):
    env = {"MCP_FS_BATCH_CONCURRENCY": "5"}
    flags = ["--config", str(toml)]

    assert load_config(flags + ["--batch-concurrency", "7"], env).batch_concurrency == 7
    from_env = load_config(flags, env)
    assert from_env.batch_concurrency == 5 and from_env.sources["batch_concurrency"] == "env"
    from_file = load_config(flags, {})
    assert from_file.batch_concurrency == 3 and from_file.sources["batch_concurrency"] == "file"
    assert from_file.batch_max_files == 200 and from_file.sources["batch_max_files"] == "default"
    assert from_file.root == tmp_path  # a relative root in the file is relative to the file


def test_lists_and_booleans_from_the_environment(toml):
    cfg = load_config(["--config", str(toml)], {"MCP_FS_ALLOWED_EXTENSIONS": ".txt, .md",
                                                "MCP_FS_WATCH_FORCE_POLLING": "yes"})
    assert cfg.allowed_extensions == (".txt", ".md") and cfg.watch_force_polling is True


@pytest.mark.parametrize("flags, env, message", [
    (["--batch-concurrency", "0"], {}, r"batch_concurrency \(cli\): must be between 1 and 64"),
    (["--port", "x"], {}, r"port \(cli\)"),
    ([], {"MCP_FS_WATCH_FORCE_POLLING": "maybe"}, r"watch_force_polling \(env\): expected true or false"),
    (["--root", "/no/such/folder"], {}, r"root \(cli\): .* is not a folder"),
    (["--writable-folders", "reports,secrets"], {}, r"writable_folders .* must also be readable"),
    (["--readable-folders", "../etc"], {}, r"readable_folders .* plain, non-hidden folder name"),
    (["--max-file-mb", "50"], {}, r"max_file_mb .* at most 20"),
    (["--transport", "carrier-pigeon"], {}, r"transport .* one of stdio"),
])
def test_bad_values_fail_fast_and_name_the_field_and_layer(toml, flags, env, message):
    with pytest.raises(ConfigError, match=message):
        load_config(["--config", str(toml), *flags], env)


def test_unknown_settings_and_missing_files_are_refused(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[server]\nbatch_size = 3\n")
    with pytest.raises(ConfigError, match=r"unknown setting\(s\) \['batch_size'\]"):
        load_config(["--config", str(bad)], {})
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(["--config", str(tmp_path / "nope.toml")], {})


def test_the_server_refuses_to_start_with_a_bad_configuration():
    result = subprocess.run([sys.executable, str(ROOT / "filesystem_mcp_server.py"), "--batch-concurrency", "0"],
                            capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert result.returncode == 2 and "invalid configuration: batch_concurrency" in result.stderr
    assert result.stdout == ""


def test_the_committed_toml_matches_the_defaults():
    cfg = load_config([], {})
    assert all(source in ("file", "default") for source in cfg.sources.values())
    assert cfg.root == ROOT and cfg.writable_folders == ("reports",)
