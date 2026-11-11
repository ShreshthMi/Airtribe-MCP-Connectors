"""
Configuration for the filesystem MCP server.

One frozen ServerConfig, built from four layers. Each layer overrides the ones after it:

  1. command-line flags          --root ./data --max-file-mb 5
  2. environment variables       MCP_FS_ROOT=./data MCP_FS_MAX_FILE_MB=5
  3. a TOML file                 mcp_server.toml (or --config / MCP_FS_CONFIG)
  4. the defaults below

The result is validated once at start-up. A bad value stops the server with a message
naming the field and the layer it came from, instead of failing later on a tool call.
The effective configuration is published read-only as the `config://server` resource.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "mcp_server.toml"
ENV_PREFIX = "MCP_FS_"
TRANSPORTS = ("stdio", "streamable-http", "sse")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    """The configuration is invalid; the message says which field and why."""


@dataclass(frozen=True)
class ServerConfig:
    root: Path = PROJECT_ROOT
    readable_folders: tuple[str, ...] = ("resumes", "job_descriptions", "reports")
    writable_folders: tuple[str, ...] = ("reports",)
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".pdf", ".docx", ".json", ".csv")
    max_file_mb: float = 20.0
    batch_max_files: int = 200
    batch_concurrency: int = 4  # measured: more threads did not help (docs/benchmarks.md)
    batch_preview_chars: int = 160
    watch_debounce_ms: int = 300
    watch_force_polling: bool = False
    watch_buffer_size: int = 500
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    sources: dict = field(default_factory=dict, compare=False)  # field -> "cli" | "env" | "file" | "default"

    def validate(self) -> "ServerConfig":
        def bad(name: str, why: str) -> ConfigError:
            return ConfigError(f"{name} ({self.sources.get(name, 'default')}): {why}")

        if not self.root.is_dir():
            raise bad("root", f"{self.root} is not a folder")
        for name in ("readable_folders", "writable_folders"):
            for folder in getattr(self, name):
                if not folder or "/" in folder or "\\" in folder or folder.startswith(".") or folder == "..":
                    raise bad(name, f"{folder!r} must be a plain, non-hidden folder name")
        if missing := [f for f in self.writable_folders if f not in self.readable_folders]:
            raise bad("writable_folders", f"{missing} must also be readable")
        if not self.allowed_extensions or any(not e.startswith(".") for e in self.allowed_extensions):
            raise bad("allowed_extensions", "give extensions such as '.pdf'")
        if not 0 < self.max_file_mb <= 20:
            raise bad("max_file_mb", "must be above 0 and at most 20 (Milestone 1's own read limit)")
        for name, low, high in (("batch_max_files", 1, 10_000), ("batch_concurrency", 1, 64),
                                ("batch_preview_chars", 0, 4000), ("watch_debounce_ms", 10, 60_000),
                                ("watch_buffer_size", 10, 100_000), ("port", 1, 65_535)):
            if not low <= getattr(self, name) <= high:
                raise bad(name, f"must be between {low} and {high}")
        if self.transport not in TRANSPORTS:
            raise bad("transport", f"must be one of {', '.join(TRANSPORTS)}")
        if self.log_level not in LOG_LEVELS:
            raise bad("log_level", f"must be one of {', '.join(LOG_LEVELS)}")
        return self

    def public_dict(self) -> dict:
        """The effective configuration, for config://server. The root is shown by name only,
        so no absolute host path leaves the server; nothing here is secret."""
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name not in ("root", "sources")}
        data = {k: list(v) if isinstance(v, tuple) else v for k, v in data.items()}
        return {"root": self.root.name, **data, "sources": dict(self.sources)}


FIELDS = {f.name: f for f in dataclasses.fields(ServerConfig) if f.name != "sources"}


def _coerce(name: str, raw, origin: str):
    """Turn a raw value from a flag, variable or TOML file into the field's type."""
    default = FIELDS[name].default
    try:
        if name == "root":
            return Path(str(raw)).expanduser().resolve()
        if isinstance(default, tuple):
            items = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
            return tuple(str(i).strip() for i in items if str(i).strip())
        if isinstance(default, bool):
            if isinstance(raw, bool):
                return raw
            text = str(raw).strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"expected true or false, got {raw!r}")
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        return str(raw).upper() if name == "log_level" else str(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} ({origin}): {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filesystem MCP server over the Milestone 1 tools.")
    parser.add_argument("--config", type=Path, help=f"TOML file (default {DEFAULT_CONFIG_FILE.name} if present)")
    for name in FIELDS:
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, default=None)
    return parser


def load_config(argv: list[str] | None = None, env: dict | None = None) -> ServerConfig:
    """Build and validate the configuration from flags, environment, TOML file and defaults."""
    env = os.environ if env is None else env
    args = _parser().parse_args([] if argv is None else argv)
    config_file = args.config or (Path(env[f"{ENV_PREFIX}CONFIG"]) if env.get(f"{ENV_PREFIX}CONFIG") else None)
    explicit = config_file is not None
    config_file = config_file or DEFAULT_CONFIG_FILE
    from_file: dict = {}
    if config_file.is_file():
        try:
            from_file = tomllib.loads(config_file.read_text()).get("server", {})
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{config_file.name}: not valid TOML ({exc})") from exc
        if unknown := sorted(set(from_file) - set(FIELDS)):
            raise ConfigError(f"{config_file.name}: unknown setting(s) {unknown}")
    elif explicit:
        raise ConfigError(f"config file {config_file} does not exist")

    values, sources = {}, {}
    for name in FIELDS:
        for origin, raw in (("cli", getattr(args, name)), ("env", env.get(f"{ENV_PREFIX}{name.upper()}")),
                            ("file", from_file.get(name))):
            if raw is not None and raw != "":
                values[name], sources[name] = _coerce(name, raw, origin), origin
                break
        else:
            sources[name] = "default"
    if "root" in values and sources["root"] == "file" and not Path(str(from_file["root"])).is_absolute():
        values["root"] = (config_file.parent / str(from_file["root"])).resolve()  # relative to the TOML file
    return ServerConfig(**values, sources=sources).validate()
