"""Runtime settings from the environment, with an optional KEY=VALUE `.env` file in the project root."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = Path(os.environ.get("ORIONKIT_HOME", Path.home() / ".orionkit"))


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments; real environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def is_local(url: str | None) -> bool:
    """A loopback, private-network or .local address (Ollama, Home Assistant, a Hue bridge). A corporate proxy can't
    reach those, so clients for them ignore HTTP(S)_PROXY and the system proxy."""
    import ipaddress
    from urllib.parse import urlsplit
    if not url:
        return False
    host = (urlsplit(url if "://" in url else f"//{url}").hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host.endswith((".local", ".lan", ".home.arpa", ".internal")) or ("." not in host and ":" not in host):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def local_only() -> bool:
    """ORIONKIT_LOCAL_ONLY=1: no cloud models, web or weather tools, location lookup, update check or diagnostics."""
    return os.environ.get("ORIONKIT_LOCAL_ONLY", "0") == "1"


def _paths(var: str, default: list[Path]) -> list[Path]:
    raw = os.environ.get(var)
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p] if raw else default


@dataclass(frozen=True)
class Settings:
    fast_model: str = "qwen3:4b-instruct"
    vision_model: str = "gemini-flash-latest"   # Gemini 2.0 Flash is retired
    deep_model: str = "claude-sonnet-5-5"       # Claude 3.5 Sonnet is retired
    deep_effort: str = "medium"
    ollama_host: str = "http://127.0.0.1:11434"
    workspace: Path = STATE_DIR / "workspace"
    allowed_dirs: list[Path] = field(default_factory=list)
    bash_allow_network: bool = False
    web_enabled: bool = True  # web_search / web_fetch tools (ORIONKIT_WEB=0 turns them off)
    followup: bool = False    # ORIONKIT_FOLLOWUP=1: one short wake-word-free window after each reply
    ha_url: str | None = None
    ha_token: str | None = None
    mcp_config: Path = PROJECT_ROOT / "mcp_servers.json"
    local_only: bool = False  # ORIONKIT_LOCAL_ONLY=1: nothing leaves this computer (see README, Privacy)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        workspace = Path(os.environ.get("ORIONKIT_WORKSPACE", STATE_DIR / "workspace")).expanduser()
        return cls(
            fast_model=os.environ.get("ORIONKIT_FAST_MODEL", cls.fast_model),
            vision_model=os.environ.get("ORIONKIT_VISION_MODEL", cls.vision_model),
            deep_model=os.environ.get("ORIONKIT_DEEP_MODEL", cls.deep_model),
            deep_effort=os.environ.get("ORIONKIT_DEEP_EFFORT", cls.deep_effort),
            ollama_host=os.environ.get("OLLAMA_HOST", cls.ollama_host),
            workspace=workspace,
            allowed_dirs=_paths("ORIONKIT_ALLOWED_DIRS", [workspace]),
            bash_allow_network=os.environ.get("ORIONKIT_BASH_NETWORK", "0") == "1",
            web_enabled=os.environ.get("ORIONKIT_WEB", "1") == "1" and not local_only(),
            local_only=local_only(),
            followup=os.environ.get("ORIONKIT_FOLLOWUP", "0") == "1",
            ha_url=os.environ.get("HA_URL"),
            ha_token=os.environ.get("HA_TOKEN"),
            mcp_config=Path(os.environ.get("ORIONKIT_MCP_CONFIG", cls.mcp_config)),
        )
