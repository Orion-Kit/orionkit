"""Orion's features: what's free, what's Pro, what's switched on and how to set each one up.

One list drives the orb UI's Features panel, the terminal dashboard's hotkeys and `python main.py --features`.
Settings are written to the project's .env (owner-only), so they survive restarts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from core.config import PROJECT_ROOT

ENV_FILE = PROJECT_ROOT / ".env"
UPGRADE_URL = "https://orion-kit.com/#pricing"


@dataclass(frozen=True)
class Field:
    env: str
    label: str
    secret: bool = True
    placeholder: str = ""


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    description: str
    star: str                     # constellation star that represents it
    pro: bool
    kind: str                     # info | toggle | keys | settings | addon | pair | mcp_add
    env: str | None = None        # toggle: the env var holding "1"/"0"
    default_on: bool = False
    fields: tuple[Field, ...] = field(default_factory=tuple)
    how: str = ""


FEATURES: tuple[Feature, ...] = (
    Feature("voice", "Voice & “Hey Orion”", "Wake word, barge-in and on-device speech.", "Meissa", False, "info"),
    Feature("push_to_talk", "Push-to-talk", "Hold ⌃⌥Space (Ctrl+Alt+Space) anywhere to talk without the wake word; "
            "⌃⌥M mutes the mic.", "Meissa", False, "toggle", env="ORIONKIT_PTT", default_on=True,
            how="macOS asks once for Accessibility and Input Monitoring for your terminal app."),
    Feature("followup", "Follow-up mode", "One short window after each reply where you don't need the wake word.", "Meissa", False,
            "toggle", env="ORIONKIT_FOLLOWUP"),
    Feature("persona", "Voice & personality", "Choose Orion's voice, how fast it talks and its personality.", "Bellatrix", False,
            "settings", fields=(Field("ORIONKIT_VOICE", "Voice", secret=False,
                                      placeholder="Voice: bm_george, af_heart, am_michael, bf_emma… or a blend like bm_george:0.6,bm_fable:0.4"),
                                Field("ORIONKIT_SPEED", "Speed", secret=False, placeholder="Speed: 1.0 (0.7 slower – 1.4 faster)"),
                                Field("ORIONKIT_PERSONA", "Personality", secret=False,
                                      placeholder="Personality: concise · friendly · dev pair · narrator · or describe it")),
            how="Fill in any of these; leave the rest blank. Orion speaks a sample in the new voice."),
    Feature("local", "Local model", "Private, offline answers on your own machine through Ollama.", "Mintaka", False, "info"),
    Feature("claude", "Claude", "Deep reasoning, coding and multi-step work.", "Alnitak", True, "keys",
            fields=(Field("ANTHROPIC_API_KEY", "Anthropic API key", placeholder="sk-ant-…"),),
            how="Create a key at console.anthropic.com → API keys."),
    Feature("gemini", "Gemini", "Fast answers, screen reading and quick multi-tool requests.", "Alnilam", True, "keys",
            fields=(Field("GEMINI_API_KEY", "Gemini API key", placeholder="AIza…"),), how="Create a key at aistudio.google.com/apikey."),
    Feature("web", "Web access", "Search the web and read pages.", "Rigel", True, "toggle", env="ORIONKIT_WEB", default_on=True),
    Feature("tools", "Weather, reminders & system", "Live weather, timers and reminders that survive restarts, sandboxed shell, "
            "files and system stats.", "M42", True, "info"),
    Feature("documents", "Draft documents", "Say “draft a … ” and Orion writes it, saves it as Word, PDF or Markdown in its "
            "workspace's Documents folder, and opens it.", "M42", True, "info"),
    Feature("desktop", "Clipboard, apps & music", "“Rewrite what I copied”, “open Safari”, “next song”, “volume 30”: your "
            "clipboard, apps, Spotify or Music and the volume.", "M42", True, "info"),
    Feature("history", "Conversation history", "Remembers past conversations on this computer, so you can ask “what did I "
            "ask about Docker last week?”.", "Saiph", True, "toggle", env="ORIONKIT_HISTORY", default_on=True),
    Feature("routines", "Routines", "Say “good morning” to run several things at once; ask Orion to create one.", "Saiph", True,
            "info"),
    Feature("home_assistant", "Home Assistant", "Control 3,000+ smart-home brands, including Google Nest and Alexa devices.",
            "M42", True, "keys", fields=(Field("HA_URL", "Home Assistant URL", secret=False, placeholder="http://homeassistant.local:8123"),
                                       Field("HA_TOKEN", "Long-lived access token")),
            how="In Home Assistant: your profile → Security → Long-lived access tokens → Create."),
    Feature("smartthings", "SmartThings", "Samsung SmartThings plus Matter, Zigbee and Z-Wave devices on its hub.", "M42", True, "keys",
            fields=(Field("SMARTTHINGS_TOKEN", "Personal access token"),),
            how="Create a token at account.smartthings.com/tokens (devices: read and execute)."),
    Feature("hue", "Philips Hue", "Lights on your local Hue bridge.", "M42", True, "pair",
            fields=(Field("HUE_APP_KEY", "Hue key"),), how="Run: python tools/hue_pair.py, then press the button on your bridge."),
    Feature("companion", "Phone & browser companion", "Use Orion from your phone or any browser, privately over Tailscale.",
            "Meissa", True, "info", how="Run: python main.py --remote"),
    Feature("memory", "Long-term memory", "Remembers facts and preferences across conversations.", "Saiph", True, "addon"),
    Feature("browser", "Browser control", "Opens sites, clicks, types and fills forms in a real browser.", "Saiph", True, "addon"),
    Feature("notion", "Notion", "Search, read and edit your Notion pages.", "Saiph", True, "addon",
            fields=(Field("NOTION_TOKEN", "Notion integration secret"),), how="Create an internal integration at notion.so/profile/integrations."),
    Feature("git", "Git", "Status, diffs and history for your workspace repository.", "Saiph", True, "addon"),
    Feature("mcp", "Add an MCP server", "Plug in any Model Context Protocol server: its tools become Orion's tools.", "Saiph", True,
            "mcp_add", fields=(Field("MCP_NAME", "Name", secret=False, placeholder="Name, e.g. linear"),
                               Field("MCP_TARGET", "URL or command", secret=False, placeholder="https://…/mcp  or  npx -y @scope/server")),
            how="Paste the server's URL, or the command that starts it."),
)
BY_ID = {f.id: f for f in FEATURES}


def is_on(f: Feature, installed_addons: set[str] | None = None) -> bool:
    if f.kind == "info":
        return True
    if f.kind == "toggle":
        return os.environ.get(f.env or "", "1" if f.default_on else "0") == "1"
    if f.kind == "addon":
        return f.id in (installed_addons or set())
    if f.kind == "mcp_add":
        return False
    if f.kind == "settings":
        return any(os.environ.get(x.env) for x in f.fields)
    return all(os.environ.get(x.env) for x in f.fields)


def describe(licensed: bool, installed_addons: set[str] | None = None,
             custom_servers: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Every feature for the UI. ``custom_servers``: user-added MCP servers (name -> tool count), listed after "Add"."""
    out = []
    for f in FEATURES:
        locked = f.pro and not licensed
        on = (not locked) and is_on(f, installed_addons)
        status = "locked" if locked else ("on" if on else ("off" if f.kind == "toggle" else "add" if f.kind == "mcp_add" else
                                                              "customize" if f.kind == "settings" else "set up"))
        out.append({"id": f.id, "name": f.name, "description": f.description, "star": f.star, "pro": f.pro, "kind": f.kind,
                    "locked": locked, "on": on, "status": status, "how": f.how,
                    "fields": [{"env": x.env, "label": x.label, "secret": x.secret, "placeholder": x.placeholder} for x in f.fields]})
        if f.id == "mcp" and licensed:
            for name, n in sorted((custom_servers or {}).items()):
                out.append({"id": f"mcp:{name}", "name": f"MCP · {name}", "description": f"{n} tool{'s' * (n != 1)} · tap to remove",
                            "star": "Saiph", "pro": True, "kind": "mcp_installed", "locked": False, "on": True, "status": "on",
                            "how": "", "fields": []})
    return out


_KEY = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f\x85\u2028\u2029]")


def save_env(updates: dict[str, str]) -> None:
    """Set KEY=VALUE pairs in .env (owner-only file) and in this process."""
    for k, v in updates.items():
        if not _KEY.match(k) or _CONTROL.search(v):  # any line separator would let a value add other settings
            raise ValueError(f"invalid setting {k!r}")
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.is_file() else []
    kept = [ln for ln in lines if ln.split("=", 1)[0].strip() not in updates]
    kept += [f"{k}={v}" for k, v in updates.items()]
    tmp = ENV_FILE.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(kept) + "\n")
    os.replace(tmp, ENV_FILE)
    os.environ.update(updates)
