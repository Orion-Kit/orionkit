"""Bug reports: a diagnostics bundle with secrets removed, previewed by the user before anything is sent.

Sent to the OrionKit support service (which emails support@orion-kit.com); if that fails, saved to
~/.orionkit/reports/ with a mailto link so nothing is lost.
"""
from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from core.config import STATE_DIR
from core.version import __version__

SERVER = os.environ.get("ORIONKIT_LICENSE_SERVER", "https://orionkit-licensing.fly.dev")
SUPPORT = "support@orion-kit.com"
SECRETS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "[anthropic-key]"), (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[google-key]"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}"), "[google-key]"), (re.compile(r"\bre_[A-Za-z0-9]{16,}"), "[resend-key]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"), "[github-token]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[token]"),
    (re.compile(r"(?i)(bearer|token|api[_-]?key|secret|password|authorization)([\"'=:\s]+)[^\s\"',]{8,}"), r"\1\2[redacted]"),
    (re.compile(r"(?i)\?t=[A-Za-z0-9_\-]{16,}"), "?t=[access-code]"),
    (re.compile(r"\b(ntn|secret)_[A-Za-z0-9]{20,}"), "[notion-token]"), (re.compile(r"\bBSA[A-Za-z0-9_\-]{20,}"), "[brave-key]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"/Users/[^/\s]+"), "/Users/[you]"), (re.compile(r"/home/[^/\s]+"), "/home/[you]"),
]


def redact(text: str) -> str:
    """Known key shapes, plus the live value of every secret setting (SmartThings, Hue and HA tokens have no shape)."""
    from core.features import FEATURES
    for f in FEATURES:
        for x in f.fields:
            value = os.environ.get(x.env, "")
            if x.secret and len(value) >= 6:
                text = text.replace(value, f"[{x.env.lower()}]")
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    return text


def collect(description: str, status: dict[str, Any] | None = None, conversation: list[str] | None = None,
            log_lines: int = 200) -> str:
    log = STATE_DIR / "logs" / "orionkit.log"
    tail = log.read_text(errors="replace").splitlines()[-log_lines:] if log.is_file() else []
    parts = [
        f"# OrionKit bug report\n\n**What happened**\n\n{description.strip() or '(no description)'}\n",
        "## Environment",
        f"- OrionKit {__version__} · Python {platform.python_version()} · {platform.system()} {platform.release()} ({platform.machine()})",
        f"- Executable: {sys.executable}",
        f"- Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
    ]
    if status:
        parts += ["## Status", "```json", json.dumps(status, indent=1, default=str)[:4000], "```"]
    if conversation:
        parts += ["## Recent conversation (included by the user)", *[f"> {line}" for line in conversation[-20:]]]
    parts += ["## Log (last lines)", "```", *tail, "```"]
    return redact("\n".join(parts))


def send(report: str, license_key: str | None) -> tuple[bool, str]:
    try:
        r = httpx.post(f"{SERVER}/support/report", json={"report": report[:200_000], "license_key": license_key or "",
                                                         "version": __version__}, timeout=30)
        if r.status_code == 200:
            return True, f"Sent. Thanks! Reference {r.json().get('id', '')}. We'll reply by email if you added one."
        reason = r.json().get("detail", f"HTTP {r.status_code}") if r.headers.get("content-type", "").startswith("application/json") else f"HTTP {r.status_code}"
    except (httpx.HTTPError, ValueError) as e:
        reason = type(e).__name__
    return False, save_local(report, reason)


def save_local(report: str, reason: str = "") -> str:
    folder = STATE_DIR / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(report)
    mail = f"mailto:{SUPPORT}?subject={quote('OrionKit bug report')}&body={quote('The report is attached: ' + path.name)}"
    return (f"Couldn't send ({reason}). Saved to {path}; please email it to {SUPPORT}." if reason else str(path)) + f"\n{mail}"


def latest_report_path() -> Path | None:
    reports = sorted((STATE_DIR / "reports").glob("report-*.md"))
    return reports[-1] if reports else None
