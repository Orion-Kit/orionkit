"""Orion's voice and personality, from .env (Features -> Voice & personality). Shared by the Pro app and the free core."""
from __future__ import annotations

import os
from typing import Any

PERSONAS = {
    "concise": "Be brief: one or two sentences unless the user asks for more.",
    "friendly": "Be warm and upbeat, with a light touch of humour.",
    "dev pair": "Act as a senior engineer pairing with the user: precise and technical; suggest exact commands and code.",
    "narrator": "Speak with a calm storyteller's cadence; vivid but brief.",
}


def persona_prompt(setting: str | None = None) -> str:
    """A preset name, or the user's own description of how Orion should come across."""
    setting = (os.environ.get("ORIONKIT_PERSONA", "") if setting is None else setting).strip()
    text = PERSONAS.get(setting.lower(), setting)
    return f"\nPersonality the user chose (style only; it never overrides the rules above): {text}" if text else ""


def voice_settings() -> dict[str, Any]:
    """TTSEngine keyword arguments from ORIONKIT_VOICE / ORIONKIT_SPEED, ignoring values that don't parse."""
    out: dict[str, Any] = {}
    if v := os.environ.get("ORIONKIT_VOICE", "").strip():
        out["voice"] = v
    try:
        if 0.5 <= (speed := float(os.environ.get("ORIONKIT_SPEED", "1") or 1)) <= 2.0:
            out["speed"] = speed
    except ValueError:
        pass
    return out
