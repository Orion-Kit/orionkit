"""Global push-to-talk and mute hotkeys (pynput), handed to the asyncio loop.

Hold ``ORIONKIT_PTT_KEY`` (default ctrl+alt+space) to talk without the wake word; press ``ORIONKIT_MUTE_KEY``
(default ctrl+alt+m) to mute or unmute the microphone. Keys use pynput's syntax, e.g. ``<ctrl>+<alt>+<space>``.

Never fatal: without permission (macOS Accessibility / Input Monitoring), on Wayland, or with no display, the
listener doesn't start, ``status`` says why, and Orion carries on with the wake word and tap-to-talk.
Importing this module starts nothing; ``start()`` does.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

log = logging.getLogger("orionkit.hotkey")

DEFAULT_PTT = "<ctrl>+<alt>+<space>"
DEFAULT_MUTE = "<ctrl>+<alt>+m"
PERMISSION_HELP = ("Push-to-talk needs permission: System Settings > Privacy & Security > Accessibility (and Input "
                   "Monitoring) → allow your terminal app, then restart Orion.")


def label(chord: str) -> str:
    """'<ctrl>+<alt>+<space>' -> '⌃⌥Space' on a Mac, 'Ctrl+Alt+Space' elsewhere."""
    mac = sys.platform == "darwin"
    names = {"ctrl": "⌃" if mac else "Ctrl", "alt": "⌥" if mac else "Alt", "shift": "⇧" if mac else "Shift",
             "cmd": "⌘" if mac else "Win", "space": "Space"}
    parts = [names.get(p.strip("<>"), p.strip("<>").upper() if len(p.strip("<>")) == 1 else p.strip("<>").title())
             for p in chord.split("+")]
    return ("" if mac else "+").join(parts)


def _mac_trusted() -> bool | None:
    """Is this process allowed to watch the keyboard (Accessibility)? None if it can't be determined."""
    try:
        import ctypes
        lib = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except Exception:
        return None


class Hotkeys:
    def __init__(self, on_ptt_down: Callable[[], None], on_ptt_up: Callable[[], None], on_mute: Callable[[], None],
                 ptt: str | None = None, mute: str | None = None) -> None:
        self.ptt = ptt or os.environ.get("ORIONKIT_PTT_KEY", "").strip() or DEFAULT_PTT
        self.mute = mute or os.environ.get("ORIONKIT_MUTE_KEY", "").strip() or DEFAULT_MUTE
        self._down, self._up, self._mute = on_ptt_down, on_ptt_up, on_mute
        self._listener: Any = None
        self._held = False
        self.status = "off"          # off | on | needs permission | unavailable: <why>
        self.problem: str | None = None  # a sentence for the user when it couldn't start

    @property
    def hint(self) -> str:
        return f"Hold {label(self.ptt)} to talk"

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        if os.environ.get("ORIONKIT_PTT", "1") == "0":
            self.status = "off"
            return False
        if sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            return self._unavailable("Wayland doesn't allow global hotkeys for apps; use the wake word or tap to talk")
        if sys.platform == "darwin" and _mac_trusted() is False:
            self.status, self.problem = "needs permission", PERMISSION_HELP
            log.warning("push-to-talk: no Accessibility permission")
            return False
        try:
            from pynput import keyboard
            ptt_keys = set(keyboard.HotKey.parse(self.ptt))
            mute_hotkey = keyboard.HotKey(keyboard.HotKey.parse(self.mute), lambda: loop.call_soon_threadsafe(self._mute))
            pressed: set[Any] = set()

            def canon(key: Any) -> Any:
                return self._listener.canonical(key) if self._listener else key

            def on_press(key: Any) -> None:
                k = canon(key)
                pressed.add(k)
                mute_hotkey.press(k)
                if not self._held and ptt_keys <= pressed:
                    self._held = True
                    loop.call_soon_threadsafe(self._down)

            def on_release(key: Any) -> None:
                k = canon(key)
                pressed.discard(k)
                mute_hotkey.release(k)
                if self._held and k in ptt_keys:  # letting go of any key in the chord ends the hold
                    self._held = False
                    loop.call_soon_threadsafe(self._up)

            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.daemon = True
            self._listener.start()
            if sys.platform == "darwin" and not self._listener_trusted():
                self.stop()
                self.status, self.problem = "needs permission", PERMISSION_HELP
                return False
        except ImportError:  # not installed by default on Linux: its evdev dependency has no wheels and needs a compiler
            return self._unavailable("it needs pynput, which builds from source on Linux: sudo apt install build-essential "
                                     "python3-dev, then .venv/bin/pip install pynput, and restart Orion")
        except ValueError as e:
            return self._unavailable(f"can't read the hotkey setting ({e}); check ORIONKIT_PTT_KEY / ORIONKIT_MUTE_KEY")
        except Exception as e:  # no display, no X server, missing backend…
            return self._unavailable(f"{type(e).__name__}: {e}")
        self.status = "on"
        log.info("push-to-talk on: hold %s; mute %s", self.ptt, self.mute)
        return True

    def _listener_trusted(self, wait_s: float = 1.0) -> bool:
        """pynput sets IS_TRUSTED from its own thread once it has asked macOS; wait for that answer."""
        import time
        deadline = time.monotonic() + wait_s
        while "IS_TRUSTED" not in vars(self._listener) and time.monotonic() < deadline:
            time.sleep(0.02)
        return bool(vars(self._listener).get("IS_TRUSTED", _mac_trusted()))

    def _unavailable(self, why: str) -> bool:
        self.status, self.problem = f"unavailable: {why}"[:160], f"Push-to-talk is unavailable: {why}."
        log.warning("push-to-talk unavailable: %s", why)
        return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        if self.status == "on":
            self.status = "off"
