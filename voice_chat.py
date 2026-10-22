#!/usr/bin/env python3
"""Talk to a local Ollama model by voice: "Hey Orion", streaming replies, talk over it to interrupt.

  python voice_chat.py                 the orb window (say "Hey Orion", or type)
  python voice_chat.py --terminal      voice in the terminal only
  python voice_chat.py --no-wake       always listening
  python voice_chat.py --ask "hello"   text in, streamed text out (no audio)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator

import ollama

from core.persona import persona_prompt, voice_settings

SYSTEM = ("You are Orion, a voice assistant. Your replies are spoken aloud: plain sentences, no markdown, "
          "short unless asked for depth. Latency-sensitive: begin your answer immediately.")


class LocalBrain:
    """Streaming chat with conversation memory against a local Ollama model."""

    def __init__(self, model: str, host: str, max_turns: int = 12) -> None:
        self._client = ollama.AsyncClient(host=host)
        self._model = self.model = model
        self._history: list[dict[str, str]] = []
        self._max = max_turns * 2

    async def respond(self, text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": text})
        reply = ""
        try:
            stream = await self._client.chat(model=self._model, stream=True, keep_alive="30m",
                                             messages=[{"role": "system", "content": SYSTEM + persona_prompt()}, *self._history])
            async for part in stream:
                if part.message.content:
                    reply += part.message.content
                    yield part.message.content
        except (ollama.ResponseError, OSError) as e:
            msg = f"I can't reach the local model ({e}). Is Ollama running with {self._model} pulled?"
            reply += msg
            yield msg
        finally:  # also runs on barge-in, so the partial reply stays in context
            if reply:
                self._history.append({"role": "assistant", "content": reply})
            del self._history[:-self._max]


async def voice(args: argparse.Namespace, brain: LocalBrain) -> None:
    from core.audio.duplex import DuplexConfig, DuplexController
    from core.audio.echo import EchoGate
    from core.audio.mic import MicStream
    from core.audio.stt import STTWorker
    from core.audio.tts import AudioPlayer, TTSEngine
    from core.audio.vad import SpeechSegmenter
    from core.audio.wake import create_wake

    mic, player = MicStream(device=args.input), AudioPlayer(device=args.output)
    print("Loading voice models…")
    wake = None if args.no_wake else create_wake(mic, args.wake)
    duplex = DuplexController(SpeechSegmenter(mic), STTWorker(), TTSEngine(player, **voice_settings()), brain.respond, wake,
                              DuplexConfig(), EchoGate(mic, player))
    async with mic:
        await player.start()
        task = asyncio.create_task(duplex.run())
        print(f"Ready. {'Speak any time.' if args.no_wake else 'Say \"Hey Orion\".'} Ctrl-C quits.")
        try:
            while True:
                e = await duplex.events.get()
                if e.kind == "wake":
                    print("👂 listening")
                elif e.kind == "transcript" and e.transcript:
                    print(f"you  › {e.transcript.text}")
                elif e.kind == "barge_in":
                    print("⚡ interrupted")
        finally:
            task.cancel()
            await player.stop()


async def orb(args: argparse.Namespace, brain: LocalBrain) -> None:
    """The orb UI for the free core. Pro features are listed and open the upgrade prompt when tapped."""
    import webbrowser

    from core.audio.duplex import DuplexConfig, DuplexController
    from core.audio.echo import EchoGate
    from core.audio.mic import MicStream
    from core.audio.stt import STTWorker
    from core.audio.tts import AudioPlayer, TTSEngine
    from core.audio.vad import SpeechSegmenter
    from core.audio.wake import create_wake
    from core.features import UPGRADE_URL, describe, save_env
    from core.ui.server import UIServer

    holder: dict = {}

    def features_msg() -> dict:
        items = [f for f in describe(licensed=False) if f["id"] not in ("push_to_talk", "diagnostics")]  # Pro app only
        for f in items:  # the live follow-up switch, not the saved default
            if f["id"] == "followup" and "duplex" in holder:
                f["on"] = holder["duplex"].followup
                f["status"] = "on" if f["on"] else "off"
        return {"t": "features", "items": items, "licensed": False}

    async def on_command(cmd: dict) -> None:
        d, kind = holder.get("duplex"), cmd["t"]
        if kind == "features":
            ui.send(features_msg(), keep="features")
        elif kind == "feature":
            if cmd.get("id") == "followup" and d:
                d.followup = bool(cmd.get("on"))
                save_env({"ORIONKIT_FOLLOWUP": "1" if d.followup else "0"})
                ui.send({"t": "feature_result", "id": "followup", "ok": True, "message": f"Follow-up {'on' if d.followup else 'off'}."})
            elif cmd.get("id") == "persona" and cmd.get("values"):
                v = {k: str(x).strip() for k, x in cmd["values"].items()
                     if k in ("ORIONKIT_VOICE", "ORIONKIT_SPEED", "ORIONKIT_PERSONA") and str(x).strip()}
                try:
                    save_env(v)
                    if d and (v.get("ORIONKIT_VOICE") or v.get("ORIONKIT_SPEED")):
                        d._tts.configure(voice=v.get("ORIONKIT_VOICE"), speed=float(v["ORIONKIT_SPEED"]) if "ORIONKIT_SPEED" in v else None)
                    ui.send({"t": "feature_result", "id": "persona", "ok": True, "message": "Saved."})
                except (ValueError, KeyError) as e:
                    ui.send({"t": "feature_result", "id": "persona", "ok": False, "message": f"Couldn't use that: {e}"})
            elif cmd.get("id") in ("voice", "local"):
                ui.send({"t": "feature_result", "id": cmd["id"], "ok": True, "message": "Built in and always on."})
            else:
                ui.send({"t": "paywall", "id": cmd.get("id")})
            ui.send(features_msg(), keep="features")
        elif kind == "report_preview":
            from core import bugreport
            ui.send({"t": "report_preview", "report": await asyncio.to_thread(bugreport.collect, str(cmd.get("text", ""))[:5000])})
        elif kind == "report_send":
            from core import bugreport
            ok, message = await asyncio.to_thread(bugreport.send, bugreport.redact(str(cmd.get("report", ""))), None)
            ui.send({"t": "report_result", "ok": ok, "message": message,
                     "mailto": next((x for x in message.split("\n") if x.startswith("mailto:")), None)})
        elif kind == "upgrade":
            webbrowser.open(UPGRADE_URL)
        elif kind == "activate":
            ui.send({"t": "activate_result", "ok": False, "message": "License keys activate OrionKit Pro. Download it from your purchase "
                                                                     "receipt, then run: python main.py --license <key>"})
        elif d is None:
            return
        elif kind == "say" and str(cmd.get("text", "")).strip():
            await d.submit_text(str(cmd["text"]).strip()[:4000])
        elif kind == "talk":
            d.talk()
        elif kind == "stop":
            d.stop()
        elif kind == "mute":
            mic.muted = bool(cmd.get("on"))
        elif kind == "followup":
            d.followup = bool(cmd.get("on"))
        elif kind == "mode":
            ui.send({"t": "alert", "text": "Terminal mode: run python voice_chat.py --terminal"})
        status()

    ui = UIServer(on_command, port=args.port)
    mic, player = MicStream(device=args.input), AudioPlayer(device=args.output)

    def status() -> None:
        d = holder.get("duplex")
        ui.send({"t": "status", "version": "core", "license": "Free", "licensed": False, "location": "", "wake": args.wake,
                 "followup": bool(d and d.followup), "muted": mic.muted,
                 "providers": {"local": {"state": "closed", "label": brain.model}}, "mcp": {"tools": "Pro"}, "telemetry": {}}, keep="status")

    async def respond(text: str):
        # The reply is shown as it's spoken, not as the model writes it (that runs a sentence or more ahead of the voice).
        tts.caption(lambda chunk: ui.send({"t": "token", "text": chunk}),
                    lambda interrupted: ui.send({"t": "reply_end", "interrupted": interrupted}))
        ui.send({"t": "event", "kind": "route", "detail": {"provider": "ollama", "model": brain.model, "tier": "fast"}})
        async for token in brain.respond(text):
            yield token

    await ui.start()
    print(f"Orion is running: {ui.url}\nCtrl-C quits.")
    webbrowser.open(ui.url)
    print("Loading voice models…")
    wake = None if args.no_wake else create_wake(mic, args.wake)
    tts = TTSEngine(player, **voice_settings())
    duplex = DuplexController(SpeechSegmenter(mic), STTWorker(), tts, respond, wake,
                              DuplexConfig(followup=os.environ.get("ORIONKIT_FOLLOWUP") == "1"), EchoGate(mic, player))
    holder["duplex"] = duplex
    async with mic:
        await player.start()
        ui.send(features_msg(), keep="features")
        status()

        async def ticks() -> None:
            while True:
                ui.send({"t": "tick", "state": duplex.state.value, "mic": round(mic.rms, 4), "out": round(player.rms, 4), "muted": mic.muted})
                await asyncio.sleep(1 / 15)

        async def forward() -> None:
            while True:
                e = await duplex.events.get()
                if e.kind == "transcript" and e.transcript:
                    ui.send({"t": "user", "text": e.transcript.text})
                elif e.kind in ("barge_in", "wake"):
                    ui.send({"t": "event", "kind": e.kind, "detail": {}})
        try:
            await asyncio.gather(duplex.run(), ticks(), forward())
        finally:
            await player.stop()
            await ui.stop()


async def ask(brain: LocalBrain, text: str) -> None:
    async for token in brain.respond(text):
        print(token, end="", flush=True)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("ORIONKIT_FAST_MODEL", "qwen3:4b-instruct"))
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--ask")
    p.add_argument("--no-wake", action="store_true")
    p.add_argument("--wake", default="hey orion", help='wake phrase, or an openWakeWord model name / .onnx path')
    p.add_argument("--terminal", action="store_true", help="voice in the terminal only, no orb window")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--input", type=int)
    p.add_argument("--output", type=int)
    args = p.parse_args()
    brain = LocalBrain(args.model, args.ollama)
    try:
        asyncio.run(ask(brain, args.ask) if args.ask else voice(args, brain) if args.terminal else orb(args, brain))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
