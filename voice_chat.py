#!/usr/bin/env python3
"""Talk to a local Ollama model by voice: "Hey Orion", streaming replies, talk over it to interrupt.

  python voice_chat.py                 voice (say "Hey Orion")
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

SYSTEM = ("You are Orion, a voice assistant. Your replies are spoken aloud: plain sentences, no markdown, "
          "short unless asked for depth. Latency-sensitive: begin your answer immediately.")


class LocalBrain:
    """Streaming chat with conversation memory against a local Ollama model."""

    def __init__(self, model: str, host: str, max_turns: int = 12) -> None:
        self._client = ollama.AsyncClient(host=host)
        self._model = model
        self._history: list[dict[str, str]] = []
        self._max = max_turns * 2

    async def respond(self, text: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": text})
        reply = ""
        try:
            stream = await self._client.chat(model=self._model, stream=True, keep_alive="30m",
                                             messages=[{"role": "system", "content": SYSTEM}, *self._history])
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
    duplex = DuplexController(SpeechSegmenter(mic), STTWorker(), TTSEngine(player), brain.respond, wake,
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


async def ask(brain: LocalBrain, text: str) -> None:
    async for token in brain.respond(text):
        print(token, end="", flush=True)
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("ORIONKIT_FAST_MODEL", "qwen2.5:3b"))
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--ask")
    p.add_argument("--no-wake", action="store_true")
    p.add_argument("--wake", default="hey orion", help='wake phrase, or an openWakeWord model name / .onnx path')
    p.add_argument("--input", type=int)
    p.add_argument("--output", type=int)
    args = p.parse_args()
    brain = LocalBrain(args.model, args.ollama)
    try:
        asyncio.run(ask(brain, args.ask) if args.ask else voice(args, brain))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
