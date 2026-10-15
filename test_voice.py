#!/usr/bin/env python3
"""OrionKit voice harness.

  python test_voice.py                 live: say the wake word, speak, hear the reply echoed back
  python test_voice.py --no-wake       live, always listening (no wake word)
  python test_voice.py --self-test     headless: synthetic speech through the full pipeline, with asserts
  python test_voice.py --list-devices

Talk over a reply to test barge-in (use headphones, or speak clearly louder than the speaker).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import AsyncIterator

import numpy as np
from scipy.signal import resample_poly

from core.audio.duplex import DuplexConfig, DuplexController, State, TurnEvent
from core.audio.mic import FRAME_SAMPLES, SAMPLE_RATE, MicStream
from core.audio.stt import STTWorker
from core.audio.tts import AudioPlayer, TTSEngine
from core.audio.vad import SpeechSegmenter
from core.audio.echo import EchoGate
from core.audio.wake import DEFAULT_MODEL, KeywordWakeDetector, create_wake

LONG_REPLY = (
    "Here is a deliberately long answer so you have time to interrupt me. "
    "The quick brown fox jumps over the lazy dog, again and again, across the meadow. "
    "I will keep talking about foxes, dogs and meadows until somebody stops me. "
    "Seriously, this could go on for quite a while, so feel free to cut in at any moment."
)


def echo_responder(token_delay_s: float = 0.03):
    """Stand-in for the LLM: streams the reply word by word like a token stream."""

    async def respond(text: str) -> AsyncIterator[str]:
        reply = LONG_REPLY if "long" in text.lower() else f"You said: {text}"
        for word in reply.split(" "):
            await asyncio.sleep(token_delay_s)
            yield word + " "

    return respond


def build(args: argparse.Namespace, headless: bool):
    mic = MicStream(device=False if headless else args.input)
    player = AudioPlayer(device=False if headless else args.output)
    t0 = time.perf_counter()
    wake = None if args.no_wake else create_wake(mic, args.wake_model, args.wake_threshold)
    seg = SpeechSegmenter(mic)
    stt = STTWorker(args.stt_model)
    tts = TTSEngine(player, voice=args.voice)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")
    duplex = DuplexController(seg, stt, tts, echo_responder(), wake, DuplexConfig(), EchoGate(mic, player))
    return mic, player, tts, duplex, wake


def describe(e: TurnEvent) -> str | None:
    if e.kind == "transcript" and e.transcript:
        t = e.transcript
        return f"📝 \"{t.text}\"  (STT {t.latency_ms:.0f} ms for {t.audio_s:.1f}s audio)"
    if e.kind == "reply" and e.result:
        r = e.result
        return "✋ reply interrupted" if r.interrupted else f"🔊 replied: {r.sentences} chunks, first audio {r.first_audio_ms:.0f} ms after first token"
    return {"wake": "👂 wake word", "barge_in": "⚡ barge-in: speaker paused", "resumed": "▶ not an interruption: resumed", "timeout": "💤 no speech, back to idle"}.get(e.kind)


async def live(args: argparse.Namespace) -> None:
    mic, player, _tts, duplex, _wake = build(args, headless=False)
    async with mic:
        await player.start()
        task = asyncio.create_task(duplex.run())
        print(f"Listening. {'Speak any time.' if args.no_wake else f'Say the wake word ({args.wake_model}) to start.'} Ctrl-C quits.")
        try:
            while True:
                e = await duplex.events.get()
                if e.kind == "state":
                    print(f"   · {e.state.value}")
                elif line := describe(e):
                    print(line)
        finally:
            task.cancel()
            await player.stop()


# ---- headless self-test ---------------------------------------------------------------------------

async def self_test(args: argparse.Namespace) -> int:
    mic, player, tts, duplex, wake = build(args, headless=True)
    wake_phrase = "Hey Orion." if isinstance(wake, KeywordWakeDetector) else "Hey Jarvis."
    await mic.start()
    await player.start()
    events: list[TurnEvent] = []

    def to_mic(text: str) -> np.ndarray:  # synthesize "user" speech, 24 kHz -> 16 kHz int16
        pcm = resample_poly(tts.synthesize(text), 2, 3)
        return (np.clip(pcm, -1, 1) * 32767).astype(np.int16)

    pending: list[np.ndarray] = []  # speech queued for the virtual mic
    loopback = {"on": False}        # simulate laptop speakers leaking into the mic
    rng = np.random.default_rng(1)

    def speaker_leak() -> np.ndarray:
        """What a laptop mic hears of the speaker: ~100 ms late, attenuated, saturated, plus room noise."""
        delay = int(0.1 * SAMPLE_RATE)
        ref = player.reference.latest(FRAME_SAMPLES + delay)[:FRAME_SAMPLES].astype(np.float32) / 32768
        if len(ref) < FRAME_SAMPLES:
            ref = np.pad(ref, (FRAME_SAMPLES - len(ref), 0))
        return np.tanh(2.0 * ref) / 2.0 * 0.6 + rng.normal(0, 0.002, FRAME_SAMPLES)

    async def mic_clock() -> None:
        """Like a real microphone: a 512-sample frame every 32 ms, silence when nobody talks."""
        silent = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        while True:
            frame = pending.pop(0) if pending else silent
            if loopback["on"]:
                mixed = frame.astype(np.float32) / 32768 + speaker_leak()
                frame = (np.clip(mixed, -1, 1) * 32767).astype(np.int16)
            mic.feed(frame)
            await asyncio.sleep(FRAME_SAMPLES / SAMPLE_RATE)

    async def say(pcm: np.ndarray) -> float:
        """Queue speech on the virtual mic; returns when its last sample has been captured."""
        pcm = np.pad(pcm, (0, -len(pcm) % FRAME_SAMPLES))  # whole frames, like a real sound card
        pending.extend(pcm[i:i + FRAME_SAMPLES] for i in range(0, len(pcm), FRAME_SAMPLES))
        while pending:
            await asyncio.sleep(0.005)
        return time.monotonic()

    async def silence(seconds: float) -> None:
        await asyncio.sleep(seconds)

    async def wait_for(kind: str, timeout: float = 15.0) -> TurnEvent:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if hit := next((e for e in events if e.kind == kind), None):
                events.remove(hit)
                return hit
            await asyncio.sleep(0.01)
        raise TimeoutError(kind)

    async def collect() -> None:
        while True:
            e = await duplex.events.get()
            if args.verbose and e.kind == "state":
                print(f"      · {e.state.value} t={e.at:.2f}")
            if e.kind != "state":
                events.append(e)
                if line := describe(e):
                    print("   " + line)

    checks: list[tuple[str, bool, str]] = []
    task = asyncio.gather(duplex.run(), collect(), mic_clock())
    try:
        await silence(1.0)
        if not args.no_wake:
            print("1. wake word")
            await say(to_mic(wake_phrase))
            try:
                e = await wait_for("wake", 5)
                checks.append(("wake word fires", True, f"state={e.state.value}"))
            except TimeoutError:
                checks.append(("wake word fires", False, "no detection on synthetic voice"))
                return report(checks)
            await silence(0.3)

        print("2. speech -> STT -> streaming TTS echo")
        spoken = "What is the weather like in Chicago today?"
        pcm = to_mic(spoken)
        voiced = np.flatnonzero(np.abs(pcm) > 0.02 * np.abs(pcm).max())
        tail_s = (len(pcm) - voiced[-1]) / SAMPLE_RATE  # the clip's own trailing silence
        spoke_until = await say(pcm) - tail_s
        te = await wait_for("transcript")
        t = te.transcript
        print(f"   ⏱  end of speech → transcript {(te.at - spoke_until) * 1000:.0f} ms (turn detection + STT)")
        assert t is not None
        words = lambda s: [w.strip(".,?!").lower() for w in s.split()]
        checks.append(("transcript matches", words(t.text) == words(spoken), repr(t.text)))
        checks.append(("STT latency < 300 ms", t.latency_ms < 300, f"{t.latency_ms:.0f} ms"))
        r = (await wait_for("reply")).result
        assert r is not None
        checks.append(("reply streamed + played", not r.interrupted and r.sentences >= 1, f"{r.sentences} chunk(s)"))
        checks.append(("first audio < 500 ms after first token", (r.first_audio_ms or 1e9) < 500, f"{r.first_audio_ms:.0f} ms"))
        played_s = sum(len(c) for c in player.captured) / player.sample_rate
        checks.append(("audio reached the speaker", played_s > 1.0, f"{played_s:.1f}s played"))
        await silence(1.0)

        print("3. barge-in (with speaker leaking into the mic)")
        loopback["on"] = True
        if not args.no_wake:
            await say(to_mic(wake_phrase))
            await wait_for("wake", 5)
            await silence(0.3)
        await say(to_mic("Tell me something long."))
        await wait_for("transcript")
        deadline = time.monotonic() + 10
        while duplex.state is not State.SPEAKING or not player.playing:
            if time.monotonic() > deadline:
                raise TimeoutError("speaking")
            await asyncio.sleep(0.01)
        await silence(2.5)  # Orion talks; its own voice leaks into the mic
        own_voice_barge = [e for e in events if e.kind == "barge_in"]
        ignored = sum(e.kind == "echo_ignored" for e in events)
        checks.append(("own voice through speakers ignored", not own_voice_barge, f"{ignored} echo segment(s) rejected"))
        interruption = to_mic("Stop. What time is it in London?")
        interrupt_at = time.monotonic()
        speech_task = asyncio.create_task(say(interruption))
        e = await wait_for("barge_in", 5)
        reaction_ms = (e.at - interrupt_at) * 1000
        checks.append(("barge-in < 800 ms (speakers leaking)", reaction_ms < 800, f"{reaction_ms:.0f} ms"))
        checks.append(("speaker silenced", not player.audible, f"audible={player.audible}"))
        r = (await wait_for("reply")).result
        checks.append(("TTS generation halted", bool(r and r.interrupted), ""))
        await speech_task
        await silence(0.6)
        t2 = (await wait_for("transcript")).transcript
        checks.append(("interrupting speech becomes next turn", bool(t2 and "london" in t2.text.lower()), repr(t2.text if t2 else None)))
        await wait_for("reply")
    except TimeoutError as exc:
        checks.append((f"timed out waiting for {exc}", False, ""))
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await player.stop()
    return report(checks)


def report(checks: list[tuple[str, bool, str]]) -> int:
    print()
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} {detail}")
    failed = sum(not ok for _, ok, _ in checks)
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--no-wake", action="store_true")
    p.add_argument("--wake-model", default=DEFAULT_MODEL, help='phrase ("hey orion") or openWakeWord model name / .onnx path')
    p.add_argument("--wake-threshold", type=float, default=None)
    p.add_argument("--stt-model", default="base.en")
    p.add_argument("--voice", default="bm_george:0.55,bm_fable:0.45")
    p.add_argument("--input", type=int, default=None, help="input device index")
    p.add_argument("--output", type=int, default=None, help="output device index")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, force=True, format="%(relativeCreated)7.0f %(name)s %(message)s")
    logging.getLogger("orionkit").setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0
    try:
        return asyncio.run(self_test(args)) if args.self_test else asyncio.run(live(args)) or 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
