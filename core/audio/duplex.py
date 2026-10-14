"""Duplex turn controller with barge-in.

States: IDLE (waiting for the wake word) -> LISTENING -> THINKING -> SPEAKING -> LISTENING (follow-up
window) -> IDLE. If confirmed user speech (or the wake word) arrives while SPEAKING, the TTS job is
cancelled, the output buffer is flushed and the controller returns to LISTENING with the interrupting
utterance still being captured, so it becomes the next turn.

Echo: without hardware echo cancellation the mic hears the speaker. While SPEAKING the VAD threshold
and minimum speech length are raised, and an optional ``EchoGate`` rejects "speech" that is explained
by what the speaker just played, so Orion's own voice does not trigger barge-in.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from enum import Enum

from .echo import EchoGate
from .stt import STTWorker, Transcript
from .tts import SpeakResult, TTSEngine
from .vad import SpeechSegmenter, SpeechStart, Utterance
from .wake import KeywordWakeDetector, WakeEvent, WakeWordDetector

log = logging.getLogger("orionkit.duplex")

Responder = Callable[[str], AsyncIterator[str]]  # transcript -> streamed reply tokens


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class DuplexConfig:
    listen_timeout_s: float = 8.0   # after wake, give up if nobody speaks
    followup_s: float = 6.0         # after a reply, keep listening without the wake word
    barge_in_threshold: float = 0.8  # VAD threshold while Orion is talking
    barge_in_min_ms: int = 300       # confirmed speech needed to interrupt
    idle_threshold: float = 0.5
    idle_min_ms: int = 160


@dataclass(frozen=True)
class TurnEvent:
    """Observable record of what happened, for UIs, logs and tests."""
    kind: str  # wake | transcript | reply | barge_in | timeout | state
    state: State
    transcript: Transcript | None = None
    result: SpeakResult | None = None
    at: float = 0.0


class DuplexController:
    def __init__(
        self,
        segmenter: SpeechSegmenter,
        stt: STTWorker,
        tts: TTSEngine,
        respond: Responder,
        wake: "WakeWordDetector | KeywordWakeDetector | None" = None,
        config: DuplexConfig = DuplexConfig(),
        echo_gate: EchoGate | None = None,
    ) -> None:
        self._seg = segmenter
        self._stt = stt
        self._tts = tts
        self._respond = respond
        self._wake = wake
        self._cfg = config
        self._echo = echo_gate
        self.state = State.IDLE if wake else State.LISTENING
        self.events: asyncio.Queue[TurnEvent] = asyncio.Queue()
        self._deadline: float | None = None
        self._reply: asyncio.Task[None] | None = None
        self._apply_vad_profile()

    async def run(self) -> None:
        tasks = [asyncio.create_task(self._seg.run()), asyncio.create_task(self._vad_loop())]
        if self._wake:
            tasks += [asyncio.create_task(self._wake.run()), asyncio.create_task(self._wake_loop())]
            tasks.append(asyncio.create_task(self._timeout_loop()))
        try:
            await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()

    # ---- inputs -------------------------------------------------------------------------------

    async def _wake_loop(self) -> None:
        assert self._wake is not None
        while True:
            event: WakeEvent = await self._wake.events.get()
            if self.state is State.SPEAKING:
                if self._is_echo():
                    continue
                self._barge_in()
            else:
                self._seg.reset()  # discard the wake phrase itself so it never becomes a turn
                if self.state is State.IDLE:
                    self._set(State.LISTENING, self._cfg.listen_timeout_s)
            self._emit("wake")
            log.info("wake %s %.2f", event.model, event.score)

    async def _vad_loop(self) -> None:
        while True:
            event = await self._seg.events.get()
            log.debug("vad %s in %s", type(event).__name__, self.state.value)
            if isinstance(event, SpeechStart):
                if self.state is State.SPEAKING:
                    if self._is_echo():
                        # Drop the echo segment so continued real speech raises a fresh SpeechStart.
                        self._seg.reset()
                        continue
                    self._barge_in()
                elif self.state is State.LISTENING:
                    self._deadline = None  # user is talking; don't time out mid-sentence
            elif isinstance(event, Utterance) and self.state is State.LISTENING:
                await self._handle_utterance(event)

    async def _timeout_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self.state is State.LISTENING and self._deadline and time.monotonic() > self._deadline:
                self._set(State.IDLE)
                self._emit("timeout")

    # ---- turn handling ------------------------------------------------------------------------

    async def _handle_utterance(self, utt: Utterance) -> None:
        self._set(State.THINKING)
        transcript = await self._stt.transcribe(utt.audio)
        if not transcript.text:
            self._set(State.LISTENING, self._cfg.followup_s)
            return
        self._emit("transcript", transcript=transcript)
        self._reply = asyncio.create_task(self._speak(transcript.text))

    async def _speak(self, text: str) -> None:
        self._set(State.SPEAKING)
        result = await self._tts.speak(self._respond(text))
        self._emit("reply", result=result)
        if not result.interrupted:  # barge-in already moved us to LISTENING
            self._set(State.LISTENING, self._cfg.followup_s)

    def _is_echo(self) -> bool:
        if self._echo is None or not self._echo.is_echo():
            return False
        self._emit("echo_ignored")
        log.debug("speech during playback matched the speaker output (%.2f): ignored", self._echo.last_similarity)
        return True

    def _barge_in(self) -> None:
        self._tts.interrupt()  # cancels token consumption + synthesis, flushes the speaker
        self._set(State.LISTENING)  # keep capturing: the interrupting speech is the next turn
        self._emit("barge_in")
        log.info("barge-in")

    # ---- helpers ------------------------------------------------------------------------------

    def _set(self, state: State, timeout_s: float | None = None) -> None:
        self.state = state
        self._deadline = time.monotonic() + timeout_s if timeout_s and self._wake else None
        self._apply_vad_profile()
        self._emit("state")

    def _apply_vad_profile(self) -> None:
        speaking = self.state is State.SPEAKING
        self._seg.threshold = self._cfg.barge_in_threshold if speaking else self._cfg.idle_threshold
        self._seg.min_speech_ms = self._cfg.barge_in_min_ms if speaking else self._cfg.idle_min_ms

    def _emit(self, kind: str, **kw: object) -> None:
        self.events.put_nowait(TurnEvent(kind, self.state, at=time.monotonic(), **kw))  # type: ignore[arg-type]
