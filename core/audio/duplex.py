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
import re
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
    # Second-chance wake: short phrases heard while idle are transcribed and checked for the name.
    # Catches accents and mics the keyword spotter misses; also lets "Hey Orion, <request>" run in one go.
    stt_wake: bool = True
    stt_wake_max_s: float = 6.0
    stt_wake_pattern: str = (r"^\W*(?:(?:hey|hi|hello|ok(?:ay)?|hail|yo)[\s,]+)?"
                             r"(?:orion|orian|o'?ryan|oreon|aryan)(?:'s|s)?\b[\s,.!?]*(.*)$")


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
                self._barge_in("wake word")
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
                    self._barge_in("speech")
                elif self.state is State.LISTENING:
                    self._deadline = None  # user is talking; don't time out mid-sentence
            elif isinstance(event, Utterance) and self.state is State.LISTENING:
                await self._handle_utterance(event)
            elif isinstance(event, Utterance) and self.state is State.IDLE and self._cfg.stt_wake \
                    and event.duration_s <= self._cfg.stt_wake_max_s:
                await self._stt_wake(event)

    async def _timeout_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self.state is State.LISTENING and self._deadline and time.monotonic() > self._deadline:
                self._set(State.IDLE)
                self._emit("timeout")

    # ---- turn handling ------------------------------------------------------------------------

    async def _stt_wake(self, utt: Utterance) -> None:
        transcript = await self._stt.transcribe(utt.audio)
        m = re.match(self._cfg.stt_wake_pattern, transcript.text, re.I)
        if not m or self.state is not State.IDLE:  # not addressed to us, or the keyword spotter already woke
            return
        self._emit("wake")
        log.info("wake via transcript %r", transcript.text)
        request = m.group(1).strip()
        if len(request.split()) >= 2:  # "Hey Orion, what's the weather?": answer straight away
            rest = Transcript(request, transcript.latency_ms, transcript.audio_s)
            self._emit("transcript", transcript=rest)
            self._reply = asyncio.create_task(self._speak(request))
        else:
            self._seg.reset()
            self._set(State.LISTENING, self._cfg.listen_timeout_s)

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

    async def announce(self, text: str) -> None:
        """Speak something Orion starts on its own (a timer going off), after any reply in progress."""
        while self.state in (State.THINKING, State.SPEAKING):
            await asyncio.sleep(0.2)
        self._emit("announce")
        self._set(State.SPEAKING)
        result = await self._tts.speak(text)
        if not result.interrupted:
            self._set(State.LISTENING, self._cfg.followup_s)

    def _is_echo(self) -> bool:
        if self._echo is None or not self._echo.is_echo():
            return False
        self._emit("echo_ignored")
        log.debug("speech during playback matched the speaker output (%.2f): ignored", self._echo.last_similarity)
        return True

    def _barge_in(self, cause: str = "speech") -> None:
        sim = self._echo.last_similarity if self._echo else float("nan")
        self._tts.interrupt()  # cancels token consumption + synthesis, flushes the speaker
        self._set(State.LISTENING)  # keep capturing: the interrupting speech is the next turn
        self._emit("barge_in")
        log.info("barge-in by %s (echo similarity %.2f, vad %.2f)", cause, sim, self._seg.last_prob)

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
