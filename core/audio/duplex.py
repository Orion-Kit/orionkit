"""Duplex turn controller with barge-in.

States: IDLE (waiting for the wake word) -> LISTENING -> THINKING -> SPEAKING -> LISTENING (follow-up
window) -> IDLE. If confirmed user speech arrives while SPEAKING, Orion goes quiet at once (the speaker pauses,
generation carries on) and listens to what was said: a real request replaces the reply and becomes the next turn,
"stop" ends it, and a cough, "mm-hmm" or "yeah" (or its own voice) resumes the reply where it paused. The wake
word, tap-to-talk and push-to-talk always interrupt outright.

Echo: without hardware echo cancellation the mic hears the speaker. While SPEAKING the VAD threshold
and minimum speech length are raised, and an optional ``EchoGate`` rejects "speech" that is explained
by what the speaker just played, so Orion's own voice does not trigger barge-in.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum

from .echo import EchoGate
from .stt import STTWorker, Transcript
from .tts import SpeakResult, TTSEngine
from .vad import SpeechSegmenter, SpeechStart, Utterance
from .wake import KeywordWakeDetector, WakeEvent, WakeWordDetector, sensitivity

log = logging.getLogger("orionkit.duplex")

# Listener noises while Orion talks: not a new request, so the reply carries on.
BACKCHANNEL = re.compile(r"^\W*(m+[- ]?h*m+|mhm+|hmm+|uh[- ]?huh|uh+|um+|ah+|oh+|yeah|yep|yes|ok(ay)?|right|sure|cool|nice|"
                         r"got it|i see|wow|interesting|go on)([\s,.!]+(yeah|ok(ay)?|right|sure|mhm))*\W*$", re.I)
# "Stop" / "wait" while Orion talks: stop the reply and listen for what comes next (not the end of the conversation).
HOLD_ON = re.compile(r"^\W*(stop|wait|hold on|hang on|one (sec(ond)?|moment)|just a (sec(ond)?|moment))\W*$", re.I)
# "That wasn't for you": the wake word fired by mistake. Back to idle, and counted (opt-in diagnostics) as a correction.
NOT_FOR_YOU = re.compile(r"^\W*(no,? )?(that|it|this) (was ?n'?t|was not|is ?n'?t|is not) (meant )?(for|to) you\W*$|"
                         r"^\W*(no,? )?(i was ?n'?t|i was not|i'?m not) talking to you\W*$|^\W*not (you|for you)\W*$", re.I)

Responder = Callable[[str], AsyncIterator[str]]  # transcript -> streamed reply tokens


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class DuplexConfig:
    listen_timeout_s: float = 8.0   # after wake, give up if nobody speaks
    # Follow-up mode (off by default): one fixed window after a reply where the next request needs no wake word.
    # Background speech can't hold it open, and follow-ups are only answered if they're meant for Orion.
    followup: bool = False
    followup_s: float = 5.0
    chime: bool = True              # soft tones when Orion starts and stops listening
    end_pattern: str = (r"^\W*(thanks?( you)?( so much)?|cheers|that'?s (all|it|everything)|never ?mind|nothing|no thanks?|"
                        r"stop|cancel|goodbye|bye|that will be all|i'?m good|all good)\b[\w\s,.!']{0,20}$")
    barge_in_threshold: float = 0.8  # VAD threshold while Orion is talking
    barge_in_min_ms: int = 300       # confirmed speech needed to interrupt
    idle_threshold: float = 0.5
    idle_min_ms: int = 160
    # Second-chance wake: short phrases heard while idle are transcribed and checked for the name.
    # Catches accents and mics the keyword spotter misses; also lets "Hey Orion, <request>" run in one go.
    # Off with ORIONKIT_WAKE_SENSITIVITY=low: in a room with a TV it would transcribe every phrase looking for the name.
    stt_wake: bool = field(default_factory=lambda: sensitivity() != "low")
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
    detail: str = ""  # wake: what woke it (wake word, transcript, tap, push-to-talk); timeout: why it went idle


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
        addressed: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self._seg = segmenter
        self._stt = stt
        self._tts = tts
        self._respond = respond
        self._wake = wake
        self._cfg = config
        self._echo = echo_gate
        self._addressed = addressed
        self.followup = config.followup  # can be toggled at runtime (UI)
        self._in_followup = False  # LISTENING because a reply just ended, not because of the wake word
        self.state = State.IDLE if wake else State.LISTENING
        self.events: asyncio.Queue[TurnEvent] = asyncio.Queue()
        self._deadline: float | None = None
        self._reply: asyncio.Task[None] | None = None
        self._ptt_since: float | None = None   # push-to-talk key held since (monotonic), else None
        self._paused_at: float | None = None   # speaker paused for a possible interruption
        self._deciding = False                 # transcribing that interruption
        self._reply_text = ""                  # what the current reply has said so far (to spot echo)
        self._ptt_task: asyncio.Task[None] | None = None
        self._apply_vad_profile()

    async def run(self) -> None:
        tasks = [asyncio.create_task(self._seg.run()), asyncio.create_task(self._vad_loop()),
                 asyncio.create_task(self._pause_watchdog())]
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
                    self._start_listening()
                else:
                    self._in_followup = False  # the wake word makes a follow-up explicit
            self._emit("wake", detail="wake word")
            log.info("wake %s %.2f", event.model, event.score)

    async def _vad_loop(self) -> None:
        while True:
            event = await self._seg.events.get()
            log.debug("vad %s in %s", type(event).__name__, self.state.value)
            if isinstance(event, SpeechStart):
                if self.state is State.SPEAKING and not self._speaker_paused:
                    if self._is_echo():
                        # Drop the echo segment so continued real speech raises a fresh SpeechStart.
                        self._seg.reset()
                        continue
                    if hasattr(self._tts, "pause"):
                        self._pause_for_interruption()
                    else:  # a TTS that can't pause: interrupt outright, as before
                        self._barge_in("speech")
                # while LISTENING the timeout loop waits for an utterance in progress to finish (no deadline reset)
                # In a follow-up window the deadline stays fixed: background chatter can't keep it open.
            elif isinstance(event, Utterance) and self.state is State.SPEAKING and self._speaker_paused:
                await self._decide_interruption(event)
            elif isinstance(event, Utterance) and self.state is State.LISTENING:
                await self._contained(self._handle_utterance(event))
            elif isinstance(event, Utterance) and self.state is State.IDLE and self._cfg.stt_wake \
                    and event.duration_s <= self._cfg.stt_wake_max_s:
                await self._contained(self._stt_wake(event))

    async def _contained(self, turn: Awaitable[None]) -> None:
        """One failed turn (a transcription error, say) must not end the loop: that would leave Orion deaf
        until restart. Log it and go back to a clean state."""
        try:
            await turn
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("handling an utterance failed")
            if self.state is State.THINKING:
                self._stop_listening("couldn't transcribe")

    async def _timeout_loop(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self.state is State.LISTENING and self._deadline and time.monotonic() > self._deadline and not self.ptt_held:
                if self._seg.in_speech:
                    continue  # they're mid-sentence: let them finish (background chatter ends too, then we time out)
                self._stop_listening("timeout")

    # ---- turn handling ------------------------------------------------------------------------

    async def _stt_wake(self, utt: Utterance) -> None:
        transcript = await self._stt.transcribe(utt.audio)
        m = re.match(self._cfg.stt_wake_pattern, transcript.text, re.I)
        if not m or self.state is not State.IDLE:  # not addressed to us, or the keyword spotter already woke
            return
        self._emit("wake", detail="transcript")
        log.info("wake via transcript (%d words)", len(transcript.text.split()))
        request = m.group(1).strip()
        if len(request.split()) >= 2:  # "Hey Orion, what's the weather?": answer straight away
            rest = Transcript(request, transcript.latency_ms, transcript.audio_s)
            self._emit("transcript", transcript=rest)
            self._start_reply(request)
        else:
            self._seg.reset()
            self._start_listening()

    async def _handle_utterance(self, utt: Utterance) -> None:
        followup = self._in_followup
        self._set(State.THINKING)
        transcript = await self._stt.transcribe(utt.audio)
        if self.state is not State.THINKING:  # stopped, or a typed request took over while we transcribed
            return
        text = transcript.text.strip()
        if not text:
            self._stop_listening("nothing heard")
            return
        if NOT_FOR_YOU.match(text):
            self._stop_listening("not for you")
            return
        if re.match(self._cfg.end_pattern, text, re.I):
            self._emit("transcript", transcript=transcript)
            self._stop_listening("dismissed")
            return
        if followup and self._addressed is not None and not await self._addressed(text):
            if self.state is State.THINKING:
                log.info("follow-up not meant for Orion, ignored (%d words)", len(text.split()))
                self._stop_listening("not addressed")
            return
        if self.state is not State.THINKING:
            return
        self._emit("transcript", transcript=transcript)
        self._start_reply(text)

    def talk(self) -> None:
        """Tap-to-talk (UI button / orb click): like saying the wake word."""
        if self.state is State.SPEAKING:
            self._barge_in("tap")
        elif self.state is State.IDLE:
            self._seg.reset()
            self._start_listening()
        self._emit("wake", detail="tap")

    # ---- push-to-talk ---------------------------------------------------------------------------------

    PTT_TAP_S = 0.25  # shorter than this is a tap: same as the talk button

    def ptt_start(self) -> None:
        """Key pressed: interrupt any reply and listen for as long as the key is held (no timeout, pauses allowed)."""
        if self._ptt_since is not None:  # key auto-repeat
            return
        self._ptt_since = time.monotonic()
        if self.state is State.SPEAKING:
            self._tts.interrupt()
        elif self.state is State.IDLE:
            self._chime("start")
        self._in_followup = False
        self._set(State.LISTENING)  # no deadline while held; a THINKING turn in flight sees this and stands down
        self._seg.begin_hold()
        self._emit("wake", detail="push-to-talk")

    def ptt_end(self) -> None:
        """Key released: what was said while it was held is the request. A tap just opens a normal listening turn."""
        if self._ptt_since is None:
            return
        held, self._ptt_since = time.monotonic() - self._ptt_since, None
        audio = self._seg.end_hold()
        if self.state is not State.LISTENING:  # something else took over meanwhile (a typed request)
            return
        if held < self.PTT_TAP_S:
            self._set(State.LISTENING, self._cfg.listen_timeout_s)
            return
        if audio is None or len(audio) < 0.3 * 16000:
            self._stop_listening("push-to-talk: nothing held")
            return
        self._ptt_task = asyncio.ensure_future(self._contained(self._handle_utterance(Utterance(audio))))

    @property
    def ptt_held(self) -> bool:
        return self._ptt_since is not None

    def stop(self) -> None:
        """Stop talking now and go back to waiting for the wake word."""
        if self.state is State.SPEAKING:
            self._tts.interrupt()
        self._stop_listening("stopped")

    async def submit_text(self, text: str) -> None:
        """A typed request (from the UI): handled like speech, interrupting any reply in progress."""
        self._emit("transcript", transcript=Transcript(text, 0.0, 0.0))
        self._start_reply(text)

    def _start_reply(self, text: str) -> None:
        """One reply at a time: a new request replaces the one in progress (two would interleave audio and history)."""
        if self.state is State.SPEAKING:  # a reply or an announcement is playing
            self._tts.interrupt()
        if self._reply is not None and not self._reply.done():
            self._reply.cancel()
        self._set(State.SPEAKING)
        self._reply = asyncio.create_task(self._speak(text))

    async def _speak(self, text: str) -> None:
        self._in_followup = False
        self._set(State.SPEAKING)
        self._reply_text = ""

        async def tokens() -> AsyncIterator[str]:
            async with contextlib.aclosing(self._respond(text)) as inner:
                async for token in inner:
                    self._reply_text += token
                    yield token
        try:
            result = await self._tts.speak(tokens())
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("reply failed")
            if self.state is State.SPEAKING:
                self._after_reply()
            return
        self._emit("reply", result=result)
        if not result.interrupted:  # barge-in already moved us to LISTENING
            self._after_reply()

    async def announce(self, text: str) -> None:
        """Speak something Orion starts on its own (a timer going off), after any reply in progress, and never over the
        user: not while they're talking, holding push-to-talk, or in a listening turn they started."""
        while self._busy_for_announcement():
            await asyncio.sleep(0.2)
        self._emit("announce")
        self._reply_text = text
        self._set(State.SPEAKING)
        try:
            result = await self._tts.speak(text)
        except Exception:
            log.exception("announcement failed")
            self._after_reply()
            return
        if not result.interrupted:
            self._after_reply()

    def _busy_for_announcement(self) -> bool:
        if self.state in (State.THINKING, State.SPEAKING) or self.ptt_held or self._seg.in_speech:
            return True
        return self.state is State.LISTENING and self._wake is not None  # a turn the user opened (it times out)

    def _is_echo(self) -> bool:
        if self._echo is None or not self._echo.is_echo():
            return False
        self._emit("echo_ignored")
        log.debug("speech during playback matched the speaker output (%.2f): ignored", self._echo.last_similarity)
        return True

    def _pause_for_interruption(self) -> None:
        """Speech over Orion: go quiet now, decide once we know what was said."""
        self._tts.pause()
        self._paused_at = time.monotonic()
        self._apply_vad_profile()  # the speaker is silent now: no echo to resist, so hear every word
        self._emit("barge_in")
        log.info("paused for an interruption (vad %.2f)", self._seg.last_prob)

    def _resume(self, why: str) -> None:
        self._paused_at = None
        if self.state is State.SPEAKING and self._speaker_paused:
            self._tts.resume()
            self._apply_vad_profile()
            self._emit("resumed")
            log.info("not an interruption (%s): resuming", why)

    async def _decide_interruption(self, utt: Utterance) -> None:
        self._deciding = True
        try:
            transcript = await self._stt.transcribe(utt.audio)
        except Exception:  # never leave the speaker paused (or kill the voice loop) over one bad transcription
            log.exception("transcribing an interruption failed; resuming")
            self._resume("transcription failed")
            return
        finally:
            self._deciding = False
        if self.state is not State.SPEAKING or not self._speaker_paused:
            return  # the reply ended or was replaced while we listened
        self._paused_at = None
        text = transcript.text.strip()
        if not text or BACKCHANNEL.match(text):
            self._resume(repr(text) if text else "no words")
            return
        if self._sounds_like_reply(text):
            self._emit("echo_ignored")
            self._resume("its own voice")
            return
        await self._end_reply()  # a real interruption: the paused reply ends (and reports itself interrupted)
        if NOT_FOR_YOU.match(text):
            self._stop_listening("not for you")
            return
        if HOLD_ON.match(text):  # a control word, like the wake word: not a request of its own
            self._start_listening()
            return
        self._emit("transcript", transcript=transcript)
        if re.match(self._cfg.end_pattern, text, re.I):
            self._stop_listening("dismissed while speaking")
        else:
            self._start_reply(text)

    async def _end_reply(self) -> None:
        self._tts.interrupt()
        if self._reply is not None and not self._reply.done():
            await asyncio.wait({self._reply}, timeout=1.0)

    def _sounds_like_reply(self, text: str) -> bool:
        """Did the mic just hear Orion's own words (echo the gate missed)?"""
        heard = [w.strip(".,!?'\"").lower() for w in text.split()]
        said = {w.strip(".,!?'\"").lower() for w in self._reply_text.split()}
        return len(heard) >= 3 and sum(w in said for w in heard) / len(heard) >= 0.7

    async def _pause_watchdog(self) -> None:
        """A pause that never turned into an utterance (a click, a door) mustn't leave Orion silent."""
        while True:
            await asyncio.sleep(0.2)
            if self._paused_at is None or self._deciding or self._seg.in_speech:
                continue
            if self.state is not State.SPEAKING or not self._speaker_paused:
                self._paused_at = None
            elif time.monotonic() - self._paused_at > 1.5:
                self._resume("no speech followed")

    def _barge_in(self, cause: str = "speech") -> None:
        sim = self._echo.last_similarity if self._echo else float("nan")
        self._tts.interrupt()  # cancels token consumption + synthesis, flushes the speaker
        self._in_followup = False  # interrupting is talking to Orion
        self._set(State.LISTENING, self._cfg.listen_timeout_s)  # keep capturing: the interruption is the next turn
        self._emit("barge_in")
        log.info("barge-in by %s (echo similarity %.2f, vad %.2f)", cause, sim, self._seg.last_prob)

    # ---- listening windows --------------------------------------------------------------------

    def _start_listening(self) -> None:
        self._in_followup = False
        self._set(State.LISTENING, self._cfg.listen_timeout_s)
        self._chime("start")

    def _after_reply(self) -> None:
        if self.followup and self._wake:
            self._in_followup = True
            self._set(State.LISTENING, self._cfg.followup_s)
        else:
            self._stop_listening("reply finished")

    def _stop_listening(self, why: str) -> None:
        self._in_followup = False
        if not self._wake:  # no wake word configured: always listening
            self._set(State.LISTENING)
            return
        self._seg.reset()
        self._set(State.IDLE)
        self._chime("end")
        self._emit("timeout", detail=why)
        log.info("back to wake-word-only (%s)", why)

    def _chime(self, kind: str) -> None:
        if self._cfg.chime:
            self._tts.chime(kind)

    # ---- helpers ------------------------------------------------------------------------------

    def _set(self, state: State, timeout_s: float | None = None) -> None:
        self.state = state
        self._deadline = time.monotonic() + timeout_s if timeout_s and self._wake else None
        self._apply_vad_profile()
        self._emit("state")

    @property
    def _speaker_paused(self) -> bool:
        return bool(getattr(self._tts, "paused", False))  # TTS stand-ins without pause support never pause

    def _apply_vad_profile(self) -> None:
        speaking = self.state is State.SPEAKING and not self._speaker_paused
        self._seg.threshold = self._cfg.barge_in_threshold if speaking else self._cfg.idle_threshold
        self._seg.min_speech_ms = self._cfg.barge_in_min_ms if speaking else self._cfg.idle_min_ms

    def _emit(self, kind: str, **kw: object) -> None:
        self.events.put_nowait(TurnEvent(kind, self.state, at=time.monotonic(), **kw))  # type: ignore[arg-type]
