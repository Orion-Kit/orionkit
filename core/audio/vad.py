"""Silero VAD (ONNX, no torch) plus a segmenter that turns frames into clean speech utterances."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Union

import numpy as np

from .mic import FRAME_SAMPLES, SAMPLE_RATE, MicStream
from .turn import SmartTurn

log = logging.getLogger("orionkit.vad")
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "silero_vad.onnx"
CONTEXT_SAMPLES = 64  # Silero v5 expects the last 64 samples of the previous frame prepended


class SileroVAD:
    """Stateful per-frame speech probability. Feed consecutive 512-sample 16 kHz frames."""

    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = frame.astype(np.float32).reshape(1, -1)
        if frame.dtype == np.int16:
            x /= 32768.0
        x = np.concatenate([self._context, x], axis=1)
        prob, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(prob[0][0])


class VADEventKind(Enum):
    SPEECH_START = auto()
    SPEECH_END = auto()


@dataclass(frozen=True)
class SpeechStart:
    at: float = field(default_factory=time.monotonic)
    kind: VADEventKind = VADEventKind.SPEECH_START


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray  # float32 mono 16 kHz, including pre/post padding
    ended_at: float = field(default_factory=time.monotonic)
    kind: VADEventKind = VADEventKind.SPEECH_END

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


VADEvent = Union[SpeechStart, Utterance]


class SpeechSegmenter:
    """Hysteresis state machine over Silero probabilities.

    Emits ``SpeechStart`` once ``min_speech_ms`` of speech is confirmed (the barge-in signal) and an
    ``Utterance`` when the speaker is done. Ambient silence never leaves this class. ``threshold`` can be raised
    at runtime (duplex does this while TTS plays, to resist echo).

    "Done" is decided by Smart Turn when its model is installed: after ``check_ms`` of silence it asks whether the
    turn sounds finished; yes ends it right away, no keeps listening (re-asking every ``recheck_ms``) up to
    ``max_silence_ms``, so a mid-thought pause doesn't cut you off. Without the model: ``min_silence_ms`` of silence.
    """

    def __init__(
        self,
        mic: MicStream,
        vad: SileroVAD | None = None,
        threshold: float = 0.5,
        min_speech_ms: int = 160,
        min_silence_ms: int = 400,
        pad_ms: int = 200,
        max_utterance_s: float = 20.0,
        turn: SmartTurn | None = None,
        check_ms: int = 200,
        recheck_ms: int = 500,
        max_silence_ms: int | None = None,
    ) -> None:
        self._frames = mic.subscribe()
        self._vad = vad or SileroVAD()
        self.threshold = threshold
        self._ms_per_frame = FRAME_SAMPLES * 1000 / SAMPLE_RATE
        self.min_speech_ms = min_speech_ms
        self._end_frames = max(1, round(min_silence_ms / self._ms_per_frame))
        self._pad_frames = max(0, round(pad_ms / self._ms_per_frame))
        self._max_frames = int(max_utterance_s * 1000 / self._ms_per_frame)
        self._max_hold_frames = int(60 * 1000 / self._ms_per_frame)
        self.turn = turn if turn is not None else SmartTurn()
        if max_silence_ms is None:
            max_silence_ms = int(float(os.environ.get("ORIONKIT_TURN_MAX_SILENCE", "1.6") or 1.6) * 1000)
        self._check_frames = max(1, round(check_ms / self._ms_per_frame))
        self._recheck_frames = max(1, round(recheck_ms / self._ms_per_frame))
        self._max_silence_frames = max(self._check_frames, round(max_silence_ms / self._ms_per_frame))
        self._pool: ThreadPoolExecutor | None = None  # off the event loop: a check takes ~10-60 ms
        self._turn_future: Future[float | None] | None = None
        self.last_turn_p: float | None = None
        self._turn_off = False  # a minute of held speech at most
        self.events: asyncio.Queue[VADEvent] = asyncio.Queue()
        self.last_prob = 0.0
        self.reset()

    @property
    def min_speech_ms(self) -> int:
        return round(self._start_frames * self._ms_per_frame)

    @min_speech_ms.setter
    def min_speech_ms(self, ms: int) -> None:
        self._start_frames = max(1, round(ms / self._ms_per_frame))

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def begin_hold(self) -> None:
        """Push-to-talk: from now on keep every frame (starting with the short pre-roll, so the first syllable is
        there) and emit nothing, however long the pauses, until end_hold()."""
        self._hold = list(self._preroll if not self._in_speech else self._speech)

    def end_hold(self) -> np.ndarray | None:
        """Stop holding; the audio captured since begin_hold() (float32 16 kHz), or None if there was no hold."""
        frames, self._hold = self._hold, None
        self.reset()
        if frames is None:
            return None
        return np.concatenate(frames).astype(np.float32) / 32768.0 if frames else np.zeros(0, np.float32)

    def reset(self) -> None:
        """Drop any partial utterance (used after wake / barge-in to start a clean turn)."""
        self._hold = getattr(self, "_hold", None)  # a hold survives resets (wake word mid-hold); end_hold() ends it
        self._vad.reset()
        self._preroll: list[np.ndarray] = []
        self._speech: list[np.ndarray] = []
        self._in_speech = False
        self._voiced_run = 0
        self._silent_run = 0
        self._turn_future = None
        self._next_check = 0

    async def run(self) -> None:
        while True:
            frame = await self._frames.get()
            for event in self.process(frame):
                await self.events.put(event)

    def process(self, frame: np.ndarray) -> list[VADEvent]:
        if self._hold is not None:  # push-to-talk: the key, not silence, ends the turn
            if len(self._hold) < self._max_hold_frames:
                self._hold.append(frame)
            return []
        prob = self._vad(frame)
        self.last_prob = prob
        # Hysteresis: once inside speech, stay in until prob falls well under the entry threshold.
        voiced = prob >= (self.threshold if not self._in_speech else self.threshold - 0.15)
        out: list[VADEvent] = []

        if not self._in_speech:
            self._preroll.append(frame)
            self._voiced_run = self._voiced_run + 1 if voiced else 0
            keep = self._pad_frames + self._start_frames
            if len(self._preroll) > keep:
                self._preroll = self._preroll[-keep:]
            if self._voiced_run >= self._start_frames:
                self._in_speech = True
                self._speech = list(self._preroll)
                self._silent_run = 0
                out.append(SpeechStart())
            return out

        self._speech.append(frame)
        self._silent_run = 0 if voiced else self._silent_run + 1
        if self._silent_run == 0:
            self._turn_future, self._next_check = None, 0  # still talking: any pending verdict is stale
        if self._turn_done() or len(self._speech) >= self._max_frames:
            # Keep pad_ms of trailing silence; trim the rest so STT sees clean speech.
            trim = max(0, self._silent_run - self._pad_frames)
            frames = self._speech[: len(self._speech) - trim] if trim else self._speech
            audio = np.concatenate(frames).astype(np.float32) / 32768.0
            out.append(Utterance(audio))
            self._in_speech = False
            self._speech, self._preroll = [], []
            self._voiced_run = self._silent_run = 0
            self._turn_future, self._next_check = None, 0
        return out

    def _turn_done(self) -> bool:
        """Is the current utterance over? Smart Turn when available, else the fixed silence rule."""
        if self._turn_off or not self.turn.available:
            return self._silent_run >= self._end_frames
        if self._silent_run >= self._max_silence_frames:
            return True
        if self._turn_future is not None and self._turn_future.done():
            p, self._turn_future = self._turn_future.result(), None
            self.last_turn_p = p
            if p is None:  # the model failed: from now on, the fixed rule
                self._turn_off = True
                return self._silent_run >= self._end_frames
            log.debug("smart turn p=%.2f after %d ms of silence", p, self._silent_run * self._ms_per_frame)
            if p >= self.turn.threshold:
                return True
            self._next_check = self._silent_run + self._recheck_frames
        if self._turn_future is None and self._silent_run >= max(self._check_frames, self._next_check):
            audio = np.concatenate(self._speech).astype(np.float32) / 32768.0
            self._pool = self._pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="smart-turn")
            self._turn_future = self._pool.submit(self.turn.probability, audio)
            self._next_check = self._silent_run + self._recheck_frames
        return False
