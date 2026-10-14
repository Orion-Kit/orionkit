"""Sentence-streaming TTS: LLM tokens -> sentence chunks -> Kokoro-82M (ONNX) -> PCM at the speaker.

Sentence N+1 is synthesized while sentence N plays, so time-to-first-audio is one short sentence.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from .mic import SAMPLE_RATE, RingBuffer

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
TTS_SAMPLE_RATE = 24_000
DEFAULT_VOICE = "bm_george:0.55,bm_fable:0.45"  # Orion's British blend

_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "prof", "inc", "etc", "e.g", "i.e", "no", "approx"})
_BOUNDARY = re.compile(r"[.!?…]+[\"')\]]*\s+|\n+")
_CLAUSE = re.compile(r"[,;:—]\s+")
_MARKDOWN = re.compile(r"[*_#`>]+")


class SentenceSplitter:
    """Incremental splitter for a token stream.

    The first chunk may break at a clause (after ``first_clause_chars``) so speech starts sooner;
    later chunks break at sentence ends. Anything over ``max_chars`` breaks at the last space.
    """

    def __init__(self, first_clause_chars: int = 8, max_chars: int = 240, first_words_chars: int = 24) -> None:
        self._buf = ""
        self._first = True
        self._first_clause_chars = first_clause_chars
        self._first_words_chars = first_words_chars
        self._max_chars = max_chars

    def feed(self, token: str) -> Iterator[str]:
        self._buf += token
        while (cut := self._find_cut()) is not None:
            chunk, self._buf = self._buf[:cut], self._buf[cut:]
            self._first = False
            if chunk := _clean(chunk):
                yield chunk

    def flush(self) -> Iterator[str]:
        chunk, self._buf = _clean(self._buf), ""
        if chunk:
            yield chunk

    def _find_cut(self) -> Optional[int]:
        for m in _BOUNDARY.finditer(self._buf):
            if m.group().startswith("\n") or not self._is_abbreviation(m.start()):
                return m.end()
        if self._first and len(self._buf) >= self._first_clause_chars:
            for m in _CLAUSE.finditer(self._buf, self._first_clause_chars - 1):
                return m.end()
            # No clause break yet in a long opening sentence: speak the first few words now.
            # Synthesis time grows with length (~150 ms per second of audio), so this is what gets audio out fast.
            if len(self._buf) >= self._first_words_chars + 12:
                space = self._buf.rfind(" ", 0, self._first_words_chars + 1)
                if space > 0 and len(self._buf[:space].split()) >= 4:
                    return space + 1
        if len(self._buf) > self._max_chars:
            space = self._buf.rfind(" ", 0, self._max_chars)
            return space + 1 if space > 0 else self._max_chars
        return None

    def _is_abbreviation(self, dot: int) -> bool:
        word = self._buf[:dot].rsplit(None, 1)[-1].lower() if self._buf[:dot].strip() else ""
        return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


_SPOKEN_UNITS = ((re.compile(r"\s?°\s?F\b"), " degrees Fahrenheit"), (re.compile(r"\s?°\s?C\b"), " degrees Celsius"),
                 (re.compile(r"°"), " degrees"), (re.compile(r"(?<=\d)\s?mph\b"), " miles per hour"),
                 (re.compile(r"(?<=\d)\s?km/h\b"), " kilometers per hour"))


_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)?[^)\s]+\)")
_URL = re.compile(r"https?://(?:www\.)?([^/\s)]+)[^\s)]*")


def _clean(text: str) -> str:
    text = _LINK.sub(r"\1", text)   # [Python.org](https://...) -> Python.org
    text = _URL.sub(r"\1", text)    # bare URLs -> just the site name
    text = _MARKDOWN.sub("", text)
    for pattern, spoken in _SPOKEN_UNITS:  # say units the way a person would
        text = pattern.sub(spoken, text)
    return " ".join(text.split())


class AudioPlayer:
    """Low-latency PCM sink whose queue can be flushed instantly (barge-in).

    ``device=False`` runs headless: audio is consumed in real time and kept in ``captured`` (tests).
    """

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE, device: Optional[int | str | bool] = None, block_ms: int = 20) -> None:
        self.sample_rate = sample_rate
        self._device = device
        self._block = sample_rate * block_ms // 1000
        self._chunks: deque[np.ndarray] = deque()
        self._offset = 0
        self._lock = threading.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream = None
        self._headless: Optional[asyncio.Task[None]] = None
        self.captured: list[np.ndarray] = []
        self.rms = 0.0  # level of the most recent output block
        # What actually went to the speaker, at the mic's 16 kHz, for echo detection.
        self.reference = RingBuffer(int(SAMPLE_RATE * 2.0))

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._device is False:
            self._headless = asyncio.create_task(self._headless_clock())
            return
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self._block, latency="low", device=self._device, callback=self._callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        self.flush()
        if self._headless:
            self._headless.cancel()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def enqueue(self, pcm: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(pcm.astype(np.float32, copy=False))
        self._idle.clear()

    def flush(self) -> None:
        """Drop everything queued; the speaker goes silent within one block (~20 ms)."""
        with self._lock:
            self._chunks.clear()
            self._offset = 0
        self._idle.set()

    async def drain(self) -> None:
        await self._idle.wait()

    def _pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                head = self._chunks[0]
                take = min(n - filled, len(head) - self._offset)
                out[filled:filled + take] = head[self._offset:self._offset + take]
                filled += take
                self._offset += take
                if self._offset >= len(head):
                    self._chunks.popleft()
                    self._offset = 0
            empty = not self._chunks
        self.rms = float(np.sqrt(np.mean(out ** 2))) if filled else 0.0
        self.reference.write((np.clip(resample_poly(out, 2, 3), -1, 1) * 32767).astype(np.int16))
        if empty and self._loop is not None:
            self._loop.call_soon_threadsafe(self._idle.set)
        return out

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        outdata[:, 0] = self._pull(frames)

    async def _headless_clock(self) -> None:
        period = self._block / self.sample_rate
        while True:
            await asyncio.sleep(period)
            playing = self.playing
            block = self._pull(self._block)  # always pull, like a sound card, so the echo reference stays current
            if playing:
                self.captured.append(block)


@dataclass(frozen=True)
class SpeakResult:
    interrupted: bool
    sentences: int
    first_audio_ms: Optional[float]  # first token in -> first PCM queued


class TTSEngine:
    def __init__(
        self,
        player: AudioPlayer,
        voice: str = DEFAULT_VOICE,
        lang: str = "en-gb",
        speed: float = 1.0,
        model_path: Path = MODELS_DIR / "kokoro-v1.0.int8.onnx",
        voices_path: Path = MODELS_DIR / "voices-v1.0.bin",
    ) -> None:
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self._voice = self._blend(voice)
        self._lang = lang
        self._speed = speed
        self._player = player
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._job: Optional[asyncio.Task[SpeakResult]] = None
        self._interrupted = False
        self.synthesize("Ready.")  # warm-up

    def _blend(self, spec: str) -> np.ndarray:
        """"bm_george:0.55,bm_fable:0.45" -> weighted style vector; a bare name means weight 1."""
        style: np.ndarray | float = 0.0
        for part in spec.split(","):
            name, _, weight = part.partition(":")
            style = style + float(weight or 1) * self._kokoro.get_voice_style(name.strip())
        return np.asarray(style, dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        audio, sr = self._kokoro.create(text, voice=self._voice, speed=self._speed, lang=self._lang)
        assert sr == TTS_SAMPLE_RATE
        return audio

    def chime(self, kind: str) -> None:
        """Short soft two-note cue: rising when Orion starts listening, falling when it stops."""
        if self._player.playing:
            return
        notes = (660.0, 990.0) if kind == "start" else (880.0, 587.0)
        sr = TTS_SAMPLE_RATE
        n = int(sr * 0.07)
        t = np.arange(n) / sr
        env = np.sin(np.pi * np.arange(n) / n) ** 2  # smooth attack/release, no clicks
        tone = np.concatenate([0.12 * np.sin(2 * np.pi * f * t) * env for f in notes]).astype(np.float32)
        self._player.enqueue(tone)

    def interrupt(self) -> None:
        """Barge-in: stop consuming tokens, abandon pending synthesis, silence the speaker."""
        self._interrupted = True
        if self._job is not None:
            self._job.cancel()
        self._player.flush()

    async def speak(self, tokens: AsyncIterable[str] | str) -> SpeakResult:
        """Speak a token stream (or a string) and return once playback ends or is interrupted."""
        self._interrupted = False
        self._job = asyncio.create_task(self._run(_as_stream(tokens) if isinstance(tokens, str) else tokens))
        try:
            return await self._job
        except asyncio.CancelledError:
            if not self._interrupted:
                raise
            return SpeakResult(True, 0, None)

    async def _run(self, tokens: AsyncIterable[str]) -> SpeakResult:
        loop = asyncio.get_running_loop()
        sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()
        started: list[float] = []

        async def produce() -> None:
            splitter = SentenceSplitter()
            try:
                async for token in tokens:
                    if not started:
                        started.append(time.perf_counter())
                    for s in splitter.feed(token):
                        sentences.put_nowait(s)
                for s in splitter.flush():
                    sentences.put_nowait(s)
            finally:
                sentences.put_nowait(None)
                if hasattr(tokens, "aclose"):
                    await tokens.aclose()  # halts the upstream generator (e.g. the LLM stream)

        producer = asyncio.create_task(produce())
        count, first_ms = 0, None
        try:
            while (sentence := await sentences.get()) is not None:
                pcm = await loop.run_in_executor(self._pool, self.synthesize, sentence)
                self._player.enqueue(pcm)
                count += 1
                if first_ms is None and started:
                    first_ms = (time.perf_counter() - started[0]) * 1000
            await self._player.drain()
            return SpeakResult(False, count, first_ms)
        finally:
            producer.cancel()


async def _as_stream(text: str) -> AsyncIterator[str]:
    for word in re.findall(r"\S+\s*", text):
        yield word
