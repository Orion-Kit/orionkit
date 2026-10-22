"""Sentence-streaming TTS: LLM tokens -> sentence chunks -> Kokoro-82M (ONNX) -> PCM at the speaker.

Sentence N+1 is synthesized while sentence N plays, so time-to-first-audio is one short sentence.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .mic import _OPEN, SAMPLE_RATE, STALL_S, RingBuffer, refresh_devices

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
log = logging.getLogger("orionkit.tts")
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
        for _, clean in self.feed_pairs(token):
            if clean:
                yield clean

    def flush(self) -> Iterator[str]:
        for _, clean in self.flush_pairs():
            if clean:
                yield clean

    def feed_pairs(self, token: str) -> Iterator[tuple[str, str]]:
        """Like feed(), but yields (original text, text to speak); the original keeps links, units and spacing for display."""
        self._buf += token
        while (cut := self._find_cut()) is not None:
            raw, self._buf = self._buf[:cut], self._buf[cut:]
            self._first = False
            yield raw, _clean(raw)

    def flush_pairs(self) -> Iterator[tuple[str, str]]:
        raw, self._buf = self._buf, ""
        if raw:
            yield raw, _clean(raw)

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

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE, device: Optional[int | str | bool] = None, block_ms: int = 20,
                 capture: bool = False) -> None:
        """``device=False`` plays to nobody locally: blocks go to ``sink`` (a remote browser) and/or ``captured`` (tests)."""
        self.sample_rate = sample_rate
        self._device = device
        self._block = sample_rate * block_ms // 1000
        self._chunks: deque[np.ndarray] = deque()
        self._on_start: deque[Optional[Callable[[], None]]] = deque()  # parallel to _chunks: run when that chunk starts playing
        self._offset = 0
        self._lock = threading.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream = None
        self._headless: Optional[asyncio.Task[None]] = None
        self.captured: list[np.ndarray] = []
        self._capture = capture or device is False
        self.sink: Optional[Callable[[np.ndarray], None]] = None      # receives every played block (remote playback)
        self.on_flush: Optional[Callable[[], None]] = None           # told when playback is cut (barge-in)
        self.rms = 0.0  # level of the most recent output block
        # What actually went to the speaker, at the mic's 16 kHz, for echo detection.
        self.reference = RingBuffer(int(SAMPLE_RATE * 2.0))
        self.paused = False
        self._last_pull = 0.0
        self._watchdog: Optional[asyncio.Task[None]] = None

    @property
    def playing(self) -> bool:
        return not self._idle.is_set()

    @property
    def audible(self) -> bool:
        return self.playing and not self.paused

    def pause(self) -> None:
        """Go silent within one block but keep the queue: resume() carries on from the same sample."""
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def start(self) -> None:
        # scipy costs ~0.9 s to import: TTSEngine loads it in its loader thread, so it's cached by now
        from scipy.signal import resample_poly
        self._resample = resample_poly
        self._loop = asyncio.get_running_loop()
        if self._device is False:
            self._headless = asyncio.create_task(self._headless_clock())
            return
        import sounddevice as sd
        self._open(sd)
        _OPEN.add(self)
        self._watchdog = asyncio.create_task(self._watch(sd))

    def _open(self, sd: Any) -> None:
        def stream(rate: int, callback: Callable) -> object:
            return sd.OutputStream(samplerate=rate, channels=1, dtype="float32", blocksize=self._block * rate // self.sample_rate,
                                   latency="low", device=self._device, callback=callback)
        try:
            self._stream = stream(self.sample_rate, self._callback)
        except sd.PortAudioError:  # many Windows/USB outputs are 44.1/48 kHz only: play at their rate, resampled
            native = int(sd.query_devices(self._device, "output")["default_samplerate"])
            if native == self.sample_rate:
                raise
            g = math.gcd(self.sample_rate, native)
            up, down = native // g, self.sample_rate // g

            def resampled(outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
                y = self._resample(self._pull(max(1, round(frames * self.sample_rate / native))), up, down)
                outdata[:, 0] = np.pad(y[:frames], (0, max(0, frames - len(y))))
            self._stream = stream(native, resampled)
            log.info("speaker can't play %d Hz; playing at %d Hz, resampled", self.sample_rate, native)
        self._last_pull = time.monotonic()
        self._stream.start()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            for step in (stream.stop, stream.close):
                with contextlib.suppress(Exception):  # a device that's gone can fail to stop; carry on regardless
                    step()

    async def _watch(self, sd: Any) -> None:
        """Bluetooth headphones switched off: the stream stops asking for audio and a reply would wait forever to
        finish. Re-scan the devices and carry on through the new default output."""
        while self in _OPEN:
            await asyncio.sleep(1.0)
            if self not in _OPEN or time.monotonic() - self._last_pull < STALL_S:
                continue
            log.warning("the speaker stopped playing for %.0f s; re-scanning devices", time.monotonic() - self._last_pull)
            failed = await asyncio.to_thread(refresh_devices, sd)
            if self in failed:
                self._last_pull = time.monotonic()  # retry after another STALL_S
                log.error("no speaker to play on: %s", failed[self])
                self.flush()  # nothing can play this, so don't leave the reply waiting for it

    async def stop(self) -> None:
        self.flush()
        _OPEN.discard(self)
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self._headless:
            self._headless.cancel()
        self._close_stream()

    def enqueue(self, pcm: np.ndarray, on_start: Optional[Callable[[], None]] = None) -> None:
        """Queue audio; ``on_start`` runs on the event loop when its first sample reaches the speaker."""
        with self._lock:
            self._chunks.append(pcm.astype(np.float32, copy=False))
            self._on_start.append(on_start)
        self._idle.clear()

    def flush(self) -> None:
        """Drop everything queued; the speaker goes silent within one block (~20 ms)."""
        with self._lock:
            self._chunks.clear()
            self._on_start.clear()  # a flushed chunk was never heard, so its start callback must never run
            self._offset = 0
        self.paused = False
        self._idle.set()
        if self.on_flush:
            self.on_flush()

    async def drain(self) -> None:
        await self._idle.wait()

    def _pull(self, n: int) -> np.ndarray:
        self._last_pull = time.monotonic()
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks and not self.paused:
                head = self._chunks[0]
                if self._offset == 0 and self._on_start[0] is not None:
                    callback, self._on_start[0] = self._on_start[0], None
                    if self._loop is not None:
                        self._loop.call_soon_threadsafe(callback)
                take = min(n - filled, len(head) - self._offset)
                out[filled:filled + take] = head[self._offset:self._offset + take]
                filled += take
                self._offset += take
                if self._offset >= len(head):
                    self._chunks.popleft()
                    self._on_start.popleft()
                    self._offset = 0
            empty = not self._chunks and not self.paused
        self.rms = float(np.sqrt(np.mean(out ** 2))) if filled else 0.0
        self.reference.write((np.clip(self._resample(out, 2, 3), -1, 1) * 32767).astype(np.int16))
        if empty and self._loop is not None:
            self._loop.call_soon_threadsafe(self._set_idle_if_empty)
        return out

    def _set_idle_if_empty(self) -> None:
        """Runs on the loop: audio queued since the audio thread saw an empty queue keeps drain() waiting."""
        with self._lock:
            if not self._chunks:
                self._idle.set()

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        outdata[:, 0] = self._pull(frames)

    async def _headless_clock(self) -> None:
        period = self._block / self.sample_rate
        while True:
            await asyncio.sleep(period)
            playing = self.playing
            block = self._pull(self._block)  # always pull, like a sound card, so the echo reference stays current
            if playing:
                if self._capture and self.sink is None:  # tests; a remote player streams instead of keeping audio
                    self.captured.append(block)
                if self.sink:
                    self.sink(block)


@dataclass(frozen=True)
class SpeakResult:
    interrupted: bool
    sentences: int
    first_audio_ms: Optional[float]  # first token in -> first PCM queued
    first_audio_at: Optional[float] = None  # time.monotonic() when that first PCM was queued


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
        import scipy.signal  # noqa: F401  (imported here, in the loader thread, for AudioPlayer.start)
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(model_path), str(voices_path))
        try:
            self._voice = self._blend(voice)
        except Exception:  # a typo in ORIONKIT_VOICE must not cost the user their voice
            log.warning("unknown voice %r, using the default", voice)
            self._voice = self._blend(DEFAULT_VOICE)
        self._lang = lang
        self._speed = speed
        self._player = player
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._job: Optional[asyncio.Task[SpeakResult]] = None
        self._interrupted = False
        self._captions: Optional[tuple[Callable[[str], None], Callable[[bool], None]]] = None
        self._said = ""   # every token of the current reply
        self._shown = ""  # the part already captioned
        self.synthesize("Ready.")  # warm-up

    def voices(self) -> list[str]:
        return list(self._kokoro.get_voices())

    def configure(self, voice: str | None = None, speed: float | None = None) -> None:
        """Change voice and speed from the next sentence on."""
        if voice:
            self._voice = self._blend(voice)
        if speed:
            self._speed = speed

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

    def pause(self) -> None:
        """Someone may be interrupting: silence the speaker but keep generating, so resume() loses nothing."""
        self._player.pause()

    def resume(self) -> None:
        self._player.resume()

    @property
    def paused(self) -> bool:
        return self._player.paused

    def interrupt(self) -> None:
        """Barge-in: stop consuming tokens, abandon pending synthesis, silence the speaker."""
        self._interrupted = True
        if self._job is not None:
            self._job.cancel()
        self._player.flush()

    def caption(self, on_text: Callable[[str], None], on_end: Callable[[bool], None]) -> None:
        """Show the reply being spoken in step with the voice: each chunk of original text goes to ``on_text`` the moment
        its audio starts playing, and ``on_end(interrupted)`` runs once playback has finished. Lasts for the current reply."""
        self._captions = (on_text, on_end)

    def _reveal(self, text: str) -> None:
        if self._captions and text:
            self._shown += text
            self._captions[0](text)

    async def speak(self, tokens: AsyncIterable[str] | str) -> SpeakResult:
        """Speak a token stream (or a string) and return once playback ends or is interrupted."""
        self._interrupted = False
        self._captions, self._said, self._shown = None, "", ""
        self._job = asyncio.create_task(self._run(_as_stream(tokens) if isinstance(tokens, str) else tokens))
        interrupted = True
        try:
            result = await self._job
            interrupted = result.interrupted
            return result
        except asyncio.CancelledError:
            if not self._interrupted:
                raise
            return SpeakResult(True, 0, None)
        finally:
            if self._captions:
                if not interrupted:  # a chunk TTS couldn't voice, or an error: the reader still gets every word
                    self._reveal(self._said[len(self._shown):])
                on_end = self._captions[1]
                self._captions = None
                on_end(interrupted)

    async def _run(self, tokens: AsyncIterable[str]) -> SpeakResult:
        loop = asyncio.get_running_loop()
        sentences: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()
        started: list[float] = []

        async def produce() -> None:
            splitter = SentenceSplitter()
            try:
                async for token in tokens:
                    if not started:
                        started.append(time.perf_counter())
                    self._said += token
                    for pair in splitter.feed_pairs(token):
                        sentences.put_nowait(pair)
                for pair in splitter.flush_pairs():
                    sentences.put_nowait(pair)
            finally:
                sentences.put_nowait(None)
                if hasattr(tokens, "aclose"):
                    await tokens.aclose()  # halts the upstream generator (e.g. the LLM stream)

        producer = asyncio.create_task(produce())
        count, first_ms, first_at = 0, None, None
        pending = ""  # original text of chunks with nothing to say: shown together with the next chunk that's spoken
        try:
            while (item := await sentences.get()) is not None:
                raw, sentence = item
                pending += raw
                if not re.search(r"\w", sentence):  # "---", "-", "…": nothing to say, and Kokoro rejects them
                    continue
                try:
                    pcm = await loop.run_in_executor(self._pool, self.synthesize, sentence)
                except ValueError as e:  # one unpronounceable chunk must not end the whole reply
                    log.warning("skipped a %d-character chunk TTS couldn't voice (%s)", len(sentence), e)
                    continue
                text, pending = pending, ""
                self._player.enqueue(pcm, on_start=lambda text=text: self._reveal(text))
                count += 1
                if first_ms is None and started:
                    first_ms, first_at = (time.perf_counter() - started[0]) * 1000, time.monotonic()
            await self._player.drain()
            return SpeakResult(False, count, first_ms, first_at)
        finally:
            producer.cancel()


async def _as_stream(text: str) -> AsyncIterator[str]:
    for word in re.findall(r"\S+\s*", text):
        yield word
