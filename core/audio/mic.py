"""Shared microphone capture: one PortAudio input stream fanned out to asyncio subscribers."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Optional

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms: Silero VAD's native frame at 16 kHz
STALL_S = 2.5        # no callback for this long while running: the device went away (unplugged, sleep, switched)
log = logging.getLogger("orionkit.mic")


class MicError(RuntimeError):
    """No usable microphone; the message says what to do."""


class RingBuffer:
    """Fixed-size circular buffer of int16 samples (the 'pre-roll' of recent audio)."""

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.int16)
        self._pos = 0
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        with self._lock:
            n = len(samples)
            cap = len(self._buf)
            if n >= cap:
                self._buf[:] = samples[-cap:]
                self._pos, self._filled = 0, cap
                return
            end = self._pos + n
            if end <= cap:
                self._buf[self._pos:end] = samples
            else:
                split = cap - self._pos
                self._buf[self._pos:] = samples[:split]
                self._buf[: n - split] = samples[split:]
            self._pos = end % cap
            self._filled = min(cap, self._filled + n)

    def latest(self, n: int) -> np.ndarray:
        with self._lock:
            n = min(n, self._filled)
            idx = (np.arange(self._pos - n, self._pos)) % len(self._buf)
            return self._buf[idx].copy()


class MicStream:
    """16 kHz mono int16 capture. Each subscriber gets every 512-sample frame on its own queue.

    Pass ``device=False`` to run without hardware and push audio with :meth:`feed` (tests).
    """

    def __init__(
        self,
        device: Optional[int | str | bool] = None,
        preroll_s: float = 3.0,
        queue_frames: int = 256,
    ) -> None:
        self._device = device
        self._ring = RingBuffer(int(SAMPLE_RATE * preroll_s))
        self._subs: list[asyncio.Queue[np.ndarray]] = []
        self._queue_frames = queue_frames
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream = None
        self._carry = np.zeros(0, dtype=np.int16)
        self.rms: float = 0.0  # level of the most recent frame (0..1)
        self.peak: float = 0.0  # loudest frame so far: 0.0 means no audio at all is arriving
        self.on_silent: Callable[[str], None] | None = None
        self.muted = False     # privacy mute: capture continues, but everyone downstream hears silence
        self._rate = SAMPLE_RATE   # the device's rate; resampled to 16 kHz when it can't capture at 16 kHz
        self._last_audio = 0.0
        self._watchdog: Optional[asyncio.Task[None]] = None

    def subscribe(self) -> asyncio.Queue[np.ndarray]:
        q: asyncio.Queue[np.ndarray] = asyncio.Queue(self._queue_frames)
        self._subs.append(q)
        return q

    def recent(self, seconds: float) -> np.ndarray:
        return self._ring.latest(int(SAMPLE_RATE * seconds))

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._device is False:
            return
        import sounddevice as sd

        try:
            self._open(sd)
        except sd.PortAudioError as e:
            raise MicError(f"No microphone available ({e}). Pick one with --input N (see python main.py "
                           "--list-devices), or use --text for typed chat.") from e
        self._loop.call_later(6, self._check_silent)
        self._watchdog = asyncio.create_task(self._watch(sd))

    def _open(self, sd: object) -> None:
        """16 kHz if the device does it; otherwise its own rate, resampled (many USB and Windows mics are 44.1/48 kHz only)."""
        def stream(rate: int) -> object:
            return sd.InputStream(samplerate=rate, channels=1, dtype="int16", blocksize=FRAME_SAMPLES * rate // SAMPLE_RATE,
                                  latency="low", device=self._device, callback=self._callback)
        try:
            s, self._rate = stream(SAMPLE_RATE), SAMPLE_RATE
        except sd.PortAudioError:
            native = int(sd.query_devices(self._device, "input")["default_samplerate"])
            if native == SAMPLE_RATE:
                raise
            s, self._rate = stream(native), native
            log.info("microphone can't capture at 16 kHz; capturing at %d Hz and resampling", native)
        self._last_audio = time.monotonic()
        s.start()
        self._stream = s

    async def _watch(self, sd: object) -> None:
        """A stream whose device disappears just stops calling back: reopen it, and tell the user if that fails."""
        told = False
        while self._stream is not None:
            await asyncio.sleep(1.0)
            if self._stream is None or time.monotonic() - self._last_audio < STALL_S:
                continue
            log.warning("no audio from the microphone for %.0f s; reopening it", time.monotonic() - self._last_audio)
            with contextlib.suppress(Exception):
                self._stream.close()
            try:
                self._open(sd)
                told = False
            except Exception as e:  # PortAudioError, or the device index no longer exists
                self._last_audio = time.monotonic()  # retry after another STALL_S
                if not told and self.on_silent:
                    self.on_silent("The microphone stopped sending audio (unplugged?). Plug it back in, or restart Orion to "
                                   f"use another one. Typing still works. ({type(e).__name__})")
                told = True

    def _check_silent(self) -> None:
        """macOS gives silence, not an error, when the terminal isn't allowed to use the mic."""
        if self._stream is not None and self.peak == 0.0 and self.on_silent:
            self.on_silent("The microphone is completely silent. On macOS, allow your terminal app in System Settings > "
                           "Privacy & Security > Microphone, then restart it. Typing still works.")

    async def stop(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self._stream is not None:
            stream, self._stream = self._stream, None
            with contextlib.suppress(Exception):  # an unplugged device can fail to stop; shutting down must still finish
                stream.stop()
            with contextlib.suppress(Exception):
                stream.close()

    async def __aenter__(self) -> "MicStream":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    def feed(self, samples: np.ndarray) -> None:
        """Push int16 audio as if it came from the microphone (any length; reframed to 512)."""
        self._ingest(np.asarray(samples, dtype=np.int16))

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        assert self._loop is not None
        self._last_audio = time.monotonic()
        samples = indata[:, 0].copy()
        if self._rate != SAMPLE_RATE:
            g = math.gcd(SAMPLE_RATE, self._rate)
            samples = np.clip(self._resample(samples.astype(np.float32), SAMPLE_RATE // g, self._rate // g), -32768, 32767
                              ).astype(np.int16)
        self._loop.call_soon_threadsafe(self._ingest, samples)

    @staticmethod
    def _resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
        from scipy.signal import resample_poly
        return resample_poly(x, up, down)

    def _ingest(self, samples: np.ndarray) -> None:
        if self.muted:
            samples = np.zeros_like(samples)
        self._ring.write(samples)
        data = np.concatenate([self._carry, samples]) if len(self._carry) else samples
        n_frames = len(data) // FRAME_SAMPLES
        for i in range(n_frames):
            frame = data[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            self.rms = float(np.sqrt(np.mean((frame.astype(np.float32) / 32768.0) ** 2)))
            self.peak = max(self.peak, self.rms)
            for q in self._subs:
                if q.full():  # a slow consumer drops its oldest audio rather than blocking capture
                    q.get_nowait()
                q.put_nowait(frame)
        self._carry = data[n_frames * FRAME_SAMPLES:].copy()

