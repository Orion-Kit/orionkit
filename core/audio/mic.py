"""Shared microphone capture: one PortAudio input stream fanned out to asyncio subscribers."""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Optional

import numpy as np

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512  # 32 ms: Silero VAD's native frame at 16 kHz


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
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=FRAME_SAMPLES,
                latency="low",
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            raise MicError(f"No microphone available ({e}). Pick one with --input N (see python main.py "
                           "--list-devices), or use --text for typed chat.") from e
        self._loop.call_later(6, self._check_silent)

    def _check_silent(self) -> None:
        """macOS gives silence, not an error, when the terminal isn't allowed to use the mic."""
        if self._stream is not None and self.peak == 0.0 and self.on_silent:
            self.on_silent("The microphone is completely silent. On macOS, allow your terminal app in System Settings > "
                           "Privacy & Security > Microphone, then restart it. Typing still works.")

    async def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
        self._loop.call_soon_threadsafe(self._ingest, indata[:, 0].copy())

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
