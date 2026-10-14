"""Echo detection for barge-in without hardware echo cancellation.

When Orion talks through laptop speakers the mic hears it, and the VAD calls that speech. This gate
compares what the mic heard with what the speaker just played: band-energy envelopes (robust to room
reverb, speaker distortion and gain) are correlated over plausible acoustic delays. A high correlation
means the mic is mostly hearing playback; a real interruption adds energy the speaker didn't produce
and the correlation drops.
"""
from __future__ import annotations

import numpy as np

from .mic import SAMPLE_RATE, MicStream
from .tts import AudioPlayer

FRAME = 512          # 32 ms analysis window
HOP = 256            # 16 ms hop
BANDS_HZ = ((150, 600), (600, 1800), (1800, 4500))


def band_envelopes(x: np.ndarray) -> np.ndarray:
    """log band energies, shape (frames, bands)."""
    x = x.astype(np.float32)
    n = 1 + max(0, (len(x) - FRAME) // HOP)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(FRAME), axis=1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
    bands = [spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) for lo, hi in BANDS_HZ]
    return np.log(np.stack(bands, axis=1) + 1e-3)


def echo_similarity(mic: np.ndarray, ref: np.ndarray, max_lag_s: float = 0.3) -> float:
    """Best envelope correlation of the mic window against the reference shifted 0..max_lag earlier.

    ``ref`` must cover the same period as ``mic`` plus ``max_lag_s`` before it.
    """
    m = band_envelopes(mic)
    r = band_envelopes(ref)
    max_lag = int(max_lag_s * SAMPLE_RATE / HOP)
    if len(m) < 8 or len(r) < len(m):
        return 0.0
    mz = (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-6)
    best = -1.0
    end = len(r)
    for lag in range(0, min(max_lag, len(r) - len(m)) + 1):
        seg = r[end - lag - len(m): end - lag]
        rz = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        best = max(best, float((mz * rz).mean()))
    return best


class EchoGate:
    # max_lag 0.5 s: a MacBook Pro measured ~225 ms speaker-to-mic delay (CoreAudio input latency ~315 ms).
    def __init__(self, mic: MicStream, player: AudioPlayer, window_s: float = 0.6, max_lag_s: float = 0.5, threshold: float = 0.72) -> None:
        self._mic, self._player = mic, player
        self._window_s, self._max_lag_s = window_s, max_lag_s
        self.threshold = threshold
        self.last_similarity = 0.0
        self.last_levels = (0.0, 0.0)  # (mic rms, speaker rms) of the last check
        self.borderline = 0.5           # below the threshold but above this, judge by loudness too
        self.louder_than_echo = 1.6     # a real interruption is at least this much louder than the echo
        self._coupling: float | None = None

    def is_echo(self) -> bool:
        """True when the recent mic audio is explained by recent playback."""
        mic = self._mic.recent(self._window_s)
        ref = self._player.reference.latest(int(SAMPLE_RATE * (self._window_s + self._max_lag_s)))
        if np.sqrt(np.mean((ref.astype(np.float32) / 32768) ** 2)) < 1e-3:
            self.last_similarity = 0.0
            return False  # nothing was playing: whatever the mic hears is not our echo
        self.last_similarity = sim = echo_similarity(mic, ref, self._max_lag_s)
        mic_rms = float(np.sqrt(np.mean((mic.astype(np.float32) / 32768) ** 2)))
        ref_rms = float(np.sqrt(np.mean((ref.astype(np.float32) / 32768) ** 2)))
        self.last_levels = (mic_rms, ref_rms)
        ratio = mic_rms / max(ref_rms, 1e-4)
        if sim >= self.threshold:
            # Confident echo: learn how loud the speaker sounds at the mic (room, volume and mic dependent).
            self._coupling = ratio if self._coupling is None else 0.9 * self._coupling + 0.1 * ratio
            return True
        if sim >= self.borderline and self._coupling is not None:
            # Borderline match: someone talking over Orion makes the mic louder than its echo alone would.
            return ratio <= self._coupling * self.louder_than_echo
        return False
