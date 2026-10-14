"""Wake-word detection over the shared circular microphone stream.

Two backends, same interface (``run()`` + an ``events`` queue of ``WakeEvent``):

- ``KeywordWakeDetector`` (default): sherpa-onnx open-vocabulary keyword spotter. "Hey Orion" works
  out of the box because the phrase is spelled in BPE tokens, no training needed.
- ``WakeWordDetector``: openWakeWord. Use a pretrained model name (e.g. ``hey_jarvis``) or a custom
  ``.onnx`` trained with openWakeWord's notebook.

``create_wake("hey orion")`` picks the backend from the spec.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mic import MicStream

OWW_CHUNK = 1280  # openWakeWord scores 80 ms chunks
DEFAULT_MODEL = "hey orion"
KWS_DIR = Path(__file__).resolve().parents[2] / "models" / "kws"
KWS_FILE = "{}-epoch-12-avg-2-chunk-16-left-64.int8.onnx"


@dataclass(frozen=True)
class WakeEvent:
    model: str
    score: float
    at: float = field(default_factory=time.monotonic)


class WakeWordDetector:
    def __init__(
        self,
        mic: MicStream,
        model: str | Path = DEFAULT_MODEL,
        threshold: float = 0.5,
        cooldown_s: float = 1.5,
    ) -> None:
        from openwakeword.model import Model
        from openwakeword.utils import download_models

        model = str(model)
        if not model.endswith((".onnx", ".tflite")):
            download_models([model])  # no-op once cached
        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self._frames = mic.subscribe()
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self.events: asyncio.Queue[WakeEvent] = asyncio.Queue()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        pending = np.zeros(0, dtype=np.int16)
        last_fire = 0.0
        while True:
            pending = np.concatenate([pending, await self._frames.get()])
            while len(pending) >= OWW_CHUNK:
                chunk, pending = pending[:OWW_CHUNK], pending[OWW_CHUNK:]
                scores: dict[str, float] = await loop.run_in_executor(None, self._model.predict, chunk)
                name, score = max(scores.items(), key=lambda kv: kv[1])
                now = time.monotonic()
                if score >= self._threshold and now - last_fire >= self._cooldown_s:
                    last_fire = now
                    self._model.reset()  # clear the feature buffer so one utterance fires once
                    await self.events.put(WakeEvent(name, float(score)))


class KeywordWakeDetector:
    """Streaming open-vocabulary keyword spotting (sherpa-onnx zipformer, ~3 MB, Apache-2.0)."""

    def __init__(
        self,
        mic: MicStream,
        phrase: str = DEFAULT_MODEL,
        threshold: float = 0.2,  # 10/10 synthetic voices, 0/100 lookalike phrases at 0.15-0.2
        boost: float = 1.5,
        cooldown_s: float = 1.5,
        model_dir: Path = KWS_DIR,
    ) -> None:
        import tempfile

        import sentencepiece as spm
        import sherpa_onnx

        spellings = self.spellings(phrase, spm.SentencePieceProcessor(model_file=str(model_dir / "bpe.model")))
        tag = "_".join(phrase.upper().split())
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(f"{s} @{tag}" for s in spellings) + "\n")
            keywords_file = f.name
        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(model_dir / "tokens.txt"),
            encoder=str(model_dir / KWS_FILE.format("encoder")),
            decoder=str(model_dir / KWS_FILE.format("decoder")),
            joiner=str(model_dir / KWS_FILE.format("joiner")),
            keywords_file=keywords_file,
            num_threads=1,
            keywords_score=boost,
            keywords_threshold=threshold,
        )
        Path(keywords_file).unlink(missing_ok=True)
        self._stream = self._kws.create_stream()
        self._frames = mic.subscribe()
        # The spotter misses the phrase after a long run of unrelated speech, so each utterance gets a
        # fresh stream: reset after ~0.5 s of quiet relative to an adaptive noise floor.
        self._noise_floor = 0.01
        self._quiet_frames = 0
        self._heard_since_reset = False
        self._cooldown_s = cooldown_s
        self._name = phrase.lower()
        self.events: asyncio.Queue[WakeEvent] = asyncio.Queue()

    @staticmethod
    def spellings(phrase: str, sp: object) -> list[str]:
        """BPE token spellings of the phrase, plus the common mishearing of the name ("Orian")."""
        variants = {phrase.upper(), phrase.upper().replace("ORION", "ORIAN")}
        return sorted({" ".join(sp.encode_as_pieces(v)) for v in variants})  # type: ignore[attr-defined]

    def _feed(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x)))
        self._noise_floor = min(rms, self._noise_floor * 1.002) if rms > 1e-5 else self._noise_floor
        if rms < max(0.003, 2.5 * self._noise_floor):
            self._quiet_frames += 1
            if self._quiet_frames == 16 and self._heard_since_reset:  # 512 ms
                self._kws.reset_stream(self._stream)
                self._heard_since_reset = False
        else:
            self._quiet_frames = 0
            self._heard_since_reset = True
        self._stream.accept_waveform(16_000, x)
        hit = False
        while self._kws.is_ready(self._stream):
            self._kws.decode_stream(self._stream)
            if self._kws.get_result(self._stream):
                hit = True
                self._kws.reset_stream(self._stream)
        return hit

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        last_fire = 0.0
        while True:
            frame = await self._frames.get()
            if await loop.run_in_executor(None, self._feed, frame):
                now = time.monotonic()
                if now - last_fire >= self._cooldown_s:
                    last_fire = now
                    await self.events.put(WakeEvent(self._name, 1.0))


def create_wake(mic: MicStream, spec: str = DEFAULT_MODEL, threshold: float | None = None) -> "KeywordWakeDetector | WakeWordDetector":
    """A phrase ("hey orion") uses the keyword spotter; a model name or .onnx/.tflite path uses openWakeWord."""
    if spec.endswith((".onnx", ".tflite")) or "_" in spec:
        return WakeWordDetector(mic, spec, 0.5 if threshold is None else threshold)
    return KeywordWakeDetector(mic, spec, 0.2 if threshold is None else threshold)
