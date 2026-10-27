"""Local Faster-Whisper worker. Transcribes VAD utterances on a dedicated thread (CTranslate2 int8)."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "whisper"


@dataclass(frozen=True)
class Transcript:
    text: str
    latency_ms: float  # wall time inside Whisper for this utterance
    audio_s: float


class STTWorker:
    """``base.en`` int8 measures ~225 ms for a 3 s utterance on an M1 Pro; ``tiny.en`` ~125 ms."""

    def __init__(
        self,
        model: str = "base.en",
        compute_type: str | None = None,
        cpu_threads: int = 4,
        language: str = "en",
        device: str = "auto",
        hotwords: str = "Orion, OrionKit",
    ) -> None:
        from faster_whisper import WhisperModel

        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        self.device = device
        self._model = WhisperModel(
            local_model(model), device=device, compute_type=compute_type or ("float16" if device == "cuda" else "int8"),
            cpu_threads=cpu_threads, download_root=str(MODELS_DIR),
        )
        self._language = language
        self._hotwords = hotwords  # names Whisper otherwise mishears ("Orion" -> "Aryans")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")  # model is not re-entrant
        self._transcribe_sync(np.zeros(16_000, dtype=np.float32))  # warm-up: first call pays graph init

    def _transcribe_sync(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            without_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=False,  # Silero already trimmed the audio
            hotwords=self._hotwords or None,
        )
        # Drop Whisper's classic hallucinations on non-speech ("Thank you.", "you").
        text = " ".join(
            s.text.strip() for s in segments
            if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)
        ).strip()
        return Transcript(text, (time.perf_counter() - t0) * 1000, len(audio) / 16_000)

    async def transcribe(self, audio: np.ndarray) -> Transcript:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._transcribe_sync, audio)


def local_model(model: str) -> str:
    """The downloaded copy of ``model`` if there is one, else ``model`` (which downloads it).

    Given only a name, faster-whisper asks huggingface.co for the latest revision on every start, even when the model
    is already here: that sends a request each time, and a captive portal or a network that drops packets made voice
    hang for minutes (a proxy that cut the connection crashed it). A model that's been downloaded never needs the network."""
    if "/" in model or Path(model).exists():
        return model
    from faster_whisper.utils import download_model
    try:
        return download_model(model, local_files_only=True, cache_dir=str(MODELS_DIR))
    except FileNotFoundError:  # not downloaded yet (huggingface_hub's LocalEntryNotFoundError): fetch it now
        return model


def _cuda_available() -> bool:
    """NVIDIA GPU usable by CTranslate2 (Windows/Linux RTX). Apple Silicon has no CUDA: CPU int8 is the fast path."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
