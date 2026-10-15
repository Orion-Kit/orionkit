"""Semantic end-of-turn detection: has the speaker finished, or only paused?

Pipecat's Smart Turn v3 (BSD-2, ~8 MB int8 ONNX: a Whisper-tiny encoder with a small classifier head) listens to
the last 8 s of the utterance and returns P(turn complete) in ~10-60 ms on a CPU. The segmenter asks it after a
short pause: "complete" ends the turn early (faster replies), "incomplete" keeps listening through a mid-thought
pause. Without the model file every call answers None and the segmenter keeps its fixed-silence rule.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("orionkit.turn")

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "smart_turn" / "smart-turn-v3.2-cpu.onnx"
WINDOW = 8 * 16_000  # the model was trained on the last 8 s, left-padded with silence


class SmartTurn:
    def __init__(self, model_path: Path = DEFAULT_MODEL) -> None:
        self.model_path = model_path
        self.threshold = float(os.environ.get("ORIONKIT_TURN_THRESHOLD", "0.5") or 0.5)
        self._session = None
        self._features = None
        self._lock = threading.Lock()
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed and self.model_path.is_file()

    def _load(self) -> None:
        import onnxruntime as ort
        from faster_whisper.feature_extractor import FeatureExtractor

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        self._features = FeatureExtractor(chunk_length=8)

    def probability(self, audio: np.ndarray) -> float | None:
        """P(the speaker is done) for float32 16 kHz audio; None if the model isn't there or failed."""
        if not self.available:
            return None
        try:
            with self._lock:
                if self._session is None:
                    self._load()
                x = audio[-WINDOW:].astype(np.float32)
                x = np.pad(x, (WINDOW - len(x), 0))
                x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # zero-mean, unit-variance, as in training
                feats = self._features(x, padding=0)[None].astype(np.float32)
                return float(self._session.run(None, {"input_features": feats})[0][0][0])
        except Exception:
            log.exception("smart turn failed; falling back to fixed silence")
            self._failed = True
            return None
