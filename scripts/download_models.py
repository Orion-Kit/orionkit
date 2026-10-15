#!/usr/bin/env python3
"""Download Orion's voice models (~450 MB) into models/. Same as download_models.sh, but works on Windows too."""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
FILES = {
    "silero_vad.onnx": "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    # Smart Turn v3.2 (BSD-2, ~8 MB): knows when you've finished talking, not just paused
    "smart_turn/smart-turn-v3.2-cpu.onnx": "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
}
KWS = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
       "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2")


def fetch(url: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size:
        print(f"✓ {dest.name}")
        return
    print(f"↓ {dest.name}")
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.replace(dest)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    for name, url in FILES.items():
        (MODELS / name).parent.mkdir(parents=True, exist_ok=True)
        fetch(url, MODELS / name)
    kws = MODELS / "kws"
    if not (kws / "tokens.txt").is_file():
        archive = MODELS / "kws.tar.bz2"
        fetch(KWS, archive)
        kws.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:bz2") as t:
            for m in t.getmembers():
                parts = Path(m.name).parts[1:]  # strip the top folder, like tar --strip-components=1
                if not parts or m.issym() or m.islnk() or ".." in parts:
                    continue
                m.name = str(Path(*parts))
                t.extract(m, kws, filter="data")
        archive.unlink()
    print("Models ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
