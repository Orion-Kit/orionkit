#!/usr/bin/env python3
"""Download Orion's voice models (about 290 MB with Whisper base.en) into models/. Works on macOS, Windows and Linux.

Safe to run again at any time: finished files are checked and skipped (a damaged one is downloaded again), a download
that was cut off carries on from where it stopped (slow or flaky connections, a laptop going to sleep), and every new file is checked against its known
SHA-256 before it's used, so a broken or tampered download is never loaded.

  python scripts/download_models.py              voice models + Whisper base.en
  python scripts/download_models.py small.en     a different Whisper size (tiny.en, base.en, small.en)
"""
from __future__ import annotations

import hashlib
import http.client
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"
GH = "https://github.com"
# name: (url, bytes, sha256). Pinned to tagged releases / commits, so an upstream change can't swap a model under you
# (silero-vad's master branch has changed its model twice since v5).
FILES = {
    "silero_vad.onnx": (f"{GH}/snakers4/silero-vad/raw/v6.2.1/src/silero_vad/data/silero_vad.onnx", 2_327_524,
                        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"),
    "kokoro-v1.0.int8.onnx": (f"{GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
                              92_361_271, "6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb"),
    "voices-v1.0.bin": (f"{GH}/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", 28_214_398,
                        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"),
    # Smart Turn v3.2 (BSD-2): knows when you've finished talking, not just paused
    "smart_turn/smart-turn-v3.2-cpu.onnx": (
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx",
        8_679_182, "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"),
}
KWS = (f"{GH}/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2",
       17_626_723, "f170013b4716e41b62b9bfd809687c207cef798ef9bc6534d524e17af9b6561a")
WHISPER_MB = {"tiny.en": 75, "base.en": 145, "small.en": 485}
ATTEMPTS = 8


class DownloadError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _progress(name: str, done: int, total: int, started: float, resumed_from: int) -> None:
    rate = (done - resumed_from) / max(time.monotonic() - started, 1e-3)
    sys.stdout.write(f"\r  {name}: {done / 1e6:6.1f} of {total / 1e6:.1f} MB ({100 * done / total:3.0f}%)  "
                     f"{rate / 1e6:5.2f} MB/s ")
    sys.stdout.flush()


def fetch(url: str, dest: Path, size: int, sha256: str, attempts: int = ATTEMPTS, wait_s: float = 2.0) -> None:
    """Download ``url`` to ``dest``, resuming ``dest.part`` with HTTP Range requests after any interruption."""
    if dest.is_file() and dest.stat().st_size:
        print(f"✓ {dest.name}")
        return
    part = dest.with_name(dest.name + ".part")
    print(f"↓ {dest.name} ({size / 1e6:.0f} MB)")
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have > size:  # not ours, or a different version of the file: start again
            part.unlink()
            have = 0
        if have < size:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    if have and r.status != 206:  # the server ignored the range: it's sending the whole file
                        have = 0
                    started = time.monotonic()
                    with open(part, "ab" if have else "wb") as f:
                        done = have
                        while block := r.read(1 << 16):
                            f.write(block)
                            done += len(block)
                            _progress(dest.name, done, size, started, have)
                print()
            except (urllib.error.URLError, OSError, http.client.HTTPException) as e:  # dropped Wi-Fi, sleep, a proxy
                if getattr(e, "errno", None) == 28:
                    raise DownloadError("The disk is full. Free some space and run this again; it carries on from "
                                        "where it stopped.") from e
                if isinstance(e, urllib.error.HTTPError) and e.code == 416:  # range past the end: the part is stale
                    part.unlink(missing_ok=True)
                kept = part.stat().st_size if part.is_file() else 0
                print(f"\n  interrupted at {kept / 1e6:.1f} MB ({type(e).__name__}: {getattr(e, 'reason', e)}); "
                      f"retrying ({attempt}/{attempts})…")
                time.sleep(min(30.0, wait_s * attempt))
                continue
        kept = part.stat().st_size if part.is_file() else 0
        if kept < size:  # the connection closed early without an error: carry on from here
            print(f"  cut off at {kept / 1e6:.1f} MB; retrying ({attempt}/{attempts})…")
            time.sleep(min(30.0, wait_s * attempt))
            continue
        if kept == size and _sha256(part) == sha256:
            part.replace(dest)
            return
        print(f"  {dest.name} didn't match its checksum; downloading it again ({attempt}/{attempts})…")
        part.unlink(missing_ok=True)
    raise DownloadError(f"Couldn't download {dest.name} from {url.split('/')[2]}. Check your connection (a proxy or "
                        "firewall may block it), then run this again: what's already downloaded is kept.")


def damaged(models: Path | None = None) -> list[str]:
    """Files that are present but not the file we pinned (a disk error, an old interrupted download, an edit). Before,
    present files were never checked, so running this script again could not repair one: voice failed to start with an
    onnxruntime error and the only fix was deleting the file by hand."""
    models = models or MODELS
    return [n for n, (_, size, sha) in FILES.items()
            if (models / n).is_file() and ((models / n).stat().st_size != size or _sha256(models / n) != sha)]


def _missing() -> tuple[list[tuple[str, tuple[str, int, str]]], bool]:
    for name in damaged():
        print(f"✗ {name} is damaged; downloading it again")
        (MODELS / name).unlink()
    files = [(n, spec) for n, spec in FILES.items() if not ((MODELS / n).is_file() and (MODELS / n).stat().st_size)]
    return files, not (MODELS / "kws" / "tokens.txt").is_file()


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    sizes = sys.argv[1:] or ["base.en"]
    files, need_kws = _missing()
    total = sum(spec[1] for _, spec in files) + (KWS[1] if need_kws else 0)
    whisper_mb = sum(WHISPER_MB.get(s, 0) for s in sizes)
    print(f"Up to {total / 1e6 + whisper_mb:.0f} MB to download (voice {total / 1e6:.0f} MB, Whisper {whisper_mb} MB) "
          "into models/. Safe to stop and run again: it carries on where it left off.")
    need = total + KWS[1] * need_kws + whisper_mb * 1e6 + 100e6  # the keyword archive is unpacked next to itself
    if (free := shutil.disk_usage(MODELS).free) < need:
        print(f"Only {free / 1e9:.2f} GB free here; the models need about {need / 1e9:.2f} GB. Free some space and run "
              "this again.")
        return 1
    try:
        for name, (url, size, sha) in files:
            (MODELS / name).parent.mkdir(parents=True, exist_ok=True)
            fetch(url, MODELS / name, size, sha)
        if need_kws:
            kws, archive = MODELS / "kws", MODELS / "kws.tar.bz2"
            fetch(KWS[0], archive, KWS[1], KWS[2])
            kws.mkdir(exist_ok=True)
            with tarfile.open(archive, "r:bz2") as t:
                for m in t.getmembers():
                    parts = Path(m.name).parts[1:]  # strip the top folder, like tar --strip-components=1
                    if not parts or m.issym() or m.islnk() or ".." in parts:
                        continue
                    m.name = str(Path(*parts))
                    t.extract(m, kws, filter="data")
            archive.unlink()
    except DownloadError as e:
        print(f"\n{e}")
        return 1
    try:  # speech recognition: otherwise it downloads on the first voice start, which fails offline or behind a proxy
        import httpx
        from faster_whisper.utils import download_model
        from huggingface_hub.errors import HfHubHTTPError
        network_errors: tuple[type[BaseException], ...] = (OSError, HfHubHTTPError, httpx.HTTPError)
        try:  # huggingface_hub 2.x talks through its own httpx fork
            import httpx2
            network_errors += (httpx2.HTTPError,)
        except ImportError:
            pass
    except ImportError:
        print("Whisper downloads on first run (install the requirements first to fetch it now).")
    else:
        for size in sizes:
            print(f"↓ Whisper {size} (about {WHISPER_MB.get(size, '?')} MB; resumes if interrupted)")
            for attempt in range(1, 4):  # the first try after an interrupted download can fail (IncompleteSnapshotError)
                try:
                    download_model(size, cache_dir=str(MODELS / "whisper"))
                    break
                except network_errors as e:
                    if attempt == 3:
                        print(f"Couldn't download Whisper {size} from huggingface.co ({type(e).__name__}). Check your "
                              "connection (a proxy or firewall may block huggingface.co), then run this again.")
                        return 1
                    print(f"  interrupted ({type(e).__name__}); retrying ({attempt}/3)…")
                    time.sleep(3 * attempt)
    print("Models ready. Voice now works with no internet connection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
