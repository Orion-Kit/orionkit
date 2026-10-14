#!/usr/bin/env bash
# Fetch the local voice models into ./models (about 450 MB; Whisper downloads itself on first run).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models/kws
get() { [ -s "$2" ] && echo "✓ $2" || { echo "↓ $2"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }; }
get https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx models/silero_vad.onnx
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx models/kokoro-v1.0.int8.onnx
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin models/voices-v1.0.bin
if [ ! -s models/kws/tokens.txt ]; then
  echo '↓ "Hey Orion" keyword spotter'
  curl -fL --retry 3 https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2 \
    | tar -xj -C models/kws --strip-components=1
fi
echo "Models ready."
