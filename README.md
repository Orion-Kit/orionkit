# OrionKit Core

The free, MIT-licensed voice engine behind [OrionKit](https://orion-kit.com): a local, low-latency,
full-duplex voice loop in Python.

- **"Hey Orion" wake word** with no training: open-vocabulary keyword spotting (sherpa-onnx). openWakeWord models also work.
- **Silero VAD** with hysteresis and pre-roll, so speech is never clipped.
- **Faster-Whisper** speech-to-text (CPU int8, CUDA when an NVIDIA GPU is present).
- **Kokoro-82M** text-to-speech that streams sentence by sentence while the LLM is still writing.
- **Barge-in**: talk over the assistant and it stops, flushes audio, and listens.
- **Echo rejection**: its own voice through laptop speakers doesn't count as an interruption.
- **Local brain**: any Ollama model, fully offline.

## Quick start (Python 3.11 or 3.12)

Install [Ollama](https://ollama.com/download) and open it. On a Mac the built-in `python3` is too old:
`brew install python@3.12`.

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download_models.py  # ~450 MB of voice models (Windows too)
ollama pull qwen3:4b-instruct                # or any chat model
.venv/bin/python voice_chat.py               # opens the orb window; say "Hey Orion"
.venv/bin/python voice_chat.py --terminal    # prefer the terminal
```

Check everything headlessly (no mic needed):

```sh
.venv/bin/python test_voice.py --self-test
```

## Layout

```
core/audio/mic.py     shared 16 kHz capture with a circular pre-roll buffer
core/audio/wake.py    "Hey Orion" keyword spotter + openWakeWord backend
core/audio/vad.py     Silero VAD and utterance segmenter
core/audio/stt.py     Faster-Whisper worker
core/audio/tts.py     sentence-streaming Kokoro TTS and a flushable audio player
core/audio/echo.py    speaker-echo detection for barge-in
core/audio/duplex.py  listening / thinking / speaking state machine with barge-in
voice_chat.py         voice chat with a local Ollama model
test_voice.py         live harness and headless self-test
```

## OrionKit Pro

Pro adds the tri-model router (Ollama, Gemini, Claude) with circuit breakers and offline failover, the
Nebula MCP agent tools (sandboxed shell, files, telemetry, Home Assistant), screen-aware vision, the live
terminal dashboard, an offline license verifier, and commercial-use rights:
[get OrionKit Pro](https://orion-kit.com/#pricing).

## License

MIT. Third-party models keep their own licenses: Kokoro-82M (Apache-2.0), Silero VAD (MIT),
sherpa-onnx keyword spotter (Apache-2.0), Whisper (MIT), openWakeWord pretrained models (CC BY-NC-SA 4.0,
non-commercial; the default "Hey Orion" spotter doesn't use them).
