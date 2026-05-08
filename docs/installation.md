# Installation

The default supported path is the PyPI package with the default
`faster_whisper` backend:

```bash
pip install RealtimeSTT
```

This installs the core microphone, VAD, wake word, websocket client/server, and
default transcription dependencies listed in `requirements.txt`.

## Python Environment

Use a virtual environment when possible:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install RealtimeSTT
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install RealtimeSTT
```

## Platform Notes

### Linux

Install PortAudio and Python headers before installing PyAudio:

```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev
python -m pip install RealtimeSTT
```

Some examples and tests also use `ffmpeg` or `libsndfile`:

```bash
sudo apt-get install ffmpeg libsndfile1
```

### macOS

Install PortAudio through Homebrew:

```bash
brew install portaudio
python -m pip install RealtimeSTT
```

### Windows

Install from a normal terminal or PowerShell session:

```powershell
python -m pip install RealtimeSTT
```

If a dependency needs a compiler on your machine, install the relevant wheel
package first when one is available. `webrtcvad-wheels` is used by the project
to avoid the older source-only WebRTC VAD install path.

## CUDA Notes

RealtimeSTT can run on CPU with small models, but CUDA is strongly preferred
for low-latency realtime transcription and larger Whisper-family models.

Install the NVIDIA driver, CUDA runtime/toolkit, and cuDNN version that matches
your PyTorch build. Then install a CUDA-enabled PyTorch and torchaudio wheel.
For example:

```bash
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Use the PyTorch install selector for the exact command for your driver and CUDA
version. Keep `device="cuda"` for GPU inference and use `device="cpu"` for CPU
or CPU-only engine stacks.

## Default Dependencies

The current package install includes:

- `faster-whisper` for the default transcription backend.
- `PyAudio` for microphone input.
- `webrtcvad-wheels`, `torch`, `torchaudio`, and `scipy` for VAD/audio paths.
- `pvporcupine` and `openwakeword` for wake word support.
- `websockets` and `websocket-client` for the packaged client/server tools.
- `soundfile` for audio fixture and model-family paths.

Future package extras are planned so users can install smaller engine-specific
sets. Until extras exist, install optional engines explicitly.

## Optional Engine Dependencies

Install only the engine stack you plan to use:

| Engine | Install command | Model handling |
| --- | --- | --- |
| `faster_whisper` | Included by default | Downloads CTranslate2 models automatically through faster-whisper. |
| `whisper_cpp` | `python -m pip install pywhispercpp` | `pywhispercpp` can download known ggml models; local paths are also supported. |
| `openai_whisper` | `python -m pip install openai-whisper` | Downloads OpenAI Whisper models automatically to its cache or `download_root`. |
| `moonshine` | `python -m pip install transformers torch` | Downloads Hugging Face model files automatically. English-only in this adapter. |
| `sherpa_onnx_*` | `python -m pip install sherpa-onnx` | Model bundles must be downloaded and extracted manually. |
| `parakeet` | `python -m pip install -U "nemo_toolkit[asr]" soundfile` | NeMo downloads from the configured model id/cache. Best on Linux or WSL2. |
| `granite_speech` | `python -m pip install transformers torch` | Downloads Hugging Face model files automatically. |
| `qwen3_asr` | `python -m pip install -U qwen-asr` | Downloads Qwen model files through the Qwen ASR package. |
| `cohere_transcribe` | `python -m pip install transformers torch` | Downloads Hugging Face model files; gated model access may be required. |

Per-engine setup lives in [transcription-engines.md](transcription-engines.md)
and the `docs/engines/` pages.

## FastAPI Server Dependencies

For the browser streaming server:

```bash
python -m pip install -r example_fastapi_server/requirements.txt
```

Then install the selected ASR engine stack. See
[fastapi-server.md](fastapi-server.md).

## Verifying The Install

```python
from RealtimeSTT import AudioToTextRecorder

if __name__ == "__main__":
    with AudioToTextRecorder(model="tiny", device="cpu") as recorder:
        print("Speak now")
        print(recorder.text())
```

For the test workflow, see [testing.md](testing.md).
