# Kroko-ONNX

`kroko_onnx` uses the optional
[kroko-ai/kroko-onnx](https://github.com/kroko-ai/kroko-onnx) runtime with
Kroko/Banafo streaming `.data` models. The adapter is lazy-loaded, so normal
RealtimeSTT installs and tests do not require Kroko-ONNX.

## Engine Names

- `kroko_onnx`
- `kroko`
- `banafo_kroko`

Hyphenated CLI forms such as `kroko-onnx` and `banafo-kroko` are accepted by
the generic engine-name normalization.

## Install

Kroko-ONNX is currently built from source:

```bash
git clone https://github.com/kroko-ai/kroko-onnx.git
cd kroko-onnx
python -m pip install .
```

For Pro models, build/install with license support as documented upstream and
pass the license key through `transcription_engine_options["key"]` or the
FastAPI `--engine-options` JSON. Do not commit license keys.

Upstream documentation currently focuses on Linux and Docker. Windows and macOS
build instructions are listed as coming soon, so use WSL2/Linux or Docker if a
native Windows build fails.

## Windows Install Status

Native Windows installation has been attempted in a local test environment and
did not produce a usable `kroko_onnx` Python package.

The first attempt used the default upstream build:

```powershell
python -m pip install git+https://github.com/kroko-ai/kroko-onnx.git
```

It cloned `kroko-ai/kroko-onnx` at commit
`808ad7aad096467dcbfa5cc6d0cfbf2a39d54dc3` and failed while building the
native CMake wheel with MSVC.

A second attempt disabled GPU and several optional components:

```powershell
$env:SHERPA_ONNX_CMAKE_ARGS='-DSHERPA_ONNX_ENABLE_GPU=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TTS=OFF -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=OFF -DSHERPA_ONNX_ENABLE_BINARY=OFF'
python -m pip install --log test-results\kroko-onnx-pip-install-cpu.log git+https://github.com/kroko-ai/kroko-onnx.git
```

That CPU-only build also failed. The relevant MSVC errors were:

```text
online-recognizer.cc(32,10): error C1083: include file cannot be opened: "bits/stdc++.h": No such file or directory
license.h(7,10): error C1083: include file cannot be opened: "websocketpp/config/asio_client.hpp": No such file or directory
Exception: Failed to build and install sherpa
```

After these failures, `import kroko_onnx` still returned
`ModuleNotFoundError`, so neither CPU nor CUDA provider runtime tests could run
on native Windows. The community model download itself did work; the blocker is
the upstream native Windows build.

For real CPU or GPU validation, use WSL2/Linux or Docker and install
Kroko-ONNX there before running the opt-in smoke and FastAPI performance tests.

## Model Download

Community models are available from
[Banafo/Kroko-ASR](https://huggingface.co/Banafo/Kroko-ASR). The English
community streaming model is a good first smoke test:

```powershell
New-Item -ItemType Directory -Path test-model-cache\kroko-onnx -Force
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Banafo/Kroko-ASR', filename='Kroko-EN-Community-64-L-Streaming-001.data', local_dir='test-model-cache/kroko-onnx')"
```

Pass the `.data` file as `model`:

```python
from RealtimeSTT import AudioToTextRecorder

recorder = AudioToTextRecorder(
    transcription_engine="kroko_onnx",
    model="test-model-cache/kroko-onnx/Kroko-EN-Community-64-L-Streaming-001.data",
    device="cpu",
    language="en",
    transcription_engine_options={
        "provider": "cpu",
        "num_threads": 2,
    },
)
```

## Options

| Option | Meaning |
| --- | --- |
| `model_path` | Explicit `.data` model file. Overrides `model`. |
| `model_dir` | Directory containing a single `.data` file, or the default English community filename. |
| `model_filename` | File name to use inside `model_dir`. |
| `key` | License key for Pro models. |
| `referralcode` | Optional Kroko referral code. |
| `provider` | `cpu`, `cuda`, or `coreml`. Defaults from `device`. |
| `num_threads` | Runtime thread count. Defaults to `1`. |
| `sample_rate` | Kroko recognizer sample rate. Defaults to `16000`. |
| `feature_dim` | Feature dimension. Defaults to `80`. |
| `decoding_method` | `greedy_search` or `modified_beam_search`. |
| `max_active_paths` | Beam paths for modified beam search. |
| `hotwords_file`, `hotwords_score` | Optional hotword biasing inputs. |
| `blank_penalty` | Blank-symbol penalty during decoding. |
| `enable_endpoint_detection` | Enables Kroko endpoint detection. |
| `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, `rule3_min_utterance_length` | Endpoint rule values. |
| `tail_padding_seconds` | Final silence padding before decoding. Defaults to `0.66`. |
| `recognizer` | Extra dictionary merged into `OnlineRecognizer.from_transducer(...)`. |

## FastAPI Examples

Kroko for both final and realtime transcription on CPU:

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --host 0.0.0.0 `
  --port 8010 `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine kroko_onnx `
  --realtime-model $model `
  --device cpu `
  --language en `
  --engine-options '{"provider":"cpu","num_threads":2}' `
  --realtime-engine-options '{"provider":"cpu","num_threads":1}'
```

Kroko final transcription with a lighter realtime engine:

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine whisper_cpp `
  --realtime-model tiny.en `
  --device cpu `
  --language en `
  --engine-options '{"provider":"cpu","num_threads":2}'
```

CUDA provider, when the installed Kroko-ONNX build supports it:

```powershell
$model = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
python example_fastapi_server\server.py `
  --engine kroko_onnx `
  --model $model `
  --realtime-engine kroko_onnx `
  --realtime-model $model `
  --device cuda `
  --language en `
  --engine-options '{"provider":"cuda","num_threads":2}' `
  --realtime-engine-options '{"provider":"cuda","num_threads":1}'
```

## Tests

Fast contract tests use fake Kroko runtime objects and do not require the
optional dependency:

```powershell
python -m unittest -v tests.unit.test_kroko_onnx_engine
```

Opt-in real-model smoke test:

```powershell
$env:REALTIMESTT_RUN_KROKO_ONNX = "1"
$env:REALTIMESTT_KROKO_ONNX_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_KROKO_ONNX_PROVIDER = "cpu"
$env:REALTIMESTT_KROKO_ONNX_NUM_THREADS = "1"
python -m unittest -v tests.unit.test_kroko_onnx_engine.KrokoOnnxGoldenTranscriptionTests
```

FastAPI multi-user performance harness:

```powershell
$env:REALTIMESTT_RUN_FASTAPI_MULTI_USER_PERF = "1"
$env:REALTIMESTT_FASTAPI_ASR_CLIENTS = "2"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE = "kroko_onnx"
$env:REALTIMESTT_FASTAPI_ASR_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE = "kroko_onnx"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_MODEL = "test-model-cache\kroko-onnx\Kroko-EN-Community-64-L-Streaming-001.data"
$env:REALTIMESTT_FASTAPI_ASR_DEVICE = "cpu"
$env:REALTIMESTT_FASTAPI_ASR_ENGINE_OPTIONS = "provider=cpu,num_threads=2"
$env:REALTIMESTT_FASTAPI_ASR_REALTIME_ENGINE_OPTIONS = "provider=cpu,num_threads=1"
$env:REALTIMESTT_FASTAPI_ASR_METRICS_JSON = "test-results\kroko-onnx-fastapi-cpu-2clients.json"
python -m unittest -v tests.unit.test_fastapi_server_multi_user_asr_integration.FastAPIMultiUserRealEngineASRTests
```

## Troubleshooting

- Missing dependency errors mean `kroko_onnx` is not importable in the active
  environment. Install Kroko-ONNX in that same environment.
- Missing model errors name the exact `.data` file path RealtimeSTT tried.
- If native Windows builds fail, use WSL2/Linux or Docker. Kroko's own README
  currently says Windows and macOS build instructions are coming soon.
- CUDA runs require both CUDA-capable hardware and a Kroko-ONNX build with CUDA
  provider support.
