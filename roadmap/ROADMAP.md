# RealtimeSTT Development Roadmap

This roadmap describes the next major development directions for RealtimeSTT.
It is a planning document, not a release promise. The goal is to make the
project easier to understand, easier to extend, and strong enough for realistic
application deployments while preserving the simple few-line demo experience
that makes the library approachable.

## Guiding Goals

- Keep the first user experience simple: a short README, a few clear examples,
  and a direct path from install to first transcription.
- Move advanced material into focused reference documents instead of growing one
  very large README.
- Treat transcription engines, VAD engines, and wake word engines as explicit
  pluggable subsystems.
- Make the FastAPI browser server a serious reference application, not only a
  demo.
- Establish clear Python coding style and architecture guidelines before large
  new code and refactors continue.
- Reduce risk by pairing refactors with focused regression tests before and
  after each extraction.
- Keep optional dependencies optional, so users can install only the engines and
  features they actually need.

## Current Foundation

Recent unreleased work has already moved the project in this direction:

- ASR now routes through `RealtimeSTT/transcription_engines`.
- `faster_whisper` remains the compatibility default.
- Main and realtime transcription can use different engines.
- Engine-specific options can be passed through dedicated option dictionaries.
- Several non-default engines have adapters or test coverage, including
  whisper.cpp, OpenAI Whisper, Moonshine, Parakeet/NeMo, Cohere Transcribe,
  Granite Speech, Qwen3-ASR, and sherpa-onnx CPU INT8 paths.
- Regression work now covers slow final transcription gaps and pre-roll buffer
  carry-over.
- `example_fastapi_server` has grown into a multi-user browser transcription
  server with session isolation, shared inference resources, health endpoints,
  metrics, and scheduling tests.

The next phase should turn this technical foundation into a clearer project
shape.

## Phase 0: Coding Style and Architecture Discovery

Priority: high.

Before larger refactors and new subsystems continue, prepare a dedicated
discovery pass for deriving the future coding style and architecture guide.
This phase should not write the final concrete rules yet. Instead, it should
describe exactly how to study the current codebase, especially
`RealtimeSTT/audio_recorder.py`, so a later session can produce higher-quality
instructions.

Create `docs/coding-style-and-architecture.md` as a discovery plan covering how
to inspect and collect evidence about:

- How `audio_recorder.py` is written in its best parts.
- How functions and methods are named, shaped, grouped, and documented.
- How imports are arranged, including whether the project intentionally uses
  visual ordering such as longer import lines first.
- How comments are used, how much commenting is helpful, and which comments
  explain non-obvious behavior.
- How public APIs, callbacks, signatures, state transitions, and timing fields
  are documented.
- How classes, methods, variables, constants, and tests are named.
- How newer modular files differ from the legacy recorder style.
- How architecture should be inferred from current recorder responsibilities,
  engine adapters, server code, and tests.
- How to gather evidence for module size guidance, including the possible
  target of keeping ordinary new files under roughly 300 lines.
- How to evaluate PEP 8, flake, ruff, formatting, and import-order tooling
  without creating premature rules.

Acceptance criteria:

- `docs/coding-style-and-architecture.md` clearly states that it is a discovery
  plan, not the final style guide.
- The document lists the source files to inspect and the passes to perform.
- The document explains how to collect concrete examples and counterexamples
  from `audio_recorder.py` and newer modules.
- The document identifies what evidence is needed before deciding naming,
  import, comment, docstring, signature, architecture, file-size, and tooling
  rules.
- The document defines the handoff to a later dedicated style-guide session.
- No final concrete code-style rules are treated as decided in this phase.

## Phase 1: Realtime Text Stabilization

Priority: high.

Design requirements for a realtime text stabilization layer before changing the
delicate VAD-driven audio frame sizing. The current realtime text is simply the
latest transcription returned for the current audio frame. The next step is to
record every incoming realtime transcription with the best available timing
metadata, then define requirements for identifying text fragments that are
stable enough to emit as non-revisable output for downstream consumers such as
LLMs, while still displaying the unstable right side of the live transcription.

Detailed requirements live in
`roadmap/realtime-text-stabilization-requirements.md`.

## Phase 2: Documentation Rework

The documentation should be reorganized around two audiences:

- New users who want to try RealtimeSTT quickly.
- Advanced users who need exact configuration, engine setup, deployment, and
  testing references.

### README Reset

Rewrite `README.md` as a compact project overview:

- What RealtimeSTT is and when to use it.
- The shortest working microphone example.
- A short automatic recording example.
- A short external audio feeding example.
- Links to the detailed docs.
- A minimal install section with the default supported path.
- A small feature list that points to deeper reference pages.

The README should not be the full parameter manual, engine manual, wake word
manual, testing guide, and deployment guide at the same time.

### Docs Structure

Create or revise focused documents in `docs/`:

- `docs/quick-start.md`: easy examples from the smallest demo to common
  automatic recording patterns.
- `docs/installation.md`: base install, platform notes, CUDA notes, and optional
  dependency groups once extras are available.
- `docs/configuration.md`: complete `AudioToTextRecorder` parameter reference.
- `docs/transcription-engines.md`: engine selection overview plus per-engine
  setup.
- `docs/engines/faster-whisper.md`: install, model behavior, GPU notes, and
  common options.
- `docs/engines/whisper-cpp.md`: install, model download paths, expected model
  files, CPU tuning, and troubleshooting.
- `docs/engines/openai-whisper.md`: install, model choices, CPU/GPU behavior,
  and tradeoffs.
- `docs/engines/moonshine.md`: install, model cache behavior, CPU usage, and
  FastAPI recipes.
- `docs/engines/sherpa-onnx.md`: install, model download requirements, where
  models live, and supported prebuilt paths.
- `docs/engines/parakeet-nemo.md`: install, CUDA requirements, model cache
  behavior, and expected resource usage.
- `docs/engines/hf-transformers.md`: Granite Speech, Qwen3-ASR, and other
  Transformers-backed engines.
- `docs/engines/cohere.md`: API setup, credentials, supported paths, and
  latency/cost notes.
- `docs/wake-words.md`: Porcupine and OpenWakeWord setup, model files,
  sensitivities, callbacks, and examples.
- `docs/external-audio.md`: how to avoid the microphone and feed audio chunks
  from files, streams, websocket clients, or other processes.
- `docs/testing.md`: maintained unit and golden test workflow.
- `docs/test-scripts.md`: explain the scripts directly under `tests/` that are
  not in `tests/unit`, including which are demos, manual tests, regressions, and
  legacy experiments.
- `docs/fastapi-server.md`: server overview, configuration, protocol, metrics,
  deployment, and browser UI behavior.
- `docs/troubleshooting.md`: common install, audio device, CUDA, model download,
  dependency, and runtime errors.

### Documentation Acceptance Criteria

- A new user can find a working microphone example in under one minute.
- A user can identify which engine to install without reading source code.
- Every optional engine documents whether model files are downloaded
  automatically, need to be downloaded manually, and where they should be placed.
- Wake words, external audio input, unit tests, and non-unit test scripts each
  have their own focused documentation.
- The README links to reference docs instead of duplicating them.

## Phase 3: Sophisticated FastAPI Server

`example_fastapi_server` should evolve into the main reference application for
browser-based RealtimeSTT usage.

### Server Configuration

Add a configuration surface that can control:

- Final transcription engine.
- Realtime transcription engine.
- Engine-specific options.
- VAD and post-detection parameters.
- Wake word engine and wake word options.
- Buffer sizes, pre-recording behavior, queue limits, and scheduler policies.
- Per-session limits and global server capacity.

The server should support both startup configuration and runtime-safe changes
where practical. Runtime changes must be explicit about whether they apply only
to new sessions or can safely affect active sessions.

### Wake Word Integration

Integrate wake words into the FastAPI server flow:

- Enable and disable wake word mode through server configuration.
- Surface wake word detections in websocket events.
- Show wake word state in the browser UI.
- Record transitions such as waiting for wake word, wake word detected waiting
  for voice, recording, and returning to wake word wait mode.

### Better History and Timeline

The browser UI should show a detailed event timeline, not only final text:

- Exact timestamp when recording started.
- Exact timestamp when recording ended.
- Duration of the recorded segment.
- Which pre-recording buffer range was included.
- Wake word detection timestamp when present.
- Timestamp when wake-word-detected waiting-for-voice mode times out and snaps
  back to waiting for wake word.
- Realtime and final transcript updates associated with the same segment.

### Text Detection Window

Improve the live and final text display:

- Show recording start time.
- Show recording end time when known.
- Show calculated duration.
- Keep live text and final text clearly related to their segment.
- Preserve session isolation in multi-user mode.

### FastAPI Acceptance Criteria

- Server options can switch transcription engines without changing source code.
- Wake word mode works through the browser server and is visible in the UI.
- The UI timeline makes pre-roll, wake word events, recording start, recording
  end, and final transcript timing understandable.
- Tests cover protocol events, server settings, wake word state transitions,
  session isolation, and timeline data generation.

## Phase 4: Async/Await API

Add an async interface for applications that are already built around
`asyncio`.

Target capabilities:

- Async recorder lifecycle management.
- Awaitable transcription calls where appropriate.
- Async iteration or callback patterns for realtime updates.
- Clean cancellation and shutdown semantics.
- Compatibility guidance for existing synchronous users.

This should be designed carefully so async support feels native instead of being
a thin wrapper around blocking behavior.

## Phase 5: VAD Engine Abstraction

The current detection path uses WebRTC VAD for initial detection and then
double-checks with Silero VAD. The long-term goal is to make VAD configurable in
the same spirit as transcription engines.

### Target Design

- Define a VAD engine interface.
- Provide adapters for WebRTC VAD and Silero VAD.
- Allow one or more VAD engines to be loaded.
- Allow explicit VAD chains, for example:
  - WebRTC VAD as initial check, then Silero VAD confirmation.
  - Silero VAD only.
  - WebRTC VAD only.
  - Custom VAD engine plus built-in confirmation.
- Make it straightforward to add a new VAD engine without editing recorder core
  logic.

### VAD Acceptance Criteria

- Existing default behavior remains compatible.
- Users can select a VAD strategy through configuration.
- Tests cover at least the current WebRTC plus Silero behavior, Silero-only
  behavior, and WebRTC-only behavior.
- The abstraction does not make the realtime path noticeably harder to reason
  about.

## Phase 6: Wake Word Engine Abstraction

Wake words should become another pluggable subsystem rather than recorder-core
special cases.

### Target Design

- Define a wake word engine interface.
- Provide adapters for Porcupine and OpenWakeWord.
- Support engine-specific model files and sensitivity options.
- Keep callback behavior compatible.
- Make it easy to add another wake word backend later.

### Wake Word Acceptance Criteria

- Existing Porcupine and OpenWakeWord users keep working.
- Setup and model-file behavior is documented per engine.
- Tests cover engine selection, wake word detection events, timeout behavior,
  and state transitions.
- FastAPI wake word support uses the same abstraction as the library.

## Phase 7: Fine-Grained Installation Extras

Move toward dependency extras similar to RealtimeTTS so users can install only
the parts they need.

Example target installs:

```bash
pip install RealtimeSTT[whispercpp,webrtcvad]
pip install RealtimeSTT[fasterwhisper,webrtcvad,silerovad,pvporcupine,openwakeword]
```

Potential extras:

- `fasterwhisper`
- `whispercpp`
- `openaiwhisper`
- `moonshine`
- `sherpaonnx`
- `parakeet`
- `transformers`
- `cohere`
- `webrtcvad`
- `silerovad`
- `pvporcupine`
- `openwakeword`
- `fastapi`
- `server`

Acceptance criteria:

- Base install stays as small and predictable as practical.
- Each engine document names the exact extra to install.
- CI or local test documentation verifies the important install combinations.
- Optional imports fail with helpful messages that name the missing extra.

## Phase 8: Reduce `audio_recorder.py`

`RealtimeSTT/audio_recorder.py` is currently very large. Refactoring should be
done as a repeated safe loop, not as a broad rewrite.

Loop:

1. Identify one candidate responsibility to extract.
2. Check whether current behavior is already covered by a focused test.
3. If no test exists, write one before extraction.
4. Run the test and fix it until it proves current behavior.
5. Extract the responsibility into a focused module.
6. Run the same test again.
7. Fix regressions until the test passes.
8. Return to step 1.

Possible extraction candidates:

- Recorder state transitions.
- Pre-recording buffer management.
- Wake word state handling.
- VAD decision pipeline.
- Realtime transcription scheduling.
- Final transcription queueing.
- Audio input stream management.
- Callback dispatch.
- Finite-stream flushing and test helpers.
- Parameter validation and normalization.

Acceptance criteria:

- Public behavior stays compatible unless a breaking change is planned and
  documented.
- Each extraction is small enough to review.
- Each extracted module has focused tests.
- The recorder core becomes easier to read without creating a maze of tiny,
  unclear files.

## Phase 9: Linux FastAPI Reference Deployment

Add a practical Linux deployment guide and demo based on the existing
`example_fastapi_server`.

Target setup:

- Linux-based FastAPI server.
- A simple but effective CPU-friendly engine choice, likely Moonshine or another
  small reliable engine.
- Browser client served by the example server.
- Docker support.
- Clear manual Linux installation instructions.

Docker goals:

- Keep the Dockerfile short and maintainable.
- Prefer a minimal set of apt packages and pip installs.
- Avoid turning the Dockerfile into a long, fragile install script.
- Document model download and cache behavior clearly.
- Include health checks or smoke-test instructions.

Acceptance criteria:

- A Linux user can run the server from documented commands.
- Docker build and run instructions are reproducible.
- The guide explains the engine choice and the tradeoffs.
- The setup is useful as a real starting point, not just a toy demo.

## Cross-Cutting Work

### Testing

- Keep fast unit tests separate from real-engine golden tests.
- Maintain fixtures that prove timing, pre-roll, and transcription boundaries.
- Add integration tests for FastAPI server state and websocket behavior.
- Add opt-in tests for heavy engines and external model downloads.

### Compatibility

- Preserve default `faster_whisper` behavior where possible.
- Document intentional behavior changes.
- Keep optional engine failures lazy and understandable.
- Avoid forcing server dependencies on library-only users.

### Observability

- Expose timing and state transitions in server events.
- Keep metrics useful for debugging latency, queueing, drops, coalescing, and
  active sessions.
- Make event naming consistent across library callbacks and FastAPI websocket
  messages where practical.

## Suggested Order

1. Coding style and architecture discovery.
2. Realtime text stabilization requirements and design.
3. Documentation rework.
4. FastAPI configuration and timeline improvements.
5. Wake word integration in the FastAPI server.
6. Fine-grained install extras.
7. VAD abstraction.
8. Wake word abstraction.
9. Async/await API.
10. Iterative `audio_recorder.py` extractions.
11. Linux FastAPI reference deployment.

Some items can overlap, but the coding style and architecture discovery,
realtime text stabilization requirements, and documentation should happen early
because they will clarify the project surface before the next abstraction layers
are added.

## Open Questions

- Which install extras should be included in the first packaging pass, and which
  should wait?
- Should VAD and wake word engine interfaces live beside
  `transcription_engines`, or should all pluggable engines move under a broader
  package namespace?
- Which FastAPI settings should be runtime-configurable for active sessions, and
  which should only apply to new sessions?
- Which Linux reference engine should be the default for the deployment guide:
  Moonshine, sherpa-onnx Moonshine, whisper.cpp, or another CPU-friendly path?
- Should async support be implemented before or after the recorder extraction
  work, given that each may influence the recorder lifecycle shape?
- Should the style guide be enforced with ruff, flake8, black, or a lighter
  documented-only process at first?
- Should the 300-line module target be enforced as a warning in tooling, or used
  only as a review guideline with explicit exceptions?
- What exact criteria should make realtime text stable enough to emit as
  non-revisable output?
