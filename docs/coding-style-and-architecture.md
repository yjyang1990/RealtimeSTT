# RealtimeSTT Coding Style and Architecture Guide

This is the concrete style and architecture guide for future RealtimeSTT work.
It replaces the Phase 0 discovery plan. The rules below are derived from the
current codebase, with `RealtimeSTT/audio_recorder.py` treated as the source for
project vocabulary and public behavior, and the newer modular code treated as
the preferred direction for new files.

This guide is intentionally RealtimeSTT-specific. Do not use it as a generic
Python style checklist. When a rule conflicts with preserving current public
behavior, compatibility wins and the cleanup becomes a separate refactor.

## Evidence Base

Use these files as local examples when making style decisions:

- `RealtimeSTT/audio_recorder.py`: public recorder API, callback vocabulary,
  state names, timing fields, and legacy debt.
- `RealtimeSTT/transcription_engines/base.py`: project result/config objects and
  engine interface shape.
- `RealtimeSTT/transcription_engines/factory.py`: lazy engine loading and
  normalized engine names.
- `RealtimeSTT/transcription_engines/whisper_cpp_engine.py` and
  `RealtimeSTT/transcription_engines/sherpa_onnx_engine.py`: compact adapter
  modules with optional dependency errors.
- `RealtimeSTT/realtime_boundary_detector.py`: focused streaming algorithm
  module with typed result objects.
- `example_fastapi_server/protocol.py`: small protocol helper module with clear
  validation errors.
- `example_fastapi_server/server.py`: reference application architecture,
  scheduling, metrics, and server-specific logging.
- `tests/unit/*`: fake backends, optional dependency gates, and regression tests
  for extraction work.

## Style Priority

- **Required:** preserve public behavior and callback names before style cleanup.
  `AudioToTextRecorder` is already a public API; avoid breaking constructor
  names, callback names, return behavior, or synchronous lifecycle semantics.
- **Required:** new modules should follow the cleaner patterns in
  `transcription_engines`, `realtime_boundary_detector.py`, `protocol.py`, and
  the structured parts of the FastAPI server.
- **Preferred:** when improving legacy code, make the touched area more like the
  newer modular code only when tests cover the behavior.
- **Legacy/Exception:** `audio_recorder.py` contains important project language
  but also old formatting, eager optional imports, broad responsibilities,
  verbose comments, direct prints, and very large methods. Copy its domain
  concepts, not its accumulated size or incidental formatting.

## Imports And Top-Of-File Layout

- **Required:** group imports by standard library, third-party packages, and
  local RealtimeSTT imports, with one blank line between groups. Good models are
  `example_fastapi_server/server.py:1`, `RealtimeSTT/realtime_boundary_detector.py:12`,
  and `example_fastapi_server/protocol.py:1`.
- **Required:** do not use visual ordering such as longest import lines first as
  a project style. The mixed import block in `audio_recorder.py:29` through
  `audio_recorder.py:62` is legacy.
- **Required:** optional heavy dependencies must be imported lazily inside the
  backend or feature that needs them. `factory.py:30` loads engine classes lazily,
  `whisper_cpp_engine.py:31` loads `pywhispercpp` only for whisper.cpp, and
  `sherpa_onnx_engine.py:40` loads `sherpa_onnx` only for sherpa-onnx engines.
- **Required:** keep import-time side effects out of new modules. The
  `KMP_DUPLICATE_LIB_OK` environment mutation in `audio_recorder.py:69` is
  legacy and should not be repeated in new modules.
- **Preferred:** alphabetize imports inside each group unless there is a local
  reason to keep a small dependency chain together.
- **Preferred:** place module constants and loggers after imports. Examples:
  `audio_recorder.py:64`, `audio_recorder.py:71`,
  `example_fastapi_server/server.py:40`, and
  `example_fastapi_server/protocol.py:7`.
- **Legacy/Exception:** top-level imports of optional engines such as
  `openwakeword`, `pvporcupine`, and `webrtcvad` in `audio_recorder.py:30`
  through `audio_recorder.py:49` remain compatibility debt until those features
  are extracted.

## File And Module Organization

- **Required:** a normal new module should have this shape:
  module docstring when it adds useful context, imports, logger/constants,
  dataclasses/exceptions, small helper functions, then public classes.
- **Required:** keep pure protocol/data helpers separate from runtime services.
  `example_fastapi_server/protocol.py` is the model: it has packet dataclasses,
  encode/decode helpers, and validation errors without importing the server.
- **Required:** keep adapters separate from factories. The factory maps names to
  adapter classes; adapters own backend-specific setup and transcription.
- **Preferred:** keep ordinary new modules under about 300 lines. This is a
  review guideline, not a hard build rule.
- **Preferred:** split a file when it contains two independent reasons to
  change, such as protocol parsing plus websocket lifecycle, or VAD scoring plus
  recorder callbacks.
- **Preferred:** one file can exceed 300 lines when it is a cohesive algorithm
  or reference application slice. `realtime_boundary_detector.py` is about 491
  lines and remains understandable because it owns one streaming detector.
- **Legacy/Exception:** `audio_recorder.py` and `example_fastapi_server/server.py`
  are large legacy/reference files. Do not use their size as permission for new
  monoliths. Future extractions should follow the safe loop in the roadmap:
  add or confirm focused tests, extract one responsibility, rerun tests.

## Naming

- **Required:** use `PascalCase` for classes and `snake_case` for functions,
  methods, variables, module names, and engine names.
- **Required:** name public classes as domain nouns:
  `AudioToTextRecorder`, `TranscriptionEngineConfig`,
  `TranscriptionResult`, `SpeechBoundaryEvent`, `FairInferenceQueue`.
- **Required:** engine modules use the pattern
  `<BackendName>Backend` for the third-party wrapper and `<BackendName>Engine`
  for the project adapter. Examples: `PyWhisperCppBackend` /
  `WhisperCppEngine`, `SherpaOnnxMoonshineBackend` /
  `SherpaOnnxMoonshineEngine`.
- **Required:** each engine exposes a snake_case `engine_name` matching the
  factory key, such as `"whisper_cpp"` or `"sherpa_onnx_moonshine"`.
- **Required:** callbacks are named `on_<event>`, using start/stop or
  detected/timeout pairs when the state has two sides. The recorder constructor
  and attributes around `audio_recorder.py:251` and `audio_recorder.py:628`
  define the vocabulary.
- **Required:** boolean state names should say what truth means. Use prefixes
  such as `is_`, `has_`, `use_`, `enable_`, `allow_`, `should_`, or explicit
  past-tense event names such as `wakeword_detected`.
- **Required:** names that carry units must include the unit when ambiguity is
  likely. Prefer `_seconds` and `_ms` for durations, `_sample_rate` for rates,
  `_samples` for sample counts, and `_time` or `_at` for timestamps. Good
  examples include `boundary_time_seconds`, `latency_ms`,
  `post_speech_silence_duration`, and `backdate_stop_seconds`.
- **Preferred:** use `DEFAULT_` for new default constants and upper snake case
  for all module constants. Existing `INIT_*` constants in
  `audio_recorder.py:71` are public-adjacent legacy and should remain unless a
  compatibility plan says otherwise.
- **Preferred:** private helpers start with one underscore and name the operation
  they isolate: `_normalize_audio`, `_get_prompt`, `_resolve_model_dir`,
  `_maybe_detect_boundary`, `_run_callback`.
- **Legacy/Exception:** `class bcolors` in `audio_recorder.py:228` and vague
  helper names such as `format_number` in `audio_recorder.py:1911` are legacy.
  Do not copy these patterns.

## Public API Shape

- **Required:** keep the first-user recorder API simple. `AudioToTextRecorder`,
  `start()`, `stop()`, `listen()`, `text()`, `feed_audio()`, and `shutdown()`
  remain the main synchronous surface.
- **Required:** public methods must document whether they block, start threads
  or processes, return text, return `self`, invoke callbacks, or mutate recorder
  state. `text()` at `audio_recorder.py:1866` and `stop()` at
  `audio_recorder.py:1957` are the public behavior precedent.
- **Required:** new feature configuration should avoid adding another long tail
  to `AudioToTextRecorder.__init__` unless the option must be top-level for
  compatibility. Prefer focused config objects, engine option dictionaries, or
  future subsystem configs.
- **Required:** public API inputs should be normalized at the boundary. Engine
  names accept hyphen aliases but become snake_case in
  `factory.py:36` and `example_fastapi_server/protocol.py:20`.
- **Preferred:** public data exchanged across subsystem boundaries should be
  dataclasses or small result objects, as in `TranscriptionResult`,
  `TranscriptionEngineConfig`, `InferenceJob`, `InferenceResult`, and
  `AudioPacket`.
- **Preferred:** mutating lifecycle methods may return `self` only where this is
  already established (`start()` and `stop()`). New APIs should return explicit
  results or `None`.
- **Legacy/Exception:** the current recorder constructor at
  `audio_recorder.py:241` is necessarily broad. Do not treat that signature as a
  model for new classes.

## Function And Method Shape

- **Required:** keep new functions and methods focused on one responsibility.
  `example_fastapi_server/protocol.py:40` through `protocol.py:81` is a good
  model for small validation helpers.
- **Required:** long signatures use one parameter per line, aligned under the
  opening parenthesis or using the standard hanging-indent shape. Put the
  closing `):` on its own line when the parameter list spans many lines.
- **Required:** do not add spaces around `=` in default arguments. The
  `start(self, frames = None)` style in `audio_recorder.py:1920` is legacy.
- **Required:** prefer early validation and clear returns over deeply nested
  control flow.
- **Preferred:** split private helpers around durable domain operations:
  buffering, state transitions, VAD decision, wake word detection, engine
  invocation, result normalization, callback dispatch.
- **Preferred:** when a method coordinates locks, queues, threads, or processes,
  keep the concurrency boundary explicit in the name or docstring. Examples:
  `_submit_transcription_request`, `_receive_transcription_result`,
  `FairInferenceQueue.submit`, and `SharedEngineWorker._worker`.
- **Legacy/Exception:** `_recording_worker` around `audio_recorder.py:2160` is a
  behavior source, not a shape to copy. It combines queue handling, wake words,
  VAD, pre-roll, state transitions, logging, and callback dispatch; future work
  should extract one tested responsibility at a time.

## Docstrings

- **Required:** every public class, public method, public function, and public
  dataclass-like result object needs a docstring unless its purpose is entirely
  obvious from a tiny dataclass.
- **Required:** public API docstrings must state:
  what the object or function does, important side effects, callback behavior,
  units for timing/audio parameters, return value, and exceptions users are
  expected to handle.
- **Required:** callbacks must document when they fire, what arguments they
  receive, and whether they may run on a background thread. The
  `start_callback_in_new_thread` documentation at `audio_recorder.py:587` is the
  level of behavioral detail expected.
- **Preferred:** use a short summary followed by `Args:`, `Returns:`, and
  `Raises:` when helpful. Keep parameter descriptions precise but not README
  length.
- **Preferred:** private helper docstrings should explain contracts or
  non-obvious algorithms, not repeat the function name. `process_bytes()` and
  `process_samples()` in `realtime_boundary_detector.py:194` and
  `realtime_boundary_detector.py:206` are good concise examples.
- **Preferred:** regression tests may use a docstring when the bug story matters.
  `test_stopped_recording_is_queued_beyond_pre_recording_window` in
  `tests/unit/test_slow_final_transcription_audio_gap.py:77` is a good example.
- **Legacy/Exception:** very long constructor docstrings like
  `audio_recorder.py:337` through `audio_recorder.py:614` are acceptable for the
  current public API but should not grow indefinitely. New long references
  belong in `docs/`.

## Comments

- **Required:** comments explain why the code does something non-obvious in
  realtime audio, buffering, VAD, wake word, multiprocessing, or protocol
  compatibility.
- **Required:** comments must stay close to the behavior they explain. If a
  comment describes public behavior, consider a docstring or test instead.
- **Preferred:** use short block comments before tricky state transitions or
  timing decisions. For example, comments around retaining completed recordings
  while final transcription blocks belong near tests and extraction points.
- **Preferred:** comments may name hardware or dependency quirks when the code
  would otherwise look arbitrary.
- **Legacy/Exception:** comments that narrate obvious code should be removed when
  touching the area. Examples include `format_number()` comments at
  `audio_recorder.py:1912` and the step-by-step substring comments at
  `audio_recorder.py:3566`.
- **Legacy/Exception:** repetitive extended debug comments/logs such as
  "Debug: Checking ..." in `_recording_worker` are legacy. New debug logs should
  describe meaningful events or measured values.

## Type Hints

- **Required:** new public APIs, dataclasses, config objects, and result objects
  should have type hints.
- **Required:** use `typing.Optional`, `List`, `Dict`, `Tuple`, `Callable`, and
  `Union` instead of PEP 604 `|` syntax while the package declares Python
  `>=3.6` in `setup.py:36`.
- **Required:** annotate return types for public methods that return structured
  objects, booleans, text, or `None`.
- **Preferred:** annotate private helpers when the values are not obvious,
  especially audio sample arrays, PCM bytes, timestamps, executor callbacks, and
  queue items.
- **Preferred:** keep test fakes lightweight; only type them where it clarifies a
  contract.
- **Legacy/Exception:** do not churn legacy files just to add annotations.
  Add hints while extracting or changing behavior.

## Logging And Error Handling

- **Required:** library code uses named loggers, not direct `print()`. The
  recorder logger at `audio_recorder.py:64` and server logger at
  `example_fastapi_server/server.py:40` are the precedent.
- **Required:** new library modules should not configure global logging handlers
  at import time. Let applications decide handlers and levels.
- **Required:** optional dependency failures must raise project-specific,
  actionable errors that name the missing package and how to install or avoid
  it. Examples: `whisper_cpp_engine.py:31`, `parakeet_engine.py:39`,
  `qwen3_asr_engine.py:66`, and `sherpa_onnx_engine.py:40`.
- **Required:** protocol validation errors should identify the bad field or
  shape. `AudioPacketError` messages in `protocol.py:40` through `protocol.py:81`
  are the model.
- **Required:** catch broad `Exception` only at subsystem boundaries where the
  code can add context, publish an error event, keep a worker alive, or shut down
  cleanly.
- **Preferred:** use `logger.exception(...)` when preserving a stack trace at a
  boundary, and use `logger.debug(..., exc_info=True)` for expected fallback
  paths such as optional warmup failures.
- **Preferred:** include request/session/engine names in server logs, as
  `SharedEngineWorker` does for engine initialization and inference jobs.
- **Legacy/Exception:** direct prints in `audio_recorder.py:2076`,
  `audio_input.py`, and golden smoke tests are legacy or CLI/test output. Do not
  add new prints to library runtime paths.
- **Legacy/Exception:** constructor-level logger handler setup in
  `audio_recorder.py:752` through `audio_recorder.py:772` exists for backward
  behavior. New modules should not repeat it.

## Optional Dependencies

- **Required:** optional engines, VAD backends, wake word backends, and server
  dependencies must fail lazily and clearly.
- **Required:** a user should be able to import the package without installing
  every optional engine. Backend imports belong in `_load_*` helpers or factory
  paths, not in package `__init__` modules.
- **Required:** error messages for missing optional dependencies must include:
  feature or engine name, package name, install hint, and any platform caveat.
- **Preferred:** tests should patch `import_module` and assert the message, as
  seen in optional dependency tests around
  `tests/unit/test_additional_transcription_engines.py:428`,
  `tests/unit/test_cohere_transcribe_engine.py:101`, and
  `tests/unit/test_whisper_cpp_engine.py:161`.
- **Legacy/Exception:** current eager imports in `audio_recorder.py` may still
  fail before feature use. Treat that as extraction debt for VAD and wake word
  phases.

## Test Naming And Structure

- **Required:** unit tests live under `tests/unit/` and use `unittest` unless a
  dedicated test migration changes the project-wide test framework.
- **Required:** test files are named `test_<module_or_behavior>.py`; test classes
  end in `Tests`; methods start with `test_` and describe behavior.
- **Required:** use fake backends and patched imports for engine behavior that
  should not download models or require GPUs. See fake engine patterns in
  `tests/unit/test_additional_transcription_engines.py:336` and dedicated module
  tests such as `tests/unit/test_cohere_transcribe_engine.py:22`.
- **Required:** use `subTest` for matrix cases and aliases, as in
  `tests/unit/test_additional_transcription_engines.py:371`.
- **Required:** slow real-engine or hardware-dependent tests must be opt-in via
  environment variables or explicit skip conditions. Good examples are the
  golden and smoke tests that call `skipTest` when the env var or dependency is
  absent.
- **Preferred:** extraction tests should prove current behavior before moving
  code. For recorder extractions, build small recorder stubs with `__new__`
  only when full initialization would load models or audio devices, as in
  `tests/unit/test_slow_final_transcription_audio_gap.py:91`.
- **Preferred:** keep test helper fakes near the tests that use them until they
  are reused by three or more files. Then move them to a focused test helper
  module.
- **Legacy/Exception:** tests that print actual transcripts are acceptable for
  opt-in golden smoke tests, not for fast deterministic unit tests.

## Architecture Boundaries

- **Required:** dependency direction is one-way:
  server/examples may import RealtimeSTT library code; core recorder code may
  import engine interfaces and focused helpers; engine adapters must not import
  the recorder or server; protocol helpers must not import server runtime.
- **Required:** keep third-party library details inside backend classes or
  feature adapters. Project-facing classes return project results such as
  `TranscriptionResult`, not third-party objects.
- **Required:** shared state transitions and callbacks stay centralized. The
  existing `_set_state()` method at `audio_recorder.py:3442` shows why: state
  names, spinner text, and start/end callbacks must remain consistent.
- **Required:** callback dispatch stays behind one helper. `_run_callback()` at
  `audio_recorder.py:1167` is the compatibility point for threaded callback
  behavior.
- **Preferred:** new subsystems should have a small base interface, a factory or
  registry, and backend adapters. This is already working for transcription
  engines through `BaseTranscriptionEngine`, `TranscriptionEngineConfig`, and
  `create_transcription_engine`.
- **Preferred:** server-only scheduling, websocket protocol, metrics, and
  session admission remain in `example_fastapi_server`. Do not move those
  concerns into `AudioToTextRecorder`.
- **Legacy/Exception:** `audio_recorder.py` currently owns microphone input,
  buffering, WebRTC VAD, Silero VAD, wake words, realtime transcription, final
  transcription, callback dispatch, and process management. This is the roadmap
  extraction source, not the target architecture.

## Extension Points

### Transcription Engines

- **Required:** add a new engine by creating a focused adapter module under
  `RealtimeSTT/transcription_engines/`, adding aliases to
  `ENGINE_CLASS_PATHS`, and adding unit tests for factory aliases, option
  handling, optional dependency messages, and result normalization.
- **Required:** engine adapters implement `transcribe(audio, language=None,
  use_prompt=True)` and return `TranscriptionResult`.
- **Required:** use backend injection in tests (`backend`, `backend_cls`,
  fake model classes, or patched imports) so engine tests do not require real
  model downloads.
- **Preferred:** split the backend wrapper from the project engine class:
  backend loads third-party code and speaks backend-specific options; engine
  normalizes audio, maps project config to backend parameters, and wraps output.

### VAD Engines

- **Required:** future VAD work should not add more backend-specific branches to
  `_recording_worker`. Define a small VAD adapter boundary first.
- **Required:** VAD adapters should expose explicit units and state reset
  behavior, because current VAD behavior depends on recent WebRTC/Silero state
  and timing (`_is_voice_active`, `_is_webrtc_speech`, `_is_silero_speech`).
- **Preferred:** model VAD decisions as result objects that can carry
  `is_speech`, confidence or score when available, source engine name, and
  timestamp/sample metadata.
- **Preferred:** compose VAD chains outside the recorder loop. The current
  WebRTC plus Silero confirmation behavior should become one strategy, not
  hard-coded recorder control flow.

### Wake Word Engines

- **Required:** future wake word support should mirror transcription engines:
  a backend adapter per dependency, lazy imports, helpful install/model-file
  errors, normalized project detection results, and recorder callbacks kept
  stable.
- **Required:** preserve current callback names such as
  `on_wakeword_detected`, `on_wakeword_timeout`,
  `on_wakeword_detection_start`, and `on_wakeword_detection_end`.
- **Preferred:** make wake word detections include the engine name, keyword or
  model id, score when available, and sample/timestamp metadata.
- **Legacy/Exception:** Porcupine and OpenWakeWord setup currently lives inside
  `AudioToTextRecorder.__init__`. Extract only with tests that cover current
  callback timing and timeout behavior.

## Formatting And Tooling

- **Required:** do not run broad formatters over the repo as part of feature or
  documentation work.
- **Required:** follow PEP 8 where it does not cause compatibility churn:
  four-space indentation, no tabs, snake_case names, upper snake constants,
  blank lines between top-level definitions, no spaces around default `=`.
- **Required:** keep Python `>=3.6` compatibility in mind until packaging raises
  the floor. Avoid syntax that Python 3.6 cannot parse.
- **Preferred:** keep new code near 100 columns. Break long expressions when
  doing so improves scanning. Long URLs, command strings, and user-facing
  messages may exceed this when wrapping would harm readability.
- **Preferred:** add Ruff or Flake8 in a dedicated tooling PR, not incidentally.
  Start with low-churn checks such as syntax/undefined names/imported-but-unused
  on touched files, then expand.
- **Preferred:** adopt import sorting only after configuring it for the required
  grouping above. Since visual longest-line ordering is not project style, an
  import sorter can be compatible if it avoids legacy-wide churn.
- **Preferred:** consider Black only after deciding the supported Python floor
  and after accepting that legacy files will need a deliberate formatting PR.
- **Legacy/Exception:** module length around 300 lines should remain
  documentation-only for now. Enforcing it in tooling would create noisy
  exceptions for `audio_recorder.py`, `server.py`, tests with fakes, and
  cohesive algorithms.

## Legacy Cleanup Policy

- **Required:** never mix broad style cleanup with behavior changes in legacy
  realtime paths. The recorder is timing-sensitive and heavily stateful.
- **Required:** before extracting recorder logic, write or identify a focused
  regression test that proves current behavior. Run it before and after the
  extraction.
- **Required:** preserve callback names, callback timing, constructor defaults,
  engine aliases, and default `faster_whisper` compatibility unless a breaking
  change is explicitly planned and documented.
- **Preferred:** when touching a legacy method, improve only the local area:
  clarify names, remove stale comments, add missing error context, or isolate a
  helper if tests make the move safe.
- **Preferred:** leave unrelated old formatting alone. A file can contain both
  old and new style during gradual cleanup.
- **Legacy/Exception:** large commented-out code blocks, direct prints, eager
  optional imports, and verbose debug narration should be removed only when the
  surrounding behavior is already covered or being deliberately refactored.

## Quick Checklist For Future Changes

- Does the change preserve the existing public recorder API and callbacks?
- Are imports grouped as standard library, third-party, local?
- Are optional dependencies imported lazily with helpful install errors?
- Is the new file under about 300 lines, or is there a clear cohesive reason?
- Does each public API docstring state side effects, units, callbacks, returns,
  and expected errors?
- Are timing fields named with explicit units?
- Are logs emitted through a named logger instead of `print()`?
- Are server/protocol concerns kept out of recorder core?
- Are engine/VAD/wake word details behind adapters rather than branches in the
  recorder loop?
- Are tests fast by default, with real engines and hardware paths opt-in?
