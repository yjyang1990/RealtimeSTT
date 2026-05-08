# Unreleased Changes

## Scope

This document tracks relevant repository changes that should be carried forward
until the next release. It starts with the recent integration work that moved
RealtimeSTT beyond a mostly `faster_whisper`-centered path.

I did not find saved conversation transcripts in the repository. The initial
summary is based on the current repo artifacts: code, docs, tests, and the
existing `dev-log` entries.

## What Changed

### Transcription engine abstraction

RealtimeSTT now routes ASR through `RealtimeSTT/transcription_engines` instead
of binding recorder behavior directly to faster-whisper.

- `faster_whisper` remains the default backend for compatibility.
- Main/final and realtime transcription can use different backends through
  `transcription_engine` and `realtime_transcription_engine`.
- Backend-specific options are passed through
  `transcription_engine_options` and `realtime_transcription_engine_options`.
- Engine loading is lazy, so optional dependencies are only imported when that
  backend is selected.
- Supported adapters now include faster-whisper, whisper.cpp, OpenAI Whisper,
  Parakeet/NeMo, Moonshine, Cohere Transcribe, Granite Speech, Qwen3-ASR, and
  sherpa-onnx Parakeet/Moonshine CPU INT8 paths. The OpenAI API engine is still
  present as a placeholder.

The detailed user-facing notes live in `docs/transcription-engines.md`.

### Recorder correctness fixes

Two recorder bugs were isolated and documented:

- `bug001_slow_cpu_final_transcription_gap.md`: slow final transcription on CPU
  could make the recorder miss incoming utterances while the app thread was
  blocked in `text()`. The fix queues completed recordings, keeps continuous
  listening armed, tolerates delayed Silero confirmation, and supports flushing
  finite input streams.
- `bug002_pre_roll_buffer_carryover.md`: stale audio from the previous utterance
  could leak into the pre-roll of the next final transcription. The fix clears
  the pre-recording buffer at the recording-to-stopped transition while keeping
  normal pre-roll behavior for future speech.

The regression harness now also saves final-transcription snippets and manifests,
which made the pre-roll problem audible and easier to prove.

### FastAPI browser server

`example_fastapi_server` has grown into a configurable browser transcription
server instead of a minimal demo.

- It serves a browser UI and a `/ws/transcribe` binary audio protocol.
- The server exposes engine selection flags for final and realtime models.
- It documents practical CPU and GPU recipes, including whisper.cpp,
  sherpa-onnx Moonshine, and Parakeet.
- It exposes `/health` and `/api/metrics` for readiness, scheduler state,
  latency, queue depth, drops, coalescing, and per-session counters.
- Tests cover protocol parsing, settings behavior, and server/session logic.

### Multi-user strategy and implementation

The FastAPI server now follows the multi-user design captured in
`example_fastapi_server/MULTI_USER_IMPLEMENTATION_GUIDE.md`.

- Each websocket gets a `sessionId`.
- Transcript state, segment ids, audio buffers, warnings, errors, `clear`, and
  status are scoped to the owning session.
- Transcript-bearing events are routed to the owning session instead of being
  globally broadcast.
- Each accepted session owns lightweight recorder/VAD stream state, while heavy
  ASR engines are injected as shared executors.
- Scheduling uses per-session fair queues. Final jobs are preserved up to the
  configured limit, while stale realtime jobs can be coalesced or dropped.
- Capacity is explicit through settings such as `max_sessions`,
  `max_active_speakers`, `max_global_inference_queue_depth`,
  `max_final_queue_depth_per_session`, and
  `max_audio_queue_seconds_per_session`.
- Session slots are reserved before recorder construction, so overload does not
  accidentally create more recorder instances than the configured capacity.

The important architectural shift is that the server is no longer "one global
recorder plus many sockets". It is now "many isolated stream sessions plus
shared inference resources".

## Verification Landmarks

- Engine contract and optional golden-test paths are described in
  `docs/testing.md`.
- The slow CPU final transcription regression now preserves all expected
  utterances in the documented 9/9 fixture case.
- The pre-roll carry-over bug was verified by regenerated snippets and corrected
  expected text.
- FastAPI multi-user tests cover session isolation, clear/reset behavior,
  admission limits, fair scheduling, realtime coalescing, stale result discard,
  and websocket routing.
- An opt-in real-engine FastAPI multi-user performance test streams the shared
  audio fixture through multiple sessions and records WER, latency, scheduler,
  coalescing, and drop metrics.

## Current Shape

RealtimeSTT is now positioned as a backend-flexible recorder and browser server:
default-compatible for existing faster-whisper users, but open to CPU, ONNX,
Whisper, NeMo, Transformers, and future server/API engines. The recorder fixes
make slow final transcription and finite-stream tests much safer, while the
FastAPI work gives the project a realistic foundation for isolated multi-user
realtime transcription.
