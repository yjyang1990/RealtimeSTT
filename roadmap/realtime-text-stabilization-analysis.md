# Realtime Text Stabilization Analysis

This document is an analysis and design-research companion to
`roadmap/realtime-text-stabilization-requirements.md`. It does not define final
requirements and does not propose source changes as already decided. The goal is
to expose product use cases, current-code coupling, failure modes, metadata
needs, and design tradeoffs before implementation begins.

## Executive Summary

RealtimeSTT currently has two different meanings of "realtime text":

- The raw live preview returned by the ASR engine for the current accumulated
  audio frame.
- A legacy "stabilized" callback derived from the common prefix of the last two
  live previews.

The second meaning is not safe for downstream LLM consumption. It is a useful
display heuristic, but it does not preserve enough history, timing, ASR
metadata, or boundary state to make non-revisable text commitments. The current
logic lives inside `AudioToTextRecorder._realtime_worker()` and is coupled to
mutable frame buffers, callback dispatch, VAD state, wake-word trimming, and
final transcription timing.

The next implementation should treat stabilization as a separate product/API
surface: a per-recording text-history subsystem that receives structured
realtime observations, emits stable deltas only when evidence is strong enough,
and exposes unstable preview text separately. The recorder and FastAPI server
should feed that subsystem with accurate metadata; they should not become the
place where the text algorithm grows.

## Current Realtime Behavior

`AudioToTextRecorder.__init__()` accepts realtime options and callbacks at
`RealtimeSTT/audio_recorder.py:262` through
`RealtimeSTT/audio_recorder.py:271`, including:

- `enable_realtime_transcription`
- `use_main_model_for_realtime`
- `realtime_transcription_engine`
- `realtime_model_type`
- `realtime_processing_pause`
- `init_realtime_after_seconds`
- `on_realtime_transcription_update`
- `on_realtime_transcription_stabilized`

The constructor stores realtime state at `RealtimeSTT/audio_recorder.py:642`
through `RealtimeSTT/audio_recorder.py:746`. Important fields include
`recording_start_time`, `recording_stop_time`, `wake_word_detect_time`,
`speech_end_silence_start`, `text_storage`,
`realtime_stabilized_safetext`, `awaiting_speech_end`, realtime counters, and
boundary-scheduler options.

Recorder initialization starts both the recording worker and realtime worker at
`RealtimeSTT/audio_recorder.py:1046` through
`RealtimeSTT/audio_recorder.py:1054`. The realtime worker therefore runs as a
background daemon for the lifetime of the recorder, even when it spends most of
its time idling.

When recording starts, `start()` resets `text_storage`,
`realtime_stabilized_text`, and `realtime_stabilized_safetext` at
`RealtimeSTT/audio_recorder.py:1935` through
`RealtimeSTT/audio_recorder.py:1938`, clears `self.frames`, then stores a new
`recording_start_time` at `RealtimeSTT/audio_recorder.py:1946`. There is no
explicit recording id in the recorder core.

During recording, `_recording_worker()` appends audio chunks to `self.frames`
at `RealtimeSTT/audio_recorder.py:2540` through
`RealtimeSTT/audio_recorder.py:2545`. It also appends pre-roll and silence-tail
audio to `audio_buffer`, starts recording after VAD/wake-word checks, and stops
recording after `post_speech_silence_duration`.

`_realtime_worker()` snapshots the entire current `self.frames` list, converts
it into float32 audio, transcribes the whole current buffer, then publishes the
result. This is done by nested helpers around
`RealtimeSTT/audio_recorder.py:2613` through
`RealtimeSTT/audio_recorder.py:2906`. It does not transcribe only the newest
audio delta.

The realtime worker has two scheduling modes:

- Timer-based scheduling from `realtime_processing_pause`, ending at
  `RealtimeSTT/audio_recorder.py:3066` through
  `RealtimeSTT/audio_recorder.py:3084`.
- Acoustic boundary scheduling through `RealtimeSpeechBoundaryDetector` at
  `RealtimeSTT/audio_recorder.py:2908` through
  `RealtimeSTT/audio_recorder.py:3042`.

Both modes skip realtime transcription while `awaiting_speech_end` is true, as
seen at `RealtimeSTT/audio_recorder.py:3057` through
`RealtimeSTT/audio_recorder.py:3060` and
`RealtimeSTT/audio_recorder.py:3080` through
`RealtimeSTT/audio_recorder.py:3082`. This matters because the silence tail is
often when a left-side prefix has the strongest evidence.

Publishing current realtime text is concentrated in
`_publish_realtime_text()` at `RealtimeSTT/audio_recorder.py:2795` through
`RealtimeSTT/audio_recorder.py:2863`. The current behavior is:

1. Strip the ASR result.
2. Drop it if empty, not currently recording, or still within
   `init_realtime_after_seconds`.
3. Save it as `self.realtime_transcription_text`.
4. Append it to `self.text_storage`.
5. Compute the exact character common prefix of the last two stored texts.
6. Grow `self.realtime_stabilized_safetext` if that prefix is longer than the
   current safetext.
7. Merge safetext with the current realtime text using
   `_find_tail_match_in_text()`.
8. Call the internal stabilized callback, then the internal update callback.

The public realtime callbacks are thin wrappers at
`RealtimeSTT/audio_recorder.py:3587` through
`RealtimeSTT/audio_recorder.py:3622`. Both are gated by `self.is_recording`.
`_run_callback()` at `RealtimeSTT/audio_recorder.py:1167` through
`RealtimeSTT/audio_recorder.py:1173` optionally dispatches callbacks in a new
thread if `start_callback_in_new_thread` is enabled.

Final transcription is separate. `stop()` deep-copies `self.frames`, queues the
stopped recording, clears `self.frames`, and stores stop times at
`RealtimeSTT/audio_recorder.py:1981` through
`RealtimeSTT/audio_recorder.py:2001`. `wait_audio()` later consumes queued
recordings at `RealtimeSTT/audio_recorder.py:1560` through
`RealtimeSTT/audio_recorder.py:1608`, and `perform_final_transcription()`
submits the final audio at `RealtimeSTT/audio_recorder.py:1730` through
`RealtimeSTT/audio_recorder.py:1784`.

## Source Map And Responsibilities

`RealtimeSTT/audio_recorder.py`

- Public recorder API, constructor options, callbacks, timing fields, and state.
- `start()`, `stop()`, `wait_audio()`, `text()`, and
  `perform_final_transcription()` define the final-transcription lifecycle.
- `_recording_worker()` owns microphone/external audio queue consumption,
  pre-roll, VAD checks, wake-word interaction, frame appends, silence timing,
  early final transcription, and recording stop.
- `_realtime_worker()` owns realtime scheduling, frame snapshots, realtime ASR
  calls, legacy text stabilization, and realtime callback publication.
- `_preprocess_output()` mutates display text casing and final punctuation.
- `_find_tail_match_in_text()` is an exact-character helper used by the legacy
  stabilized callback.

`RealtimeSTT/realtime_boundary_detector.py`

- A focused streaming acoustic-boundary detector that does not depend on ASR or
  recorder internals.
- `SpeechBoundaryEvent` stores `boundary_sample`,
  `boundary_time_seconds`, `score`, `reason`, `latency_ms`, and `created_at`
  at `RealtimeSTT/realtime_boundary_detector.py:23` through
  `RealtimeSTT/realtime_boundary_detector.py:61`.
- `RealtimeSpeechBoundaryDetector` emits boundary candidates from 10 ms energy
  and voicing windows, not linguistic syllables with certainty. The module
  docstring states this clearly at `RealtimeSTT/realtime_boundary_detector.py:1`
  through `RealtimeSTT/realtime_boundary_detector.py:9`.

`RealtimeSTT/transcription_engines/base.py`

- `TranscriptionResult` currently contains only `text` and
  `TranscriptionInfo` at `RealtimeSTT/transcription_engines/base.py:6` through
  `RealtimeSTT/transcription_engines/base.py:15`.
- There is no common place for word timestamps, token confidence, segment
  offsets, partial/final flags, or engine revision ids.

`example_fastapi_server/server.py`

- `ServerSettings` includes realtime engine, callback, VAD, queue, and latency
  options at `example_fastapi_server/server.py:98` through
  `example_fastapi_server/server.py:149`.
- `SegmentState` maps realtime and final events onto segment ids at
  `example_fastapi_server/server.py:158` through
  `example_fastapi_server/server.py:184`.
- `InferenceJob` and `InferenceResult` carry request/session/kind/segment,
  queue timing, inference timing, and latency at
  `example_fastapi_server/server.py:225` through
  `example_fastapi_server/server.py:257`.
- `FairInferenceQueue` coalesces pending realtime work but preserves final work;
  tests cover this at `tests/unit/test_fastapi_server_protocol.py:208` through
  `tests/unit/test_fastapi_server_protocol.py:229`.
- `RecorderBackedRealtimeSession` creates one recorder per session, passes
  scheduler-backed transcription executors, chooses either the realtime update
  or stabilized recorder callback, and publishes websocket `realtime` messages
  at `example_fastapi_server/server.py:1278` through
  `example_fastapi_server/server.py:1357` and
  `example_fastapi_server/server.py:1589` through
  `example_fastapi_server/server.py:1605`.
- `transcribe_for_recorder()` turns recorder ASR calls into scheduler jobs at
  `example_fastapi_server/server.py:2002` through
  `example_fastapi_server/server.py:2069`.

`example_fastapi_server/protocol.py`

- Audio packets carry metadata plus PCM bytes, but no capture timestamp,
  packet sequence, recording id, or client clock information. See
  `example_fastapi_server/protocol.py:14` through
  `example_fastapi_server/protocol.py:81`.

`tests/unit/*`

- Existing tests cover boundary detector behavior, slow final-transcription
  audio retention, FastAPI protocol parsing, queue coalescing, stale result
  rejection, session isolation, and optional engine behavior.
- There is no focused unit test for the current legacy realtime stabilization
  logic in `_publish_realtime_text()`, partly because it is nested inside
  `_realtime_worker()`.

## Product Use Cases

### Ordinary Use Cases

- A command-line user wants live captions while speaking a sentence. They need
  the latest text to remain lively, even when the right edge is wrong.
- A browser demo user wants stable text styled differently from the provisional
  right edge. The stable left side should not flicker when the ASR adds a comma.
- A dictation user speaks slowly. The display should show "inter-" or
  "transcrip-" as provisional, then only commit the fragment or word when enough
  evidence exists.
- A user pauses naturally after "I want to book a". The stabilizer should not
  commit a trailing space that prevents "book an" or "book a.m." style
  correction if the ASR revises the phrase.
- A desktop assistant wants to start processing stable fragments before final
  transcription, but still show the provisional full line to the user.

### Downstream LLM Use Cases

- Voice-to-LLM streaming: send stable deltas to an LLM as non-revisable input
  while keeping unstable text out of the prompt.
- Tool invocation: commit "turn on the kitchen" only when it is stable enough,
  and avoid prematurely committing "lights" if the ASR may revise it to
  "light strip".
- Agent planning: allow the LLM to begin semantic parsing on stable left-side
  text while waiting for final confirmation of names, numbers, and negations.
- Meeting assistant: stream stable transcript to note-taking logic without
  forcing the note-taker to handle retractions.
- Voice coding: avoid sending a half-stabilized identifier or import path as a
  final edit instruction.

### API And Integration Use Cases

- Existing callback users continue receiving `on_realtime_transcription_update`
  as a plain string.
- Existing `on_realtime_transcription_stabilized` users do not suddenly receive
  incompatible object payloads.
- Advanced users can opt into structured events that include `stable_delta`,
  `stable_text`, `unstable_text`, `display_text`, segment id, sequence id,
  timing, trigger reason, and ASR latency.
- Applications that only want stable LLM input can subscribe to stable deltas
  rather than repeatedly receiving the full stable prefix.
- Applications that render UI can subscribe to both stable and unstable text.

### Server And Multi-User Use Cases

- Multiple browser users stream audio at the same time. Each session must keep
  independent stable state, even if shared inference workers coalesce jobs.
- A user clears the session while a realtime job is pending. Late text from the
  old generation must not enter the new stable state.
- The server is overloaded and drops stale realtime jobs. Stability should
  degrade by becoming more conservative, not by pretending dropped observations
  were negative evidence.
- A browser client reconnects. The server should have enough segment/stable
  state to decide whether to replay committed stable text, only final text, or
  neither.
- A server operator chooses `--realtime-callback stabilized`; websocket clients
  still need to know whether they are receiving stable-only, full display text,
  or legacy stabilized text.

### Edge And Adversarial Use Cases

- Partial long word: "internationali..." is repeated several times before the
  user finishes "internationalization".
- Ambiguous word boundary: "ice cream" versus "I scream"; committing a space
  too early changes meaning.
- Homophones and numbers: "for" versus "four"; "two" versus "to"; "May eighth"
  versus "May 8th".
- Late negation: "do not send" where "do" appears stable early but the action
  should not be executed until enough context exists.
- Repeated phrase: "to be or not to be" can confuse suffix/prefix alignment.
- Self-correction: "send that to Alice, no, Bob" should not cause the
  stabilizer to treat "Alice" as non-revisable too early.
- ASR hallucination: one update suddenly returns unrelated text, then the next
  update returns to the prior sentence.
- Punctuation churn: "hello world", "hello, world", and "Hello world" should
  often count as the same evidence for text identity, but punctuation should
  not be committed blindly.
- Casing churn from `_preprocess_output()`: display casing should not become
  comparison evidence.
- CJK or Thai text: spaces are not reliable word boundaries, so a word-space
  policy must be language-aware or configurable.
- Right-to-left text: display merging and offsets need clear definitions if
  structured spans are exposed.
- Background speech or TV audio causes VAD to continue a recording beyond the
  intended utterance.
- Wake-word mode removes leading samples after recording begins; the stabilizer
  must not anchor text to audio offsets before trimming is accounted for.
- Final transcription contradicts already committed stable text.

## Failure Modes And Loopholes In Current Code

The current "stabilized" callback is not a non-revisable stable text stream.

- It uses only the exact character common prefix of the last two realtime texts
  at `RealtimeSTT/audio_recorder.py:2819` through
  `RealtimeSTT/audio_recorder.py:2828`.
- It has no minimum observation count beyond two adjacent updates.
- It has no elapsed-time threshold.
- It has no fragment age, confidence, or audio-span evidence.
- It ignores all older history except for the already-grown safetext.
- A single pair of similar ASR outputs can commit a wrong prefix.

`text_storage` is too weak as history.

- It stores only strings, no timestamp, trigger reason, audio length, frame
  count, sample offsets, language, queue delay, or inference duration.
- It grows for the duration of a recording with no cap at
  `RealtimeSTT/audio_recorder.py:2817`.
- It cannot support rules such as "validated 3 times within 300 ms" because it
  records neither arrival times nor ASR request times.

The current comparison is exact-character comparison.

- `os.path.commonprefix()` is sensitive to punctuation, case, whitespace, and
  partial-byte text assumptions.
- `_find_tail_match_in_text()` at `RealtimeSTT/audio_recorder.py:3542` through
  `RealtimeSTT/audio_recorder.py:3585` uses an exact 10-character suffix match.
  It fails on short strings, punctuation changes, casing changes, and repeated
  phrases.
- `_find_tail_match_in_text()` documents that it returns the match start, but
  the implementation returns the position after the matched substring. The
  current caller may rely on that behavior, so this is fragile.

Spaces can be committed too early.

- Character-prefix logic cannot distinguish a stable word fragment from a
  stable word boundary.
- If two adjacent previews both include "inter " before the ASR later revises
  to "internet", the current logic has no guardrail beyond exact prefix growth.

The stabilized callback sends full text repeatedly.

- `_publish_realtime_text()` invokes `_on_realtime_transcription_stabilized()`
  on every accepted realtime update at `RealtimeSTT/audio_recorder.py:2835`
  through `RealtimeSTT/audio_recorder.py:2858`.
- There is no stable-delta callback. A downstream consumer would need to diff
  repeated strings and guess whether the prefix really changed.

The worker can miss valuable observations.

- `_publish_realtime_text()` drops early realtime results during
  `init_realtime_after_seconds` at `RealtimeSTT/audio_recorder.py:2812`.
  Those observations may be too early to publish, but they may still be useful
  history.
- `_realtime_worker()` skips realtime transcription while
  `awaiting_speech_end` is true. The final silence window may be exactly when a
  fragment has stopped changing and should be committed or flushed.

Mutable frame buffers are not an explicit concurrency boundary.

- `_snapshot_frames()` looks for optional `frames_lock`, `frame_lock`, or
  `audio_lock` at `RealtimeSTT/audio_recorder.py:2619` through
  `RealtimeSTT/audio_recorder.py:2632`, but the current constructor does not
  initialize such a lock in the inspected code.
- `_recording_worker()` appends to `self.frames`, `start()` replaces it, and
  `stop()` deep-copies then clears/replaces it. The realtime worker defensively
  snapshots tuples, but stabilization metadata should not assume a perfect
  lock-step audio timeline unless one is explicitly captured.

Recorder boundaries are implicit.

- The recorder has `recording_start_time` and `recording_stop_time`, but no
  monotonically increasing recording id.
- `start()` resets text state, but late callback work or server jobs need a
  stable generation/segment identifier to avoid mixing observations.
- The FastAPI server has `SegmentState`, but that id does not exist in the core
  recorder API.

Final transcription can contradict stable text.

- Final transcription uses a different engine/model path in many deployments.
  The realtime engine may be tiny or fast; final may be larger and more
  accurate.
- `perform_final_transcription()` returns preprocessed final text with final
  punctuation behavior, while realtime callbacks use preview preprocessing.
- There is no reconciliation contract between stable realtime text and final
  text. This is acceptable for a live preview but dangerous for non-revisable
  downstream output.

The server protocol hides stability semantics.

- `RecorderBackedRealtimeSession._create_recorder()` chooses either
  `on_realtime_transcription_update` or
  `on_realtime_transcription_stabilized` based on `settings.realtime_callback`
  at `example_fastapi_server/server.py:1285` through
  `example_fastapi_server/server.py:1289`.
- `_on_realtime_text()` always publishes a websocket message of type
  `"realtime"` with only `text`, `timestamp`, and `segmentId` at
  `example_fastapi_server/server.py:1589` through
  `example_fastapi_server/server.py:1605`.
- A client cannot tell whether `text` is raw provisional text, legacy
  stabilized text, stable-only text, or a full display merge.

Server coalescing changes the evidence stream.

- `FairInferenceQueue` intentionally replaces older pending realtime jobs with
  newer realtime jobs for the same session. This protects latency, but it means
  the stabilizer may see fewer observations when the system is loaded.
- Stale realtime jobs are dropped based on `deadline_at`. Missing observations
  should make the algorithm slower, not less safe.

Callback exceptions can change publication shape.

- `_publish_realtime_text()` calls the stabilized callback before the update
  callback. If the stabilized callback raises and callbacks are not run in a new
  thread, the update callback for the same observation may be skipped by the
  exception path and only logged by the worker loop.

## Timing Metadata Needed For Stabilization

A future observation object should carry enough metadata to answer both text and
audio questions. Candidate fields:

- `recording_id`: monotonic id assigned by recorder core at `start()`.
- `segment_id`: server-visible segment id when available.
- `sequence`: per-recording realtime observation sequence.
- `trigger_reason`: timer, syllable-boundary, follow-up, fallback, manual,
  server inline, or final-flush.
- `raw_text`: exact ASR output before `_preprocess_output()`.
- `display_text`: optional preview-processed text.
- `engine_name` and `model_name`.
- `language` and `language_probability`.
- `observation_created_at_monotonic`: when the worker decided to transcribe.
- `observation_completed_at_monotonic`: when ASR returned.
- `published_at_wall_time`: for UI timestamps and logs.
- `queue_delay_seconds`, `inference_duration_seconds`,
  `total_latency_seconds`.
- `recording_started_at_monotonic` and `recording_started_at_wall_time`.
- `audio_start_sample` and `audio_end_sample` relative to the logical
  recording, after pre-roll and wake-word trimming are accounted for.
- `audio_duration_seconds` for the ASR buffer.
- `frame_count` and `sample_count` in the snapshot sent to ASR.
- `pre_roll_sample_count` included in the recording.
- `wake_word_detected_at`, `wake_word_trimmed_samples`, and whether the ASR
  buffer still includes any wake-word audio.
- `speech_end_silence_started_at` and `awaiting_speech_end` at observation
  time.
- `boundary_event` metadata when boundary scheduling triggered the observation:
  boundary sample, boundary time, score, reason, and detector latency.
- `dropped_or_skipped_reason` for observations attempted but not published:
  empty text, init delay, not recording, stale queue, callback rejection, or
  ASR error.

The algorithm should use monotonic time for age and latency calculations.
Wall-clock time is useful for human-visible logs and websocket timestamps, but
it can jump. The recorder currently mixes `time.time()` with server
`time.monotonic()` in different places, so a design should name each field
explicitly.

Sample offsets are more stable than wall-clock offsets for audio-region
association. They are also easier to test. The server already validates audio
packet frame counts in `packet_to_server_samples()` tests at
`tests/unit/test_fastapi_server_protocol.py:171` through
`tests/unit/test_fastapi_server_protocol.py:206`, but the protocol does not
yet carry client capture timestamps or packet sequence ids.

## Candidate Stabilization Concepts

### 1. Separate Stabilizer Module

Create a standalone text-history subsystem that can be unit-tested without
audio devices, ASR engines, threads, or FastAPI. It should consume observation
objects and return a structured result:

- stable text already committed
- stable delta newly committed by this observation
- unstable current suffix
- display text
- ignored/outlier flag
- reason codes and metrics

This matches the architecture guide's direction: new cohesive modules should
be focused, typed, and separate from recorder/server runtime concerns. It also
avoids making `audio_recorder.py` larger.

Tradeoff: the recorder integration still needs to collect metadata and call the
stabilizer at the right boundaries. The hard part moves out of the worker, but
the worker still needs careful observation capture.

### 2. Monotonic Stable Frontier

Maintain a stable frontier into the logical transcript. Once text before that
frontier is committed, it can only grow. The algorithm emits only the delta
between the previous frontier and the new frontier.

Tradeoff: final transcription may later disagree. The design must decide
whether to log, expose, or separately report that divergence without revising
stable text.

### 3. Normalized Alignment With Raw-Text Projection

Compare normalized text while emitting raw/display text. Normalization can
collapse whitespace, lowercase, and optionally ignore punctuation for evidence.
The stabilizer then maps normalized spans back to raw text spans before
committing.

Tradeoff: projection is harder than exact prefix comparison. It is also where
most subtle bugs live: repeated words, punctuation, Unicode normalization, and
languages without spaces need tests.

### 4. Evidence Window

Require a fragment to appear consistently across a minimum number of compatible
observations and a minimum elapsed time. The example from the requirements,
"3 validations within 300 ms", is one possible input to this policy, but not a
complete rule.

Candidate knobs:

- minimum compatible observations
- minimum age of fragment
- maximum evidence window
- required overlap with current stable frontier
- punctuation and space confirmation thresholds
- outlier tolerance

Tradeoff: fixed thresholds behave differently across engines and hardware. A
fast realtime engine may produce many observations in 300 ms; a slow model or
loaded server may produce one. The policy should account for both count and
time, and should degrade conservatively when observations are sparse.

### 5. Word-Fragment And Space Policy

Treat spaces as separate commitment decisions. A fragment may become stable
without the following space. A space should require evidence that the word
boundary is safe, such as repeated right-context text, an acoustic boundary, or
enough subsequent characters.

Tradeoff: this is English-friendly but not universal. The design needs a
language mode or a generic "separator policy" rather than assuming ASCII space
is always the word boundary.

### 6. Outlier And Hallucination Filtering

Classify an observation as an outlier when it has low similarity to both the
current stable prefix and recent compatible observations. A single outlier
should not advance the stable frontier. Multiple consecutive incompatible
observations may indicate a real segment reset, but only if the audio/recording
boundary supports that interpretation.

Tradeoff: a real self-correction can look like an outlier. The stabilizer must
avoid committing the correction too quickly while still showing it as unstable
text.

### 7. Boundary-Assisted Scheduling, Not Boundary-Based Commitment

`RealtimeSpeechBoundaryDetector` can improve realtime ASR timing by triggering
transcriptions near acoustic valleys. Its events can provide useful metadata:
boundary sample, created time, score, and latency. But a boundary is not proof
that text is stable. It should be evidence context, not a direct commit signal.

Tradeoff: using boundary events too aggressively could stabilize syllables or
spaces that ASR has not repeatedly validated.

### 8. Final-Transcription Reconciliation Layer

Final transcription should be associated with the same recording/segment as the
stable realtime history. Possible behaviors:

- Use final text only to emit the unstable remainder after the stable prefix,
  never to revise stable text.
- Publish final text separately and mark whether it agrees with committed
  stable text.
- Emit a diagnostic "stable_final_mismatch" event for logs/tests.
- Let applications choose whether final text replaces the visual transcript,
  while stable deltas remain non-revisable for LLM consumption.

Tradeoff: user-facing UI often wants the final transcript to be the best text,
but LLM-facing APIs need a non-revision contract. Those are different product
surfaces and should not be collapsed into one callback.

## API And Callback Implications

The current callback names are public and should be preserved. However, the
current signatures are too weak for the new product goal.

Compatibility approach:

- Keep `on_realtime_transcription_update(text)` as latest provisional full text.
- Keep `on_realtime_transcription_stabilized(text)` for compatibility, but
  decide whether it remains a legacy full-line display callback or becomes a
  true stable-prefix callback behind a version/option.
- Add a structured callback or event surface rather than overloading string
  callbacks. A name like `on_realtime_text_stabilization_update(event)` would
  make the behavior explicit.
- Add a stable-delta surface for LLM consumers. Repeated full-prefix strings
  force downstream users to diff and can cause duplicate prompt text.
- Keep raw ASR text and display-processed text separate. `_preprocess_output()`
  changes casing and final punctuation rules, so it should not be the source of
  comparison evidence.

Candidate structured event fields:

- `recording_id`
- `segment_id`
- `sequence`
- `raw_text`
- `display_text`
- `stable_text`
- `stable_delta`
- `unstable_text`
- `stable_until_normalized_offset` or equivalent span marker
- `is_outlier`
- `trigger_reason`
- `timing`
- `language`
- `engine`
- `final_consistency` when applicable

Threading matters. If callbacks can run in new threads, the event object should
be immutable or treated as immutable. If callbacks run inline, the recorder
should decide whether a user callback exception can prevent other callbacks for
the same observation.

## FastAPI And Protocol Implications

The FastAPI server currently has a single `realtime` websocket event shape for
both raw and stabilized callback modes. Stabilization needs a clearer protocol.

Recommended protocol direction:

- Keep existing `"type": "realtime"` messages compatible for current clients.
- Add optional fields such as `stableText`, `stableDelta`, `unstableText`,
  `displayText`, `sequence`, `recordingId`, `isStable`, `isFinalForSegment`,
  and timing metadata.
- Consider a distinct `"type": "stable"` or `"type": "stable_delta"` event for
  downstream consumers that only want committed text.
- Include `segmentId` on every realtime/stable/final event, as the server
  already does.
- Add a per-session `sequence` for realtime events. `timestamp` alone is not a
  reliable ordering contract across queued work, callbacks, and websocket
  delivery.
- Decide whether final events include `agreesWithStableText` or a mismatch
  diagnostic.

Server-side placement needs care:

- If the recorder is the only realtime source, stabilization can live in the
  recorder and the server can forward structured events.
- If `RealtimeSession` remains a supported non-recorder path, it needs the same
  stabilizer module per session. Otherwise the two server modes will have
  different semantics.
- Recorder-backed sessions call scheduler-backed executors from the recorder
  realtime worker. `transcribe_for_recorder()` has request ids and timing, but
  the final websocket event published by `_on_realtime_text()` currently loses
  that detail.
- Generation and clear handling must reset stabilizer state. Existing tests
  already assert stale final behavior for clears in
  `tests/unit/test_fastapi_server_multi_user.py:243` through
  `tests/unit/test_fastapi_server_multi_user.py:276`.

The server's queue policy should become visible to stabilization metrics:

- Realtime coalescing means fewer observations.
- Stale realtime drops mean missing observations, not contrary evidence.
- Queue delay and inference duration should be recorded with each observation
  because latency affects thresholds.

## Optional Engines And ASR Backend Variability

The current engine interface gives only text and language metadata. This is the
lowest common denominator and is enough for an initial stabilizer, but it limits
future quality.

Engine differences to expect:

- `faster_whisper` can produce segments internally, but the adapter currently
  joins segment text and discards segment timing.
- Some adapters may support word timestamps through engine-specific options,
  but `TranscriptionResult` does not expose them.
- Realtime and final engines may be different. A small realtime model may
  stabilize text that a larger final model later changes.
- Some engines add punctuation; others omit it.
- Some engines have strong prompt effects; others may ignore prompts.
- Some engines require language input; others detect language.
- Some CPU-only engines may return observations too slowly for fixed
  millisecond thresholds.

The stabilizer should therefore:

- Work with text-only observations.
- Accept optional richer timing/segment metadata later.
- Keep engine-specific details outside the core algorithm.
- Track engine/model names for diagnostics and threshold tuning.
- Avoid assuming that absence of punctuation or capitalization is evidence of
  instability.

## Performance, Latency, Memory, And Concurrency

Performance risks:

- Realtime ASR already transcribes the whole current frame each time. The text
  stabilizer must be cheap relative to ASR and should not add blocking work to
  `_realtime_worker()`.
- Long recordings can create long text and many observations. History retention
  needs caps or compaction.
- Alignment can become expensive if implemented as full dynamic programming
  over every previous observation. The common case is prefix-like growth; use
  fast paths before heavier alignment.
- Server overload reduces observation count. The stabilizer should become more
  conservative under sparse observations instead of adding retry work.

Concurrency risks:

- The recorder has no explicit frame lock in the inspected path, so observation
  metadata should be captured from the same snapshot passed to ASR.
- Callback dispatch may be inline or threaded.
- Server sessions have generations and cancellation paths; stabilizer state
  must reset with generation changes.
- Late ASR results should be rejected by id/generation before they touch
  stability state.

Memory strategy:

- Store full observation history only within a bounded current recording.
- Keep compact evidence state after committing stable text.
- Consider debug-only full history export for tests and diagnostics.
- Do not store audio bytes in the stabilizer by default; store sample offsets
  and hashes if needed for traceability.

## Test Strategy

### Stabilizer Unit Tests

These should be fast, deterministic, and independent of ASR/audio:

- Monotonic growth commits stable deltas after configured evidence.
- A fragment can commit without a trailing space.
- A trailing space requires stronger evidence than characters.
- Punctuation/case changes count as compatible evidence but do not force early
  punctuation commits.
- One hallucination update is ignored.
- Multiple unrelated updates do not revise already stable text.
- Repeated phrases do not confuse suffix alignment.
- Dropped observations make commits slower, not unsafe.
- Out-of-order or stale observations are ignored by sequence/recording id.
- Final mismatch is surfaced without revising stable deltas.
- Long recordings compact old history.
- Language modes handle no-space text without assuming word boundaries.

### Recorder Integration Tests

Because the current realtime logic is nested inside `_realtime_worker()`, the
implementation design should first extract a testable stabilization unit. Then
recorder integration tests can focus on metadata and boundary behavior:

- `start()` creates a new recording id and resets stabilizer state.
- `stop()` closes the current recording without mixing the next recording.
- `init_realtime_after_seconds` can suppress publication while still recording
  observations if that is the chosen design.
- `awaiting_speech_end` handling does not prevent a final stable flush.
- Pre-roll and wake-word trimming are reflected in observation sample offsets.
- Callback exceptions do not corrupt stabilization state.
- Legacy string callbacks still receive compatible values.

### FastAPI Tests

Build on the current multi-user tests:

- Realtime stable events are routed only to the owning session, similar to
  `tests/unit/test_fastapi_server_multi_user.py:463` through
  `tests/unit/test_fastapi_server_multi_user.py:480`.
- Clearing a session resets stabilizer state and ignores stale stable events.
- `realtime_callback` compatibility modes publish expected websocket shapes.
- Stable delta events include `segmentId` and monotonically increasing
  `sequence`.
- Queue coalescing and stale drops are counted in metadata but do not create
  false stable evidence.
- Server protocol tests assert new fields remain optional for older clients.

### Regression Scenarios With Audio Fixtures

Use synthetic observation sequences first, then selected audio fixtures:

- Slow final transcription while another recording completes, building on
  `tests/unit/test_slow_final_transcription_audio_gap.py:77` through
  `tests/unit/test_slow_final_transcription_audio_gap.py:159`.
- Boundary detector scheduling with stable observations after a vowel valley,
  building on `tests/unit/test_realtime_boundary_detector.py:8` through
  `tests/unit/test_realtime_boundary_detector.py:32`.
- Multi-session server stress with realtime and final latency reporting,
  building on `tests/unit/test_fastapi_server_multi_user_asr_integration.py`.

## Architecture Risks And Boundaries

Do not grow the algorithm inside `_realtime_worker()`.

The worker already owns scheduling, frame snapshots, ASR calls, callback
publication, and defensive error handling. Adding alignment, evidence windows,
language policies, and final reconciliation there would make it harder to test
and risk regressions in recording behavior.

Keep stabilization independent from ASR engines.

The stabilizer should not import engine adapters or know backend-specific
objects. It should consume project-level observations and optional metadata.

Keep server protocol separate from the stabilizer.

The stabilizer should return structured results. The server should translate
those into websocket messages. This preserves the architecture direction shown
by `example_fastapi_server/protocol.py`.

Keep VAD and wake-word events as metadata, not text decisions.

VAD and acoustic boundaries help explain when observations happen and where
audio regions begin/end. They should not be the only reason text becomes
non-revisable.

Protect compatibility.

Existing callbacks and server messages are likely used by applications. New
structured events should be additive unless a breaking change is explicitly
planned and documented.

Avoid final-text mythology.

Final transcription is usually better, but it is not a safe mechanism for
revising already committed stable deltas. The design must admit that UI-final
text and downstream-stable text are different products.

## Open Questions For Human Decision

- What exact stability policy should ship first: observation count, elapsed
  time, or a combined policy?
- Is the unit of stability characters, normalized tokens, word fragments, whole
  words, or a hybrid?
- When is a trailing space safe to commit?
- Should punctuation ever be committed before final transcription?
- Should `on_realtime_transcription_stabilized(text)` remain legacy full-line
  behavior, become true stable prefix behavior, or be versioned behind an
  option?
- What should the new structured callback be called?
- Should downstream LLM consumers receive stable deltas only, or also segment
  lifecycle events?
- How should final transcription contradictions be surfaced?
- Should early realtime observations suppressed by `init_realtime_after_seconds`
  still be stored as evidence?
- Should there be a stable flush during `awaiting_speech_end` or only at
  recording stop?
- Should recorder core grow a public `recording_id` or keep ids private and let
  the server own `segmentId`?
- What metadata belongs in library callbacks versus server websocket events?
- How much history should be retained for diagnostics?
- Are non-space languages in scope for the first implementation, or should the
  first version document an English/space-tokenized policy?
- Should engine adapters expose optional word/segment timing in
  `TranscriptionResult`, or should stabilization start text-only?
- How should callback exceptions be isolated so one subscriber does not prevent
  another realtime event from publishing?

## Recommended Next-Step Plan

1. Define the first structured observation/result dataclasses in an
   implementation design document. Include exact units and clock semantics.
2. Decide the compatibility contract for existing realtime callbacks.
3. Design a standalone stabilizer module with synthetic text-observation tests
   before touching recorder logic.
4. Define the first stability policy in terms of count, elapsed time, normalized
   alignment, outlier handling, and space handling.
5. Define recording/segment boundary behavior for start, stop, clear, final,
   wake-word activation, and silence tail.
6. Decide FastAPI websocket event additions and compatibility behavior for old
   clients.
7. Add focused tests for the stabilizer first, then recorder metadata capture,
   then FastAPI routing/protocol behavior.
8. Only after those decisions, integrate the stabilizer into
   `_realtime_worker()` with minimal source changes and no VAD frame-size
   changes.
