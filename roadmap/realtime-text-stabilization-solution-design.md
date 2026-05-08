# Realtime Text Stabilization Solution Design

This document defines the proposed implementation shape for Realtime Text
Stabilization in RealtimeSTT.

It is an implementation-design document, not implementation code. It should be
used to write the unit tests first, then implement the stabilizer until those
tests pass, then integrate the stabilizer into the recorder and FastAPI server.

## Design Goals

- Keep text stabilization out of `RealtimeSTT/audio_recorder.py` except for
  observation capture and callback/event dispatch.
- Make the core stabilizer perfectly testable without audio devices, ASR
  engines, threads, callbacks, FastAPI, wall-clock sleeps, or real time.
- Preserve two different product surfaces:
  - stable non-revisable text for downstream consumers
  - unstable live preview text for display
- Emit stable deltas incrementally before final transcription when evidence is
  strong enough.
- Never revise or withdraw already-emitted stable text.
- Treat final transcription as a separate lifecycle event that can agree or
  disagree with committed stable text without rewriting it.

## Proposed Module

Add a focused module later:

`RealtimeSTT/realtime_text_stabilizer.py`

The module should contain only pure text-stabilization logic and small typed
data models. It should not import `AudioToTextRecorder`, ASR engines, FastAPI,
threading helpers, or callback dispatch helpers.

Primary class:

`RealtimeTextStabilizer`

Primary public methods:

- `reset(recording_id, segment_id=None, started_at_monotonic=None,
  started_at_wall_time=None)`: starts a new stabilization history.
- `observe(observation) -> RealtimeTextStabilizationEvent`: accepts one realtime
  transcription observation and returns the full event state for that
  observation.
- `finalize(final_observation=None) -> RealtimeTextFinalizationEvent`: closes
  the recording/segment and reports final-text consistency without revising
  committed stable text.
- `snapshot() -> RealtimeTextStabilizationSnapshot`: returns current stable,
  unstable, display, sequence, and diagnostic state for tests/debugging.

The stabilizer should return event objects. It should not call callbacks itself.
Recorder and server integration should translate returned events into callbacks,
websocket messages, logs, and metrics.

## Data Model

### `RealtimeTextObservation`

One object per realtime ASR output. It is immutable and fully supplied by the
caller.

Required fields:

- `recording_id`: monotonic id assigned by recorder core when recording starts.
- `sequence`: monotonically increasing realtime observation sequence within the
  recording.
- `raw_text`: exact ASR output before `_preprocess_output()`.
- `audio_start_sample`: first sample in the ASR input, relative to the logical
  recording/segment audio origin.
- `audio_end_sample_exclusive`: one-past-last sample in the ASR input, relative
  to the same origin.
- `sample_rate`: sample rate for the sample offsets.
- `created_at_monotonic`: when the realtime transcription request was created.
- `completed_at_monotonic`: when the ASR result became available.

Recommended optional fields:

- `segment_id`: server-visible segment id when available.
- `audio_start_time_seconds` and `audio_end_time_seconds`, derived from sample
  offsets for convenience.
- `recording_started_at_monotonic` and `recording_started_at_wall_time`.
- `received_at_wall_time`: human-facing timestamp for logs/UI.
- `trigger_reason`: `timer`, `syllable-boundary`, `syllable-boundary-followup`,
  `syllable-boundary-fallback`, `final-flush`, `manual`, or server-specific
  values.
- `display_text`: optional preview-formatted text if an integration wants to
  carry it; comparison must still use `raw_text`.
- `language` and `language_probability`.
- `engine_name` and `model_name`.
- `queue_delay_seconds`, `inference_duration_seconds`,
  `total_latency_seconds`.
- `frame_count` and `sample_count` for the exact snapshot sent to ASR.
- `publish_allowed`: false for observations that should count as evidence but
  should not publish user callbacks yet, such as observations inside
  `init_realtime_after_seconds`.
- `awaiting_speech_end`: recorder state at observation time.
- `boundary_event`: optional copy of acoustic boundary metadata. This is
  evidence context, not direct proof of text stability.
- `source`: `realtime`, `server-realtime`, or other source label.

Important timing rule:

The stabilizer uses monotonic timestamps and sample offsets. It does not call
`time.time()` or `time.monotonic()` internally during normal operation. Tests
provide deterministic timestamps.

Important audio-offset rule:

The current realtime worker transcribes the whole current `self.frames`
snapshot, so initial integration will normally set `audio_start_sample = 0` and
`audio_end_sample_exclusive = len(snapshot_samples)` for the logical recording.
The model still supports future partial-window ASR by allowing nonzero starts.

### `RealtimeTextStabilizationEvent`

Returned for every accepted or ignored observation.

Fields:

- `recording_id`
- `segment_id`
- `sequence`
- `accepted`: true when the observation was used as evidence.
- `ignored_reason`: `None`, `outlier`, `stale-sequence`,
  `wrong-recording`, or `empty-text`.
- `publish_allowed`: copied from the observation.
- `should_publish`: false when the observation was accepted as evidence but the
  integration should suppress outward callbacks/messages.
- `raw_observation_text`
- `stable_text`: full committed non-revisable text after this observation.
- `stable_delta`: newly committed text since the previous event.
- `unstable_text`: provisional suffix from the latest accepted compatible
  preview after removing the stable prefix.
- `display_text`: readable merge of `stable_text + unstable_text`.
- `stable_normalized_offset`: normalized-text frontier for diagnostics.
- `stable_raw_end_offset`: raw-text offset in the current accepted preview when
  it can be mapped unambiguously.
- `has_new_stable_text`
- `is_outlier`
- `stable_prefix_conflict`: true when the latest text cannot be aligned with
  already committed stable text.
- `commit_reason`: `none`, `evidence-threshold`, `space-confirmed`,
  `punctuation-confirmed`, `final-flush`, or similar.
- `evidence`: compact diagnostic data, including confirmation count, first and
  last confirmation timestamps, and contributing sequence ids.
- `timing`: copied or summarized observation timing.

`stable_delta` is the primary downstream LLM payload. Consumers that want
committed text should append deltas, not repeatedly diff full stable prefixes.

### `RealtimeTextFinalizationEvent`

Returned when a recording/segment finishes.

Fields:

- `recording_id`
- `segment_id`
- `stable_text`
- `final_text`
- `final_suffix_after_stable`: suffix of final text after compatible stable
  prefix matching, if available.
- `agrees_with_stable_prefix`
- `mismatch_reason`
- `stable_text_was_revised`: always false.
- `commit_final_remainder`: false by default for realtime stabilization events;
  final text should normally be published on the final-transcription surface.

Finalization must never edit `stable_text`. If final text contradicts committed
stable text, report the mismatch and leave reconciliation to product/UI policy.

## Stabilizer State

`RealtimeTextStabilizer` should maintain:

- current `recording_id` and optional `segment_id`
- last processed sequence
- committed `stable_text`
- current accepted raw preview
- current `unstable_text`
- recent accepted observations
- recent ignored/outlier observations for diagnostics
- normalized alignment/evidence ledger
- consecutive outlier count
- compact history for the active recording

History should be bounded. A first implementation can retain recent raw
observations, such as the last 128 or 256, while compacting older evidence once
text is committed. A debug option can retain full history for tests or trace
exports.

## Text Normalization And Projection

Comparison should happen in normalized text, while output should be projected
back to display/raw text.

Initial normalization policy:

- Unicode normalize with a consistent form, preferably NFKC.
- Casefold for comparison.
- Collapse all whitespace runs to one ASCII space for comparison.
- Treat common punctuation as optional for evidence matching.
- Keep punctuation positions in the projection map; do not simply delete them
  without retaining raw offsets.
- Preserve raw text for output.

Projection policy:

- Build a normalized-to-raw offset map for each observation.
- Determine stable frontiers in normalized space.
- Project the frontier back to a raw offset only when the mapping is
  unambiguous.
- If projection is ambiguous because of repeated words or punctuation churn,
  choose the conservative shorter frontier and report that in diagnostics.

This design deliberately avoids direct `os.path.commonprefix()` as the
stability rule. Exact prefix matching can remain as a fast path, but the public
behavior must be defined by normalized alignment plus evidence thresholds.

## Stability Policy

The first implementation should make stability policy configurable, but tests
should pin one default profile.

Recommended default profile:

- `min_char_confirmations = 3`
- `min_char_evidence_span_seconds = 0.20`
- `max_char_evidence_window_seconds = 1.50`
- `space_min_confirmations = 4`
- `space_min_evidence_span_seconds = 0.30`
- `space_requires_stable_right_context = true`
- `space_right_context_min_chars = 2`
- `punctuation_min_confirmations = 4`
- `punctuation_requires_stable_right_context = true`
- `outlier_similarity_threshold = 0.35`
- `max_single_outlier_gap = 1`

Character rule:

A non-space, non-punctuation character can be committed when it appears at the
same normalized position in at least `min_char_confirmations` compatible
observations, and the time between the first and latest contributing
observation is at least `min_char_evidence_span_seconds`.

The oldest contributing observation should not be older than
`max_char_evidence_window_seconds` unless it is already part of compacted
stable evidence. Sparse observations should make the stabilizer slower, not
less safe.

Space rule:

A space is a separate commit decision. Do not commit a trailing space merely
because the preceding characters are stable. A space can be committed only when
the space itself has enough evidence and there is stable right context after
the space. This avoids forcing a word boundary for pauses in the middle of long
words.

Punctuation rule:

Punctuation can count as compatible evidence when absent or changed, but it
should not be committed just because nearby words are stable. Commit punctuation
only when it is repeatedly present in compatible observations and has stable
right context, or when a future final-text policy explicitly commits final
punctuation.

Fragment rule:

The stabilizer may commit word fragments at the right edge when the fragment
itself satisfies the character rule. It must not attach a trailing space to a
fragment unless the space rule also passes. If the user later continues the
word, the continuation is appended as a later stable delta.

Stable frontier rule:

The stable frontier is monotonic. It can only move right. Later observations may
be ignored, marked as conflicting, or used for unstable display, but they cannot
move the frontier left.

## Stable And Unstable Output

For every observation, the event should expose:

- `stable_text`: committed non-revisable left side.
- `stable_delta`: newly committed text, if any.
- `unstable_text`: current provisional right side.
- `display_text`: stable and unstable merged into one readable line.

The merge must avoid duplicating text when current preview overlaps with stable
text in a case-insensitive or punctuation-insensitive way.

If the latest accepted preview cannot be aligned with the stable prefix:

- Keep `stable_text` unchanged.
- Set `stable_prefix_conflict = true`.
- Prefer the previous compatible `display_text` for stabilization-facing UI.
- Expose the raw conflicting observation in diagnostics so legacy live-preview
  surfaces can still show it if desired.

Outlier observations should not update `unstable_text` or `display_text` on the
stabilization event by default. This prevents one hallucinated update from
making the stable display flicker. The legacy raw realtime update callback can
still receive the raw latest preview separately.

## Outlier Handling

An observation is an outlier when it is both:

- incompatible with the committed stable prefix, if any
- low-similarity to recent accepted observations

A single outlier must:

- not add evidence
- not advance stable text
- not clear existing unstable text
- return an event with `is_outlier = true`

Multiple repeated incompatible observations may represent a real branch only
when they are consistent with each other. If `stable_text` is empty, the
stabilizer may adopt that branch after the configured outlier gap. If
`stable_text` is non-empty, the branch can only become unstable right-side text
after the committed stable prefix; it still cannot revise stable text.

## Casing, Punctuation, And Spaces

Casing:

- Case differences are compatible evidence.
- A case-only change must not create a stable delta.
- Committed casing should come from the canonical observation selected when the
  text is first committed, preferably the latest accepted observation that
  contributed to the threshold.
- Later casing changes are diagnostics only.

Punctuation:

- Missing, added, or changed punctuation should not block stabilization of
  surrounding words.
- Punctuation should be committed conservatively.
- A comma that appears after text has already been committed without a comma
  must not be inserted into `stable_text` later.

Spaces:

- Collapse whitespace runs for comparison.
- Output at most one ordinary space for a committed separator in the default
  space-tokenized policy.
- Never commit a right-edge trailing space unless the space rule passes.
- If a fragment is committed at the right edge, trim any following space from
  `stable_delta`.

Partial words:

- Partial words are allowed as stable text if evidence is strong enough.
- The algorithm must not imply the partial word is complete by committing the
  following space too early.
- Later continuation should append cleanly without requiring a correction.

Language mode:

The first implementation can default to a space-tokenized policy. The config
should include a language/separator mode so a future no-space language policy
can disable ASCII-space assumptions.

## Timestamp And Audio Offset Semantics

Use sample offsets for audio-region identity and monotonic time for evidence
age.

Input observation sample offsets answer: "What audio region was transcribed?"

They do not necessarily answer: "Exactly which audio sample produced this
stable character?" Text-only ASR does not provide that mapping. Therefore
output events should report observation evidence spans and optional text/audio
alignment fields, but should not pretend to know exact per-character audio
timestamps unless future engine metadata provides word or token timing.

Recommended event timing diagnostics:

- `evidence_first_sequence`
- `evidence_last_sequence`
- `evidence_first_completed_at_monotonic`
- `evidence_last_completed_at_monotonic`
- `evidence_audio_start_sample_min`
- `evidence_audio_end_sample_max`
- `stable_audio_end_sample_exclusive`: optional, `None` for text-only mode
  unless a conservative mapping is available

## Boundary And Recording Lifecycle

Each recording/segment has independent stabilizer state.

Required behavior:

- `reset()` clears all evidence and stable text for the new recording id.
- Observations from older recording ids are ignored as `wrong-recording`.
- Observations with sequence ids less than or equal to the last processed
  sequence are ignored as `stale-sequence`.
- `finalize()` closes the current recording but does not revise stable text.
- A new recording must never inherit unstable or evidence state from the
  previous one.

Acoustic boundary events:

- Boundary metadata may be stored with observations.
- Boundary-triggered observations can help produce more timely evidence.
- A boundary event alone must never commit text or spaces.

`awaiting_speech_end`:

- The current worker skips realtime ASR while awaiting speech end.
- Later integration should consider one final stabilization observation or
  finalization pass during the silence tail, because that period often contains
  useful evidence.
- This should be integration behavior, not special logic hidden in the core
  stabilizer.

`init_realtime_after_seconds`:

- Early observations may be useful evidence.
- Later integration should feed them to the stabilizer with
  `publish_allowed = false`, then suppress outward callbacks until publication
  is allowed.
- Tests should verify that suppressed evidence can contribute to the first
  published stable delta without emitting early callbacks.

## Public Callback/API Direction

Do not overload existing string callbacks with complex objects by default.

Compatibility recommendation:

- Keep `on_realtime_transcription_update(text)` as the raw/latest realtime
  preview string.
- Keep `on_realtime_transcription_stabilized(text)` compatible unless a
  versioned option changes it. The safest compatibility shape is a display line
  derived from the new stabilizer, not a stable-delta-only string.
- Add a new structured callback later, such as
  `on_realtime_text_stabilization_update(event)`.
- Add a stable-delta-oriented surface for LLM consumers, either as the same
  structured event or as a narrower callback receiving events with non-empty
  `stable_delta`.

Recommended recorder callback dispatch order after integration:

1. Update stabilizer state with the observation.
2. Dispatch structured stabilization event if configured.
3. Dispatch stable-delta callback only when `stable_delta` is non-empty.
4. Dispatch legacy stabilized callback according to compatibility mode.
5. Dispatch raw realtime update callback.

Callback exceptions should not corrupt stabilizer state. The state update must
complete before user callback execution.

## Recorder Integration Later

No recorder code should be changed during this design step. Later integration
should be small and mechanical.

Expected changes later:

- Instantiate one `RealtimeTextStabilizer` per recorder.
- Add a monotonic `recording_id` counter in recorder core.
- Call `stabilizer.reset(...)` from `start()`.
- Capture observation metadata in `_realtime_worker()` from the same frame
  snapshot sent to ASR.
- Use `time.monotonic()` for created/completed timing and `time.time()` only
  for wall-time UI fields.
- Replace the legacy common-prefix stabilization inside `_publish_realtime_text`
  with construction of `RealtimeTextObservation` and a call to
  `stabilizer.observe(...)`.
- Keep `_preprocess_output()` out of comparison evidence. It can still be used
  for legacy display strings.
- Preserve existing counters for realtime transcription attempts, successes,
  empties, and trigger reasons.
- Add counters for stabilizer accepted observations, outliers, stable deltas,
  and final mismatches.
- Ensure late ASR results are checked against current recording id/sequence
  before they can enter the stabilizer.

Observation capture details:

- `sequence` should increment when the ASR request is created.
- `created_at_monotonic` should be captured before ASR call/submission.
- `completed_at_monotonic` should be captured immediately after ASR result.
- `audio_start_sample` and `audio_end_sample_exclusive` should describe the
  frame snapshot sent to ASR, not callback publication time.
- If wake-word trimming or pre-roll changes the logical audio origin, offsets
  must be adjusted before building the observation.

## FastAPI Integration Later

FastAPI should forward structured semantics instead of hiding them inside the
single existing `"realtime"` text field.

Recommended websocket additions:

- Preserve existing `"type": "realtime"` messages for old clients.
- Add optional fields:
  - `recordingId`
  - `segmentId`
  - `sequence`
  - `stableText`
  - `stableDelta`
  - `unstableText`
  - `displayText`
  - `isOutlier`
  - `stablePrefixConflict`
  - `triggerReason`
  - `timing`
- Consider adding a separate `"type": "stable_delta"` message for clients that
  only want committed text.
- Include final mismatch diagnostics on final events when available.

Server-specific requirements:

- Each session owns its own stabilizer state.
- Session clear/reset increments generation and resets stabilization state.
- Stale scheduler results must be rejected before they reach the stabilizer.
- Realtime queue coalescing and stale drops are missing observations, not
  negative evidence.
- `segmentId` remains required on realtime/stable/final events.

Current server note:

`transcribe_for_recorder()` currently returns only `TranscriptionResult(text=...)`
to the recorder, losing scheduler timing details. Later integration should add
a way for recorder observations to receive queue delay, inference duration, and
request/sequence metadata without coupling the stabilizer to FastAPI.

## Test-First Plan

Create focused unit tests for the stabilizer before implementation code.

Suggested test file later:

`tests/unit/test_realtime_text_stabilizer.py`

Core stability tests:

1. Empty text returns an ignored event and does not change state.
2. First compatible observation updates unstable preview but emits no stable
   delta.
3. Second compatible observation still emits no stable delta when the default
   confirmation threshold is three.
4. Third compatible observation after the minimum evidence span emits the first
   stable delta.
5. Repeated observations after a commit return empty `stable_delta` until new
   text becomes stable.
6. Stable text grows monotonically and never shrinks.
7. Stable delta contains only new committed text, not the full stable prefix.
8. Dropped or missing observations make stabilization slower, not unsafe.

Stable/unstable display tests:

9. Stable left side and unstable right side are exposed separately.
10. `display_text` merges stable and unstable text without duplication.
11. If current preview overlaps stable text with different casing, unstable text
    starts after the overlap.
12. If current preview overlaps stable text with different punctuation,
    unstable text starts after the compatible overlap.
13. If current preview cannot align with stable text, the event reports
    `stable_prefix_conflict` without changing stable text.

Partial word and space tests:

14. A repeated fragment such as `inter` can become stable without a trailing
    space.
15. Repeated `inter ` does not commit the trailing space without right context.
16. Later `international` appends continuation after committed `inter`.
17. `book a` does not commit a trailing space after `a` when `an` is still
    plausible.
18. A space is committed only after stable right context appears.
19. Multiple whitespace characters normalize to one comparison space and one
    output space.
20. A right-edge trailing space is trimmed from `stable_delta`.

Casing and punctuation tests:

21. `hello`, `Hello`, and `HELLO` count as compatible evidence.
22. A case-only change after commit produces no new stable delta.
23. `hello world`, `hello, world`, and `Hello world` can stabilize the words.
24. A comma is not committed until the punctuation policy threshold passes.
25. Punctuation at the right edge is not committed prematurely.
26. Later punctuation cannot be inserted into the middle of already committed
    stable text.

Outlier tests:

27. One unrelated observation is marked as outlier and adds no evidence.
28. A stream returns to the prior text after one outlier and continues
    stabilizing from the previous accepted history.
29. Multiple unrelated observations before any stable text can start a new
    branch only after the configured outlier gap.
30. Multiple unrelated observations after stable text do not revise stable text.
31. Outlier display does not replace stabilization-facing `display_text` by
    default.

Ordering and boundary tests:

32. Out-of-order sequence is ignored.
33. Duplicate sequence is idempotent or ignored, according to final API choice.
34. Observation from wrong recording id is ignored.
35. `reset()` starts a clean history for a new recording id.
36. Boundary-triggered observations do not commit text unless text evidence
    thresholds pass.
37. `publish_allowed = false` observations affect evidence and return
    `should_publish = false` so integrations can suppress outward publication.

Audio/timing tests:

38. Evidence diagnostics include first/latest sequence ids and monotonic times.
39. Evidence diagnostics include min/max audio sample offsets from contributing
    observations.
40. A sparse observation sequence requires the same confirmation count and does
    not infer missing confirmations.
41. Text-only mode leaves exact per-character audio timestamp fields unset.

Repeated text and alignment tests:

42. `to be or not to be` does not confuse suffix/prefix overlap.
43. Repeated words do not cause duplicated unstable suffix.
44. A preview that drops a previously seen fragment and later restores it does
    not erase stable text.
45. A preview that repeats text accidentally does not double-commit it.

Finalization tests:

46. Final text that agrees with stable prefix reports agreement and no revision.
47. Final text that contradicts stable text reports mismatch and no revision.
48. Final suffix is exposed separately from realtime `stable_delta`.
49. Finalization closes the segment so later realtime observations are ignored
    until reset.

Language/history tests:

50. No-space language mode does not treat ASCII spaces as required word
    boundaries.
51. Long histories compact old observations after stable text is committed.
52. Debug history can expose enough observations for diagnostics when enabled.

Recorder integration tests later:

- `start()` creates a new recording id and resets stabilizer state.
- `_realtime_worker()` builds observations with correct sequence, trigger
  reason, monotonic times, and snapshot sample offsets.
- Early observations inside `init_realtime_after_seconds` can be stored as
  evidence without publishing callbacks.
- `awaiting_speech_end` does not permanently prevent a final stabilization or
  finalization pass.
- Wake-word/pre-roll offset adjustments are reflected in observation sample
  offsets.
- Legacy callbacks still receive strings.
- Callback exceptions do not corrupt committed stable state.

FastAPI tests later:

- Stable events route only to the owning session.
- Session clear resets stabilizer state and rejects stale pending results.
- Websocket messages include stable/unstable fields when configured and remain
  backward-compatible when clients only read `text`.
- `stable_delta` events include `segmentId`, `recordingId`, and `sequence`.
- Queue coalescing and stale drops are visible as metrics but do not create
  false evidence.

## Implementation Order Later

1. Add only the stabilizer dataclasses/config and tests.
2. Implement normalization and projection helpers until normalization tests
   pass.
3. Implement observation ordering/reset behavior.
4. Implement evidence ledger for non-space characters.
5. Implement stable delta/frontier output.
6. Implement space policy.
7. Implement punctuation policy.
8. Implement outlier classification.
9. Implement finalization event behavior.
10. Add recorder integration with minimal changes.
11. Add structured callback/server protocol integration.

## Open Questions

- Should the default evidence timing use the recommended 0.20 second character
  span, or should it exactly mirror the user example of three validations within
  300 ms?
- Should `on_realtime_transcription_stabilized(text)` remain a display-line
  compatibility callback, or become a true stable-prefix callback behind a new
  option?
- What should the new structured callback be named?
- Should punctuation ever be committed during realtime, or only after final
  transcription?
- Should final text be allowed to emit a final stable delta when it agrees with
  the stable prefix, or should final text stay entirely separate?
- Is the first shipped language policy explicitly space-tokenized, or must it
  handle non-space languages in the first implementation?
- Should recorder core expose `recording_id` publicly, or keep it internal and
  let FastAPI expose `segmentId`?
- How much debug history should be retained by default?
- Should optional word/segment timestamps be added to `TranscriptionResult`
  before or after the first text-only stabilizer implementation?
