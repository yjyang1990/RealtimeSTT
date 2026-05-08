# Realtime Text Stabilization Requirements

This document describes the requirements for solving the realtime text
stabilization problem in RealtimeSTT.

It does not propose an implementation. It only captures the problem,
constraints, required behavior, data requirements, edge cases, and decisions
that must be made before implementation.

## Current Problem

Realtime text currently displays the latest text returned by the transcription
engine for the whole current audio frame.

That means:

- The displayed realtime text can change whenever the next realtime
  transcription arrives.
- The current output is useful as live feedback, but it is not stable enough to
  safely send onward as final or validated text.
- A downstream consumer such as an LLM needs text that will not later be
  corrected or withdrawn once communicated as validated.
- The current behavior does not preserve enough structured history of how the
  realtime text evolved during one sentence.

## Audio Frame Constraint

The existing VAD-driven audio frame sizing should not be reduced as part of this
work.

Reason:

- The current audio frame tries to capture a whole sentence.
- ASR quality depends on context.
- Cutting the audio frame too aggressively risks losing context.
- Losing context can make transcription worse.
- The stabilization work should happen on top of the realtime transcription
  stream, not by making the underlying audio frame shorter.

## High-Level Goal

For each sentence or current audio frame, RealtimeSTT should keep track of every
incoming realtime transcription and use that history to identify which parts of
the text have become stable.

The stabilized output must be suitable for use as text that cannot later be
changed.

The live display must still be able to show both:

- Stable text that is considered validated.
- Unstable text that is still only live, provisional, and changeable.

## Data That Must Be Captured

For every incoming realtime transcription belonging to the current sentence or
audio frame, store the best available data for later stabilization.

Required data:

- The realtime text returned by the engine.
- A timestamp for when that realtime text arrived.
- Timing information for where the audio frames came from initially, before the
  realtime transcription was first started.
- Enough timing metadata to relate each realtime text update back to the audio
  region that produced it.
- The order in which realtime texts arrived.
- The association between all realtime texts and the sentence or audio frame
  they belong to.

The collected data should be the best possible data available, because the
stabilization algorithm depends on high-quality history.

## Realtime Text Stream Characteristics

The algorithm must assume that realtime transcriptions for a sentence usually
grow in text size over time.

The algorithm must also assume that:

- Earlier text may be rewritten by later realtime transcriptions.
- Words may appear incomplete before they are finished.
- Punctuation may appear, disappear, or differ between updates.
- Uppercase and lowercase differences may occur.
- A comma or similar punctuation difference may cause direct text comparison to
  fail even when the text is effectively the same.
- A single incoming realtime transcription may be a clear hallucination or slip.
- Some parts near the left side of the text may become stable while parts near
  the right side remain unstable.

## Stable Text Requirement

The stabilization layer must identify text fragments that are stable enough to
be emitted as validated text.

Once text is emitted as validated:

- It must not be changed later.
- It must not be withdrawn later.
- It must remain usable by downstream consumers as non-revisable text.

This non-revisable property is the central requirement.

## Stability Criteria Requirement

Concrete stability criteria must be defined before implementation.

The criteria must decide when a text fragment has been validated often enough
and long enough to be considered stable.

The criteria must balance:

- Emitting stable text as fast as possible.
- Waiting long enough to avoid emitting incorrect text.
- Avoiding overreaction to single slips or hallucinations.
- Supporting slowly arriving words that appear syllable by syllable.
- Avoiding premature spaces that imply a word has finished when it may not have.

The user-provided example criterion is:

- A syllable or fragment validated 3 times within 300 ms may be considered
  stable enough.

That example is not yet the final rule. The final criteria still need to be
defined.

## Word Fragment Requirement

The algorithm must handle partial words carefully.

Important cases:

- A user may pause in the middle of a longer word.
- The transcription engine may output only the beginning of a word.
- The beginning of a word may be validated repeatedly before the word is
  complete.

Requirements:

- The algorithm must not incorrectly mark a complete word, including a following
  blank space, when the user has only spoken a word fragment.
- The algorithm may emit incomplete word content, such as the first syllables,
  if that fragment has been validated by enough realtime transcriptions over a
  sufficient period of time.
- If only a word fragment is emitted, the algorithm must not emit a trailing
  blank space that would make the fragment look like a completed word.
- If the user later continues the word by adding more syllables, the algorithm
  must be able to add the continuation without needing to change previously
  emitted stable text.

## Space Handling Requirement

Spaces are semantically important.

Requirements:

- Emitting a blank space after a text fragment means the preceding word is being
  treated as finished.
- The algorithm must be very careful before emitting a blank space as stable
  text.
- A blank space should not be emitted as stable text if there is still a
  realistic possibility that the current word is incomplete.
- The algorithm must avoid producing stable output that forces an incorrect word
  boundary.

## Hallucination and Slip Requirement

The algorithm must detect and ignore realtime updates that are clearly not
consistent with the remembered realtime text history.

Requirements:

- A single realtime transcription that is completely different from the previous
  remembered realtime texts must not cause stable output to change.
- Such a transcription must not be considered strong evidence for stabilizing
  new text.
- The purpose of the algorithm is to stabilize the stream, so it must not react
  to single slips and sudden changes as if they were validated text.
- Hallucination-like updates must be handled without corrupting the stable text
  history.

## Stable and Unstable Text Display Requirement

For a full realtime text, the display may contain two conceptual parts:

- A left-side stable part that has already been validated.
- A right-side unstable part that is still live and provisional.

Requirements:

- Stable text must be shown as stable.
- Unstable text must still be visible to preserve the live transcription
  experience.
- The display must make it possible to merge the stable and unstable parts into
  one readable live line.
- The unstable right side may differ from the stable left side in small ways,
  such as punctuation or casing.
- The merge must account for cases where direct text comparison fails because of
  small formatting differences.
- The merge must avoid duplicating text.
- The merge must avoid hiding unstable text that the user expects to see live.

## Text Comparison Requirement

The stabilization process must compare incoming realtime texts with previous
realtime texts and with already-stable output.

The comparison must account for:

- Exact matches.
- Text growth over time.
- Case differences.
- Missing or added punctuation.
- Missing or added commas.
- Word fragments.
- Stable left-side text plus unstable right-side text.
- Incoming text that partially overlaps with stable text but is not identical.
- Incoming text that is clearly unrelated to the remembered history.

The comparison requirements must be defined before implementation because direct
string comparison will miss some cases that should count as the same text.

## Latency Requirement

The stabilized output should appear as fast as possible while still being safe.

Requirements:

- The algorithm must not wait until final transcription to emit stable text.
- The algorithm must emit stable text incrementally when enough evidence exists.
- The algorithm must avoid adding unnecessary delay to clearly stable text.
- The algorithm must still validate often enough to avoid premature output.

## Downstream Consumer Requirement

One intended use of the stable text is to provide text to an LLM or another
consumer that needs non-revisable input.

Requirements:

- Text sent as stable must be safe to consume as committed text.
- The downstream consumer must not need to handle later corrections to already
  emitted stable text.
- Unstable text must be distinguishable from stable text so downstream consumers
  can decide whether to use it.

## Sentence and Frame Association Requirement

The stabilization history must belong to the current sentence or current audio
frame.

Requirements:

- Realtime text updates from different sentences or audio frames must not be
  mixed incorrectly.
- Stable state from a previous sentence must not corrupt the next sentence.
- The history must be reset, finalized, or otherwise separated at the correct
  sentence or recording boundary.
- The exact boundary behavior must be defined before implementation.

## Edge Cases to Consider

The requirements and later design must explicitly consider:

- Pause in the middle of a long word.
- Slowly spoken words that arrive syllable by syllable.
- A validated word fragment that later becomes a longer word.
- A word that appears stable but has no safe trailing space yet.
- Realtime output that briefly changes to unrelated text.
- Realtime output with missing punctuation.
- Realtime output with extra punctuation.
- Realtime output with different uppercase or lowercase formatting.
- Realtime output that repeats text.
- Realtime output that drops a previously seen fragment and later brings it
  back.
- A sentence where the left side is stable but the right side keeps changing.

## Non-Requirements for This Work

This work should not require:

- Reducing the VAD-driven audio frame size.
- Cutting off sentence context to make realtime updates shorter.
- Treating the latest realtime transcription as validated text by default.
- Allowing emitted stable text to be revised later.

## Decisions Required Before Implementation

Before implementation, define:

- Exact validation criteria for stable fragments.
- Exact validation criteria for spaces.
- Exact handling for incomplete words and syllables.
- Exact handling for punctuation and casing differences.
- Exact handling for hallucination-like realtime updates.
- Exact data structure for storing realtime text history.
- Exact timestamp fields and their meaning.
- Exact boundary behavior between sentences or audio frames.
- Exact UI representation for stable and unstable text.
- Exact downstream API representation for stable and unstable text.

## Acceptance Requirements

The stabilization feature is acceptable only if:

- Stable text can be emitted incrementally before final transcription.
- Already-emitted stable text is never changed.
- Word fragments can be handled without forcing premature word boundaries.
- Hallucination-like single updates do not destabilize the output.
- The UI can show stable and unstable text together.
- The system stores enough realtime text history and timing metadata to support
  the stabilization criteria.
- The behavior can be tested against pauses, word fragments, punctuation
  differences, casing differences, and hallucination-like slips.
