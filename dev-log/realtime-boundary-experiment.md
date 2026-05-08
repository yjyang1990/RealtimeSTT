# Realtime Boundary Scheduling Experiment

## Context

RealtimeSTT's realtime transcription loop originally worked mostly as a timer:

1. Wait for `realtime_processing_pause`.
2. Snapshot the current recording frames.
3. Send the whole current audio buffer to the realtime transcription model.
4. Repeat while recording is active.

This is simple and reliable, but it can create a lot of GPU work. With a low
pause such as `0.02`, the realtime worker can ask Whisper/faster-whisper for a
new partial transcription many times per second, even when the new audio does
not contain much useful new speech information.

The experiment was based on a different idea: do a very cheap CPU-side
pre-check first, and only start the expensive realtime transcription when the
audio signal looks like it has reached a useful speech boundary.

## Core Idea

Speech often has better update points than a fixed timer can provide. A useful
point is often shortly after a vowel or syllable nucleus ends, because by then
the recognizer has a better chance to extract a meaningful new piece of text.

So the experiment introduced a small acoustic scheduler:

- watch the incoming PCM audio in small frames
- detect likely vowel/syllable boundary candidates
- trigger realtime transcription on the positive flank of that boundary signal
- add optional follow-up checks shortly after the boundary to catch trailing
  consonants
- keep a fallback timer so realtime transcription still happens if the boundary
  detector misses something

The important design goal was that this pre-check must be tiny compared with a
Whisper transcription pass. It should add near-zero practical load and no GPU
load.

## Standalone Detector

The detector was implemented separately from `audio_recorder.py` so it can be
tested and tuned independently:

- `RealtimeSTT/realtime_boundary_detector.py`

It exposes:

- `RealtimeSpeechBoundaryDetector`
- `SpeechBoundaryResult`
- `SpeechBoundaryEvent`

The detector is streaming and stateful. It accepts int16 PCM bytes or samples,
keeps a small rolling history, and emits boundary events when a candidate is
confirmed.

It does not run any neural model. It uses cheap CPU signal features:

- RMS/log-energy
- zero-crossing rate
- a lightweight autocorrelation-style voicing score
- local energy valleys
- recent vowel-like voiced frame history

The first version was too eager because it mostly detected energy dips. In live
microphone testing, it fired too often, including during silence/noise. The
detector was then tightened to behave more like a "vowel nucleus ended" signal:

- require recent vowel-like voiced material
- require sufficient energy above the noise floor
- require a voicing score consistent with voiced speech
- use zero-crossing rate to avoid treating noisy/unvoiced material as vowels
- only emit after the vowel-like region drops or changes

## Manual Live Visualizer

To test the detector independently, a terminal microphone visualizer was added:

- `tests/realtime_boundary_detector_microphone.py`
- `tests/realtime_boundary_detector_live_test.py`

The visualizer listens to the microphone and shows states in realtime:

- `waiting`
- `speaking`
- `vowel`
- `SYLLABLE END`

It also prints boundary event details when enabled:

- event count
- boundary time
- score
- energy drop
- valley depth
- latency
- reason

This made it possible to speak into the microphone and visually judge whether
the detector was firing at plausible syllable/vowel ending points.

## Recorder Integration

The new scheduler was integrated as an opt-in mode so the existing API and
default behavior remain intact.

New recorder parameters:

```python
realtime_transcription_use_syllable_boundaries=False
realtime_boundary_detector_sensitivity=0.6
realtime_boundary_followup_delays=(0.05, 0.2)
```

When `realtime_transcription_use_syllable_boundaries=False`, the old timer mode
continues to use `realtime_processing_pause` exactly as before.

When `realtime_transcription_use_syllable_boundaries=True`, the realtime worker
uses the new hybrid scheduler:

1. If the boundary detector emits a positive flank, transcribe immediately.
2. Schedule follow-up realtime transcriptions at
   `realtime_boundary_followup_delays`.
3. If a new boundary arrives, reset the pending follow-up delays from the new
   boundary time.
4. If no boundary is detected for `realtime_processing_pause` seconds, force a
   fallback realtime transcription.
5. If `realtime_processing_pause <= 0`, disable the forced fallback interval in
   syllable-boundary mode.

This means `realtime_processing_pause` has different meaning depending on the
mode:

- old timer mode: fixed interval between realtime transcription attempts
- syllable-boundary mode: fallback interval if no useful boundary is detected

Follow-up delays were changed from fixed booleans to a configurable sequence:

```python
realtime_boundary_followup_delays=(0.05, 0.2)
```

Examples:

```python
# One follow-up after 500 ms
realtime_boundary_followup_delays=(0.5,)

# No follow-ups
realtime_boundary_followup_delays=()

# Custom refinement passes
realtime_boundary_followup_delays=(0.03, 0.08, 0.25)
```

The recorder also now tracks realtime transcription counters:

- `realtime_transcription_count`
- `realtime_transcription_success_count`
- `realtime_transcription_empty_count`
- `realtime_transcription_trigger_counts`

Trigger counts can distinguish:

- `timer`
- `syllable-boundary`
- `syllable-boundary-followup`
- `syllable-boundary-fallback`

## Why Follow-Ups Exist

The boundary detector often fires when a vowel nucleus ends. That is a good
early point, but the following consonants may not have fully arrived yet.

Examples:

- `cat`: the vowel ends before the final `t`
- `best`: the vowel ends before `s` and `t`
- `asked`: the vowel ends before a consonant cluster

So the immediate boundary transcription gives low latency, while follow-up
passes can catch short trailing material and refine the partial text.

## Benchmark Script

A deterministic benchmark script was added:

- `tests/realtime_transcription_count_comparison.py`

It feeds a WAV file into `AudioToTextRecorder(use_microphone=False)` instead of
using the live microphone. This makes both modes process the exact same audio.

Test file:

- `tests/unit/audio/asr-reference-short.wav`

Reference text:

```text
Hey guys! Welcome to the new demo of my real-time transcription library, designed to showcase its lightning-fast capabilities. As you'll see, speech is transcribed almost instantly into text
```

The file was checked as:

- mono
- 16-bit PCM
- 16 kHz
- about 12.07 seconds

The script:

1. Loads the WAV file.
2. Runs timer mode.
3. Runs syllable-boundary mode.
4. Feeds the same audio into both modes at realtime pace.
5. Resets counters after recorder initialization, so model warmup is not
   counted.
6. Captures realtime attempt/success/update counts.
7. Runs final transcription validation.
8. Computes a normalized word error rate.

Warmup is already handled by `AudioToTextRecorder` during initialization. The
main transcription worker and realtime transcription model both use
`warmup_audio.wav` during setup. The benchmark resets counters after this point.

## Benchmark Results

Command:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\realtime_transcription_count_comparison.py --timer-pause 0.02 --syllable-pause 1 --followup-delays 0.5 --sensitivity 0.6
```

Timer mode:

```text
realtime attempts: 239
attempts/sec:      17.76
trigger counts:    {'timer': 239}
final WER:         0.000
final OK:          True
```

Syllable-boundary mode:

```text
realtime attempts: 44
attempts/sec:      3.32
trigger counts:    {'syllable-boundary': 41, 'syllable-boundary-followup': 3}
final WER:         0.000
final OK:          True
```

Result:

```text
Realtime transcription attempts reduced by 81.6%.
Final transcription validation passed in both modes.
Normalized final WER was 0.000 in both modes.
```

The realtime partial text was not perfect in either mode, but the final
transcription matched the expected text after normalization.

## Syllable Count Sanity Check

The reference sentence has roughly 48 syllables, depending on pronunciation.
For example, `library` can be pronounced as three or four syllables.

The syllable-boundary mode emitted 41 immediate boundary-triggered realtime
transcriptions for this sentence, which is in a plausible range for a heuristic
that is trying to catch useful vowel/syllable endings without firing on every
tiny acoustic wiggle.

## Current Takeaway

The experiment supports the idea that a tiny CPU-only acoustic pre-check can
substantially reduce expensive realtime transcription calls.

In the measured test:

- old fast timer mode at `realtime_processing_pause=0.02`: 239 realtime calls
- new syllable-boundary mode: 44 realtime calls
- reduction: 81.6%
- final transcription quality: unchanged in the test
- added detector load: near-zero practical load, CPU-only, no GPU work

In plain terms: a tiny signal-processing scheduler decides when it is worth
asking Whisper/faster-whisper to do the heavy work.

## Slow CPU Gap Regression

While testing the scheduler, another suspected bug showed up: with a slow
CPU-only model, final transcription can take long enough that new speech starts
arriving while the application loop is still blocked inside `recorder.text()`.
The fear is that the beginning of the next utterance can be dropped before the
recorder is armed again.

To reproduce this in a controlled way, a second script was added:

- `tests/final_transcription_gap_regression.py`

It streams the long reference file through the normal `AudioToTextRecorder.text()`
loop while a separate feeder thread keeps pushing audio into the recorder. That
is important because it is closer to the live microphone timing problem than a
single full-buffer transcription.

Test file:

- `tests/unit/audio/asr-reference.wav`

Expected JSON:

- `tests/unit/audio/asr-reference.expected_sentences.json`

The expected JSON is generated with a fast GPU run first, then the same audio is
run with CPU/int8 and compared against it.

Use this command for the full repro:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode both --model large-v2 --gpu-device cuda --gpu-compute-type default --cpu-device cpu --cpu-compute-type int8 --print-transcription-time
```

The script defaults to:

```text
pre_recording_buffer_duration=2.0
post_speech_silence_duration=0.6
min_length_of_recording=0.5
chunk_ms=32.0
speed=1.0
```

The 2.0s pre-recording buffer matters. With 1.0s, the GPU-generated expected
JSON already missed the first `Hey guys.` utterance, even though a direct GPU
full-buffer transcription of the WAV includes it. With 2.0s, the GPU golden
starts correctly:

```text
[00] Hey guys.
[01] Welcome to the new demo of my real-time transcription library designed to showcase its lightning-fast capabilities.
```

Latest run:

```text
GPU expected utterances: 9
CPU actual utterances:   4
combined WER:            0.682
allowed WER:             0.080
PASS:                    False
```

The CPU/int8 pass kept the first `Hey guys.` utterance, but then skipped large
parts of the long file. That makes the regression useful: the expected file is
now sane, and the CPU run fails in a way that matches the suspected dropped-audio
gap.

If the expected JSON should be regenerated separately:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode generate --model large-v2 --gpu-device cuda --gpu-compute-type default --print-transcription-time
```

If the CPU comparison should be run against an existing expected JSON:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode compare --model large-v2 --cpu-device cpu --cpu-compute-type int8 --print-transcription-time
```

### Inspecting The Actual Transcribed Snippets

The regression script now saves the exact audio array used for every final
transcription as a WAV snippet. This is meant for debugging boundary/pre-roll
problems where previous audio may leak into the next utterance.

By default, snippets are written next to the expected JSON:

```text
tests/unit/audio/asr-reference.expected_sentences.snippets/
```

Each pass gets its own folder:

```text
gpu_expected_pass/
cpu_comparison_pass/
```

Each folder contains:

- one WAV file per utterance, for example `06_gpu_expected_pass.wav`
- `manifest.json` with utterance text, normalized text, duration, and WAV path

The script clears the pass-specific snippet folder before each run. That avoids
stale files from an older run making it look like a later run produced snippets
that it did not actually produce.

The snippet dump can also be redirected:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode generate --model large-v2 --gpu-device cuda --gpu-compute-type default --expected-json C:\tmp\asr-reference.snippet-test.json --snippet-dir C:\tmp\asr-reference.snippets --print-transcription-time
```

To disable snippet writing:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode both --no-save-snippets
```

The snippet dump confirmed a real buffer/pre-roll issue. From utterance 2
onward, the saved snippets had audible old material at the start. The suspicious
utterance was:

```text
[06] And hand treatment. Whether you are working on a small project or something bigger, I hope this tool becomes a helpful resource and sparks some innovation.
```

The likely cause was that `audio_buffer` was being filled while the recorder was
already in the post-speech-silence phase. That made end-of-previous-utterance
audio available as pre-roll for the next utterance.

The fix was to clear `audio_buffer` when a recording transitions to stopped:

```python
if not self.is_recording and was_recording:
    self.stop_recording_on_voice_deactivity = False
    self.audio_buffer.clear()
```

After the fix, the GPU expected pass generated all 9 utterances and index 6 was
clean:

```text
[06] Whether you are working on a small project or something bigger, I hope this tool becomes a helpful resource and sparks some innovation.
```

The corrected project snippets are in:

```text
tests/unit/audio/asr-reference.expected_sentences.snippets/gpu_expected_pass/
```

The CPU/int8 comparison still fails with 4 actual utterances against 9 expected
utterances. That is now a separate slow-final-transcription gap issue, not the
same stale pre-roll overlap bug.

## Caveats And Next Steps

This is still a heuristic. It is not a true linguistic syllable parser.

Things to watch:

- noisy rooms
- speakers with very different pitch/voice characteristics
- languages with different syllable timing
- very fast speech
- long consonant clusters
- detector sensitivity causing too many or too few triggers

Useful next measurements:

- run the same benchmark on more files and languages
- compare realtime partial text quality, not only final text
- measure actual GPU utilization, not only call counts
- tune default sensitivity and follow-up delays
- consider a minimum transcription interval if a detector setting fires too
  frequently
