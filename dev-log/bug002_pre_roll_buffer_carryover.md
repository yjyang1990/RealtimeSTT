# Bug 002: Pre-Roll Buffer Carry-Over Between Utterances

## Summary

We found and fixed a bug where final transcription snippets after the first
utterance could start with stale audio from the previous utterance.

The visible symptom was a wrong GPU-generated expected sentence:

```text
[06] And hand treatment. Whether you are working on a small project or something bigger, I hope this tool becomes a helpful resource and sparks some innovation.
```

The phrase `And hand treatment` was not in the source audio. It was likely
Whisper trying to decode stale audio from the previous sentence ending:

```text
... helps improve its future development.
```

After listening to the saved snippets, the issue became clear: from about
snippet `02` onward, snippets had garbage/old audio at the beginning.

## How To Reproduce

Generate GPU expected utterances and snippets:

```powershell
D:\Projekte\STT\RealtimeSTT\test_env\Scripts\python.exe -B tests\final_transcription_gap_regression.py --mode generate --model large-v2 --gpu-device cuda --gpu-compute-type default --print-transcription-time
```

The script writes snippets here by default:

```text
tests/unit/audio/asr-reference.expected_sentences.snippets/gpu_expected_pass/
```

Before the fix, listening to snippets such as:

```text
02_gpu_expected_pass.wav
03_gpu_expected_pass.wav
...
06_gpu_expected_pass.wav
```

showed that previous audio material was being prepended to later utterances.

## Root Cause

`audio_recorder.py` uses `self.audio_buffer` as a pre-recording buffer. This is
normally useful: when VAD detects speech, the recorder prepends a little buffered
audio so the beginning of the utterance is not clipped.

The problem was that `audio_buffer` was also being filled during the
post-speech-silence phase:

```python
if not self.is_recording or self.speech_end_silence_start:
    self.audio_buffer.append(data)
```

Then, when the next recording started, the recorder prepended the whole buffer:

```python
self.frames.extend(list(self.audio_buffer))
self.audio_buffer.clear()
```

That meant audio from the end of the previous recording could become the
pre-roll of the next recording.

This became especially obvious with:

```text
pre_recording_buffer_duration=2.0
```

The longer pre-roll helped preserve the first `Hey guys.` utterance, but also
made stale inter-utterance carry-over easier to hear.

## Fix

The fix was to clear `audio_buffer` exactly when a recording transitions from
recording to stopped:

```python
if not self.is_recording and was_recording:
    self.stop_recording_on_voice_deactivity = False
    self.audio_buffer.clear()
```

This removes old tail material from the previous utterance.

Important: this does not disable pre-roll for the next utterance. After the
recording has stopped, new incoming audio while not recording still fills
`audio_buffer` normally. So immediate next speech can still be buffered, but the
buffer starts fresh after the previous final recording ended.

## Debugging Support Added

To make this visible, `tests/final_transcription_gap_regression.py` now saves
the exact audio used for every final transcription:

```text
tests/unit/audio/asr-reference.expected_sentences.snippets/
```

Each pass gets a subfolder:

```text
gpu_expected_pass/
cpu_comparison_pass/
```

Each folder contains:

- one WAV per utterance
- `manifest.json` with text, normalized text, WAV path, duration, and sample count

The snippet writer was also hardened:

- snippet write failures are non-fatal
- each pass-specific snippet folder is cleared before a new run
- this avoids stale WAV files from older runs being mistaken for current output

## Verification

After the fix, the GPU expected pass produced all 9 utterances again.

The corrected index 6 is:

```text
[06] Whether you are working on a small project or something bigger, I hope this tool becomes a helpful resource and sparks some innovation.
```

The final expected JSON was regenerated:

```text
tests/unit/audio/asr-reference.expected_sentences.json
```

The corrected GPU snippets were regenerated:

```text
tests/unit/audio/asr-reference.expected_sentences.snippets/gpu_expected_pass/
```

Focused checks also passed:

```text
syntax ok
5 unit tests OK, expected failures=1
```

## What This Did Not Fix

This did not fix the separate slow CPU final transcription gap.

After this bug was fixed, CPU/int8 comparison still produced only 4 utterances
against the GPU expected 9 utterances:

```text
CPU actual utterances: 4
GPU expected utterances: 9
combined WER: 0.682
PASS: False
```

So Bug 002 fixed stale pre-roll carry-over. The CPU drop/loss issue remains a
separate bug to investigate.
