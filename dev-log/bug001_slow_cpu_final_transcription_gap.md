# Bug 001: Slow CPU Final Transcription Dropped Incoming Speech

## Summary

RealtimeSTT could lose utterances when final transcription was much slower than
the incoming audio stream.

The failure showed up with `tests/final_transcription_gap_regression.py` when
streaming `tests/unit/audio/asr-reference.wav` through
`AudioToTextRecorder(use_microphone=False)`:

- GPU/faster final transcription produced the expected 9 utterances.
- CPU/int8 final transcription with `large-v2` took about 10 to 13 seconds per
  utterance.
- CPU/int8 returned only 4 utterances before the fix.
- The first utterance was correct, but later results jumped forward through the
  audio, proving that speech arriving during final transcription was being
  skipped.

## Reproduction

Generate the expected transcript with GPU:

```powershell
python -B tests/final_transcription_gap_regression.py --mode generate --model large-v2 --gpu-device cuda --gpu-compute-type default --print-transcription-time
```

Compare CPU/int8:

```powershell
python -B tests/final_transcription_gap_regression.py --mode compare --model large-v2 --cpu-device cpu --cpu-compute-type int8 --print-transcription-time
```

Before the fix, the CPU comparison typically reported:

```text
expected utterances: 9
actual utterances:   4
combined WER:        0.676
PASS: False
```

## Root Cause

The application loop calls `recorder.text()`. Internally this does two phases:

1. `wait_audio()` waits for VAD to start and stop one recording.
2. `transcribe()` / `perform_final_transcription()` blocks until the final model
   returns text.

While the CPU final model was busy, the feeder thread kept pushing new audio
into `feed_audio()`. The recording worker still consumed those chunks, but the
recorder was no longer armed for the next recording because the app loop had not
returned from `text()` yet.

In that unarmed state, incoming audio was only retained in `self.audio_buffer`,
the finite pre-recording ring buffer. With the regression settings that buffer
covered about 2 seconds, while each CPU final transcription took over 10
seconds. As a result:

- the start of the next utterance was overwritten before `text()` re-armed VAD
- sometimes an entire utterance was overwritten
- CPU transcription resumed at a later phrase and appeared to skip chunks of the
  WAV

The bug was not a Whisper transcription accuracy problem. It was a recorder state
and buffering problem while the final transcription pipe was busy.

## Fix

The fix makes completed recordings durable across slow final transcription and
keeps the recorder armed during continuous `text()` loops.

### 1. Queue completed recordings

`AudioToTextRecorder` now owns `self.recorded_audio_queue`.

When `stop()` completes a recording, it deep-copies the stopped frames and queues
them with their backdate metadata:

- frames
- `backdate_stop_seconds`
- `backdate_resume_seconds`

`stop()` then clears `self.frames`, so live capture can continue without the
completed utterance being overwritten or reused accidentally.

### 2. Let `wait_audio()` consume queued recordings

`wait_audio()` now checks `recorded_audio_queue` before waiting for new VAD
activity. If a recording was completed while final transcription was blocking,
the next call to `text()` consumes that queued audio immediately.

The existing frame-to-float conversion and backdate logic was moved into
`_set_audio_from_frames()` so the same code path is used for:

- current live frames
- last fallback frames
- queued completed recordings

### 3. Keep VAD capture armed while final transcription blocks

For the normal no-wake-word `text()` loop, after `wait_audio()` has captured one
utterance it now enters a `continuous_listening` mode. In that mode the recorder
keeps:

- `start_recording_on_voice_activity = True`
- `stop_recording_on_voice_deactivity = True`

When the recording worker stops one utterance and notices continuous listening
is active, it re-arms those flags instead of fully idling. This allows the worker
thread to capture and queue additional utterances while the main thread is still
blocked in CPU final transcription.

### 4. Tolerate delayed Silero confirmation

Start detection requires both WebRTC and Silero VAD. Silero is checked
asynchronously, and under CPU pressure its positive result can arrive after
WebRTC has already moved past the speech chunk.

The fix records `last_webrtc_speech_time` whenever WebRTC detects speech.
`_is_voice_active()` accepts a recent WebRTC speech hit for a short hangover
window while still requiring Silero to confirm speech. This keeps the dual-VAD
gate, but prevents short phrases from being missed just because Silero was
delayed by CPU load.

### 5. Flush finite `feed_audio()` streams

The regression uses a finite WAV stream, not a live microphone. At end of stream,
a final short phrase can still be sitting in the pre-recording buffer and may not
have formed a full VAD-started recording before the feeder stops.

`flush_buffered_audio()` was added for this case. It queues non-silent buffered
audio at the end of a finite feed. The regression feeder calls this after the
tail silence. Pure silence is ignored.

### 6. Fix regression idle detection

The regression script now treats the recorder as idle only when:

- it is not currently recording
- `recorder.frames` is empty
- `recorded_audio_queue` has no pending recordings

Without that check, the test harness could stop before draining queued
utterances.

## Verification

After the fix, the CPU/int8 regression passes:

```text
expected utterances: 9
actual utterances:   9
combined WER:        0.000
PASS: True
```

Focused unit coverage was added in
`tests/unit/test_slow_final_transcription_audio_gap.py` for:

- preserving a completed recording beyond the pre-recording ring-buffer window
- consuming a queued recording in `wait_audio()`
- accepting delayed Silero confirmation after a recent WebRTC speech hit
- flushing non-silent buffered audio at the end of a finite stream
- ignoring pure silence during buffer flush

Additional checks run:

```powershell
python -B -m unittest -v tests.unit.test_slow_final_transcription_audio_gap
python -B -m unittest -v tests.unit.test_audio_fixtures
python -B -u tests/final_transcription_gap_regression.py --mode compare --model large-v2 --cpu-device cpu --cpu-compute-type int8 --print-transcription-time
```

The CPU regression completed with all 9 expected utterances and zero WER.
