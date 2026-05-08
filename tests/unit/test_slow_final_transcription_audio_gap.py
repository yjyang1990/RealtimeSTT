import collections
import queue
import threading
import time
import unittest
import wave
from pathlib import Path

import numpy as np

try:
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
except Exception as exc:  # pragma: no cover - optional runtime deps may be absent
    AudioToTextRecorder = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


AUDIO_DIR = Path(__file__).with_name("audio")
REFERENCE_AUDIO = AUDIO_DIR / "asr-reference.wav"


def read_wav_samples(path):
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if channels != 1:
        raise ValueError(f"{path.name} must be mono for this test")
    if sample_width != 2:
        raise ValueError(f"{path.name} must be 16-bit PCM for this test")

    return np.frombuffer(frames, dtype=np.int16), sample_rate


def chunk_samples(samples, chunk_samples_count):
    usable = len(samples) - (len(samples) % chunk_samples_count)
    for start in range(0, usable, chunk_samples_count):
        yield samples[start:start + chunk_samples_count].tobytes()


class SlowFinalTranscriptionAudioGapReproTests(unittest.TestCase):
    def setUp(self):
        if IMPORT_ERROR is not None:
            self.skipTest(f"RealtimeSTT import failed: {IMPORT_ERROR}")

    def test_unarmed_recorder_keeps_only_last_pre_recording_window(self):
        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        pre_recording_buffer_duration = 1.0

        audio_buffer = collections.deque(
            maxlen=int((sample_rate // buffer_size) * pre_recording_buffer_duration)
        )

        simulated_block_seconds = 2.5
        simulated_samples = samples[:int(sample_rate * simulated_block_seconds)]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        for chunk in chunks:
            audio_buffer.append(chunk)

        retained_audio = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
        retained_start_sample = len(simulated_samples[:len(chunks) * buffer_size]) - len(retained_audio)

        self.assertGreater(retained_start_sample, 0)
        self.assertLessEqual(len(audio_buffer), audio_buffer.maxlen)
        self.assertAlmostEqual(
            len(retained_audio) / sample_rate,
            pre_recording_buffer_duration,
            delta=0.05,
        )

    def test_stopped_recording_is_queued_beyond_pre_recording_window(self):
        """Completed recordings are retained while final transcription blocks.

        This guards the slow-CPU path where the application is blocked in
        final transcription while the recorder worker completes another
        utterance in the background.
        """

        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        simulated_block_seconds = 2.5
        simulated_samples = samples[:int(sample_rate * simulated_block_seconds)]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.frames = chunks.copy()
        recorder.last_frames = []
        recorder.recorded_audio_queue = queue.Queue()
        recorder.recording_start_time = time.time() - 10
        recorder.min_length_of_recording = 0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.is_recording = True
        recorder.is_silero_speech_active = True
        recorder.is_webrtc_speech_active = True
        recorder.silero_check_time = 0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.on_recording_stop = None

        recorder.stop()

        queued_recording = recorder._get_next_recorded_audio()
        retained_audio = np.frombuffer(
            b"".join(queued_recording["frames"]),
            dtype=np.int16,
        )

        self.assertFalse(recorder.frames)
        np.testing.assert_array_equal(
            retained_audio,
            simulated_samples[:len(retained_audio)],
        )

    def test_wait_audio_consumes_queued_recording(self):
        samples, sample_rate = read_wav_samples(REFERENCE_AUDIO)
        buffer_size = 512
        simulated_samples = samples[:sample_rate]
        chunks = list(chunk_samples(simulated_samples, buffer_size))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.sample_rate = sample_rate
        recorder.frames = chunks.copy()
        recorder.last_frames = []
        recorder.audio = None
        recorder.recorded_audio_queue = queue.Queue()
        recorder.recording_start_time = time.time() - 10
        recorder.min_length_of_recording = 0
        recorder.backdate_stop_seconds = 0.0
        recorder.backdate_resume_seconds = 0.0
        recorder.is_recording = True
        recorder.is_silero_speech_active = True
        recorder.is_webrtc_speech_active = True
        recorder.silero_check_time = 0
        recorder.start_recording_event = threading.Event()
        recorder.stop_recording_event = threading.Event()
        recorder.interrupt_stop_event = threading.Event()
        recorder.listen_start = 0
        recorder.use_wake_words = False
        recorder.is_shut_down = False
        recorder.start_recording_on_voice_activity = False
        recorder.stop_recording_on_voice_deactivity = False
        recorder.continuous_listening = False
        recorder.on_recording_stop = None
        recorder._set_state = lambda state: None

        recorder.stop()
        recorder.wait_audio()

        expected = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32) / 32768.0

        self.assertFalse(recorder.has_pending_recordings())
        np.testing.assert_allclose(recorder.audio, expected)

    def test_voice_activity_allows_delayed_silero_confirmation(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_webrtc_speech_active = False
        recorder.is_silero_speech_active = True
        recorder.last_webrtc_speech_time = time.time() - 0.2

        self.assertTrue(recorder._is_voice_active())

        recorder.last_webrtc_speech_time = time.time() - 2.0

        self.assertFalse(recorder._is_voice_active())

    def test_flush_buffered_audio_queues_non_silent_tail(self):
        samples, _ = read_wav_samples(REFERENCE_AUDIO)
        chunks = list(chunk_samples(samples[:4096], 512))

        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_recording = False
        recorder.audio_buffer = collections.deque(chunks)
        recorder.recorded_audio_queue = queue.Queue()

        self.assertTrue(recorder.flush_buffered_audio())
        self.assertTrue(recorder.audio_buffer == collections.deque())

        queued_recording = recorder._get_next_recorded_audio()
        retained_audio = np.frombuffer(
            b"".join(queued_recording["frames"]),
            dtype=np.int16,
        )

        np.testing.assert_array_equal(
            retained_audio,
            samples[:len(retained_audio)],
        )

    def test_flush_buffered_audio_ignores_silence(self):
        recorder = AudioToTextRecorder.__new__(AudioToTextRecorder)
        recorder.is_recording = False
        recorder.audio_buffer = collections.deque([
            np.zeros(512, dtype=np.int16).tobytes()
        ])
        recorder.recorded_audio_queue = queue.Queue()

        self.assertFalse(recorder.flush_buffered_audio())
        self.assertFalse(recorder.has_pending_recordings())


if __name__ == "__main__":
    unittest.main()
