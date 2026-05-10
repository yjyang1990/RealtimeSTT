import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from RealtimeSTT.transcription_engines import (
    TranscriptionEngineConfig,
    TranscriptionEngineError,
    create_transcription_engine,
    get_supported_transcription_engines,
)
from RealtimeSTT.transcription_engines.kroko_onnx_engine import (
    DEFAULT_KROKO_ONNX_MODEL,
    KrokoOnnxBackend,
    KrokoOnnxDecodedOutput,
    KrokoOnnxEngine,
)
from tests.unit import test_additional_transcription_engines as audio_fixtures
from tests.unit.test_additional_transcription_engines import AudioVector


class FakeKrokoResult:
    def __init__(self, text=" kroko text "):
        self.text = text


class FakeKrokoStream:
    def __init__(self, ready_count=2):
        self.accepted = []
        self.finished = False
        self.ready_count = ready_count
        self.decode_count = 0

    def accept_waveform(self, sample_rate, waveform):
        self.accepted.append((sample_rate, waveform))

    def input_finished(self):
        self.finished = True


class FakeKrokoRecognizer:
    transducer_calls = []
    instances = []

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.streams = []
        self.decode_streams_calls = []
        FakeKrokoRecognizer.instances.append(self)

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.transducer_calls.append(kwargs)
        return cls(kwargs)

    def create_stream(self):
        stream = FakeKrokoStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream):
        if stream.ready_count <= 0:
            return False
        stream.ready_count -= 1
        return True

    def decode_streams(self, streams):
        self.decode_streams_calls.append(list(streams))
        for stream in streams:
            stream.decode_count += 1

    def get_result(self, stream):
        return FakeKrokoResult()


class FakeKrokoBackend:
    def __init__(self, config=None):
        self.config = config
        self.calls = []

    def transcribe(self, audio, **params):
        self.calls.append((audio, params))
        return KrokoOnnxDecodedOutput(" mocked kroko text ", params.get("language", ""))


class KrokoOnnxEngineTests(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("NumPy is required for Kroko-ONNX adapter tests")

    def tearDown(self):
        FakeKrokoRecognizer.transducer_calls.clear()
        FakeKrokoRecognizer.instances.clear()

    def make_model_file(self, filename=DEFAULT_KROKO_ONNX_MODEL):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / filename
        path.write_bytes(b"placeholder")
        return temp_dir, path

    def test_supported_engines_include_kroko_aliases(self):
        engines = get_supported_transcription_engines()

        for name in ("kroko_onnx", "kroko", "banafo_kroko"):
            self.assertIn(name, engines)

    def test_factory_creates_kroko_engine_with_mocked_backend(self):
        with patch(
            "RealtimeSTT.transcription_engines.kroko_onnx_engine.KrokoOnnxBackend",
            FakeKrokoBackend,
        ):
            engine = create_transcription_engine(
                "kroko-onnx",
                TranscriptionEngineConfig(model="model.data"),
            )

        self.assertIsInstance(engine, KrokoOnnxEngine)
        self.assertIsInstance(engine.backend, FakeKrokoBackend)

    def test_missing_dependency_reports_install_hint(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)

        with patch(
            "RealtimeSTT.transcription_engines.kroko_onnx_engine.import_module",
            side_effect=ModuleNotFoundError("kroko_onnx"),
        ):
            with self.assertRaisesRegex(TranscriptionEngineError, "kroko-onnx"):
                KrokoOnnxBackend(
                    TranscriptionEngineConfig(model=str(model_path)),
                    numpy_module=np,
                )

    def test_resolves_absolute_model_path(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)

        KrokoOnnxBackend(
            TranscriptionEngineConfig(model=str(model_path)),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        self.assertEqual(
            FakeKrokoRecognizer.transducer_calls[0]["model_path"],
            str(model_path),
        )

    def test_resolves_relative_model_under_download_root(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)

        KrokoOnnxBackend(
            TranscriptionEngineConfig(
                model=model_path.name,
                download_root=temp_dir.name,
            ),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        self.assertEqual(
            FakeKrokoRecognizer.transducer_calls[0]["model_path"],
            str(model_path),
        )

    def test_engine_option_model_path_overrides_config_model(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)

        KrokoOnnxBackend(
            TranscriptionEngineConfig(
                model="missing.data",
                engine_options={"model_path": str(model_path)},
            ),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        self.assertEqual(
            FakeKrokoRecognizer.transducer_calls[0]["model_path"],
            str(model_path),
        )

    def test_model_dir_selects_single_data_file(self):
        temp_dir, model_path = self.make_model_file("Kroko-DE-Community-64-L-Streaming-001.data")
        self.addCleanup(temp_dir.cleanup)

        backend = KrokoOnnxBackend(
            TranscriptionEngineConfig(
                model="ignored.data",
                engine_options={"model_dir": temp_dir.name},
            ),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        self.assertEqual(backend.model_path, model_path)

    def test_missing_model_path_reports_download_hint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.data"
            with self.assertRaisesRegex(TranscriptionEngineError, "Banafo/Kroko-ASR"):
                KrokoOnnxBackend(
                    TranscriptionEngineConfig(model=str(missing)),
                    recognizer_cls=FakeKrokoRecognizer,
                    numpy_module=np,
                )

    def test_provider_mapping_from_device_and_option(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)

        cases = [
            ("cpu", {}, "cpu"),
            ("cuda", {}, "cuda"),
            ("cuda:0", {"provider": "cpu"}, "cpu"),
        ]
        for device, options, expected in cases:
            with self.subTest(device=device, options=options):
                FakeKrokoRecognizer.transducer_calls.clear()
                KrokoOnnxBackend(
                    TranscriptionEngineConfig(
                        model=str(model_path),
                        device=device,
                        engine_options=options,
                    ),
                    recognizer_cls=FakeKrokoRecognizer,
                    numpy_module=np,
                )
                self.assertEqual(
                    FakeKrokoRecognizer.transducer_calls[0]["provider"],
                    expected,
                )

    def test_recognizer_kwargs_map_kroko_options(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)
        options = {
            "key": "license-key",
            "referralcode": "project",
            "num_threads": "3",
            "provider": "cpu",
            "sample_rate": "8000",
            "feature_dim": "40",
            "decoding_method": "modified_beam_search",
            "max_active_paths": "8",
            "hotwords_file": "hotwords.txt",
            "hotwords_score": "2.5",
            "blank_penalty": "0.25",
            "enable_endpoint_detection": "false",
            "rule1_min_trailing_silence": "1.1",
            "rule2_min_trailing_silence": "1.2",
            "rule3_min_utterance_length": "9.0",
            "recognizer": {"debug": True, "num_threads": 5},
        }

        KrokoOnnxBackend(
            TranscriptionEngineConfig(model=str(model_path), engine_options=options),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        call = FakeKrokoRecognizer.transducer_calls[0]
        self.assertEqual(call["key"], "license-key")
        self.assertEqual(call["referralcode"], "project")
        self.assertEqual(call["num_threads"], 5)
        self.assertEqual(call["sample_rate"], 8000)
        self.assertEqual(call["feature_dim"], 40)
        self.assertEqual(call["decoding_method"], "modified_beam_search")
        self.assertEqual(call["max_active_paths"], 8)
        self.assertEqual(call["hotwords_file"], "hotwords.txt")
        self.assertEqual(call["hotwords_score"], 2.5)
        self.assertEqual(call["blank_penalty"], 0.25)
        self.assertFalse(call["enable_endpoint_detection"])
        self.assertEqual(call["rule1_min_trailing_silence"], 1.1)
        self.assertEqual(call["rule2_min_trailing_silence"], 1.2)
        self.assertEqual(call["rule3_min_utterance_length"], 9.0)
        self.assertTrue(call["debug"])

    def test_transcribe_feeds_tail_padding_decodes_until_ready_and_returns_text(self):
        temp_dir, model_path = self.make_model_file()
        self.addCleanup(temp_dir.cleanup)
        backend = KrokoOnnxBackend(
            TranscriptionEngineConfig(
                model=str(model_path),
                engine_options={"tail_padding_seconds": 0.1},
            ),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        output = backend.transcribe([0.25, -0.5], language="en")

        stream = backend.recognizer.streams[0]
        self.assertEqual(output.text, "kroko text")
        self.assertEqual(output.language, "en")
        self.assertTrue(stream.finished)
        self.assertEqual(stream.accepted[0][0], 16000)
        self.assertEqual(stream.accepted[0][1].dtype, np.float32)
        self.assertEqual(stream.accepted[0][1].tolist(), [0.25, -0.5])
        self.assertEqual(stream.accepted[1][1].shape[0], 1600)
        self.assertEqual(len(backend.recognizer.decode_streams_calls), 2)
        self.assertEqual(stream.decode_count, 2)

    def test_engine_normalizes_audio_and_returns_language(self):
        backend = FakeKrokoBackend()
        engine = KrokoOnnxEngine(
            TranscriptionEngineConfig(
                model="model.data",
                normalize_audio=True,
                engine_options={"language": "de"},
            ),
            backend=backend,
        )

        result = engine.transcribe(AudioVector([0.0, -2.0, 1.0]), language="en")

        backend_audio, params = backend.calls[0]
        self.assertEqual(backend_audio.values, [0.0, -0.95, 0.475])
        self.assertEqual(params["language"], "de")
        self.assertEqual(result.text, "mocked kroko text")
        self.assertEqual(result.info.language, "de")
        self.assertEqual(result.info.language_probability, 1.0)

    def test_backend_infers_language_from_kroko_model_name(self):
        temp_dir, model_path = self.make_model_file("Kroko-EN-Community-64-L-Streaming-001.data")
        self.addCleanup(temp_dir.cleanup)
        backend = KrokoOnnxBackend(
            TranscriptionEngineConfig(model=str(model_path)),
            recognizer_cls=FakeKrokoRecognizer,
            numpy_module=np,
        )

        output = backend.transcribe([0.0], language=None)

        self.assertEqual(output.language, "en")


class KrokoOnnxGoldenTranscriptionTests(unittest.TestCase):
    def setUp(self):
        if audio_fixtures.np is None:
            self.skipTest("NumPy is required for Kroko-ONNX golden tests")

    def test_transcribes_fixture_with_real_kroko_backend(self):
        if os.environ.get("REALTIMESTT_RUN_KROKO_ONNX") != "1":
            self.skipTest(
                "Set REALTIMESTT_RUN_KROKO_ONNX=1 to run the Kroko-ONNX smoke test"
            )

        model_path = Path(
            os.environ.get(
                "REALTIMESTT_KROKO_ONNX_MODEL",
                "test-model-cache/kroko-onnx/%s" % DEFAULT_KROKO_ONNX_MODEL,
            )
        )
        if not model_path.is_file():
            self.skipTest("Kroko-ONNX model file not found: %s" % model_path)

        audio, expected = audio_fixtures.read_fixture_audio()
        engine_options = {
            "provider": os.environ.get("REALTIMESTT_KROKO_ONNX_PROVIDER", "cpu"),
            "num_threads": int(os.environ.get("REALTIMESTT_KROKO_ONNX_NUM_THREADS", "1")),
        }
        key = os.environ.get("REALTIMESTT_KROKO_ONNX_KEY")
        if key:
            engine_options["key"] = key

        engine = KrokoOnnxEngine(
            TranscriptionEngineConfig(
                model=str(model_path),
                device=(
                    "cuda"
                    if engine_options["provider"].lower().startswith("cuda")
                    else "cpu"
                ),
                engine_options=engine_options,
            )
        )

        start = time.time()
        result = engine.transcribe(audio, language="en")
        elapsed = time.time() - start
        duration = len(audio) / 16000.0
        actual = audio_fixtures.normalize_transcript(result.text)

        print("\n[RealtimeSTT test] kroko_onnx expected: %s" % expected)
        print("[RealtimeSTT test] kroko_onnx actual:   %s" % actual)
        print(
            "[RealtimeSTT test] kroko_onnx RTF: %.3f / %.3f = %.3f"
            % (elapsed, duration, elapsed / duration)
        )

        self.assertTrue(actual)
        self.assertIn(" ".join(expected.split()[:2]), actual)


if __name__ == "__main__":
    unittest.main()
