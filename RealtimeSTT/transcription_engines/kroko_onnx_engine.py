import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from ._model_utils import text_from_output
from .base import (
    BaseTranscriptionEngine,
    TranscriptionEngineError,
    TranscriptionInfo,
    TranscriptionResult,
)


DEFAULT_KROKO_ONNX_MODEL = "Kroko-EN-Community-64-L-Streaming-001.data"
KROKO_ONNX_HF_REPO = "Banafo/Kroko-ASR"
KROKO_ONNX_MODEL_URL = "https://huggingface.co/Banafo/Kroko-ASR"


@dataclass
class KrokoOnnxDecodedOutput:
    text: str
    language: str = ""


def _load_numpy():
    try:
        return import_module("numpy")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'kroko_onnx' transcription engine requires numpy audio arrays."
        ) from exc


def _load_online_recognizer_class():
    try:
        kroko_onnx = import_module("kroko_onnx")
    except ModuleNotFoundError as exc:
        raise TranscriptionEngineError(
            "The 'kroko_onnx' transcription engine requires the optional "
            "'kroko-onnx' package. Install it from "
            "https://github.com/kroko-ai/kroko-onnx with 'pip install .' in a "
            "built checkout. Upstream currently documents Linux builds; "
            "Windows and macOS build support may require WSL2, Docker, or a "
            "platform-specific build."
        ) from exc

    try:
        return kroko_onnx.OnlineRecognizer
    except AttributeError as exc:
        raise TranscriptionEngineError(
            "The installed 'kroko-onnx' package does not expose "
            "OnlineRecognizer. Install a current kroko-onnx checkout."
        ) from exc


def _bool_option(options, name, default=False):
    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _int_option(options, name, default):
    try:
        return int(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError("kroko-onnx option '%s' must be an integer." % name)


def _float_option(options, name, default):
    try:
        return float(options.get(name, default))
    except (TypeError, ValueError):
        raise TranscriptionEngineError("kroko-onnx option '%s' must be a number." % name)


def _provider_from_config(config, options):
    explicit = options.get("provider")
    if explicit:
        return str(explicit)
    device = str(config.device or "").lower()
    return "cuda" if device.startswith("cuda") else "cpu"


def _maybe_under_download_root(download_root, value):
    path = Path(str(value)).expanduser()
    if path.is_absolute() or not download_root:
        return path
    return Path(download_root).expanduser() / path


def _first_existing_data_file(model_dir):
    try:
        data_files = sorted(model_dir.glob("*.data"))
    except OSError:
        return None
    return data_files[0] if len(data_files) == 1 else None


def _language_from_model_path(path):
    match = re.search(r"(?:^|[-_/\\])Kroko-([A-Za-z]{2})-", str(path))
    return match.group(1).lower() if match else ""


class KrokoOnnxBackend:
    def __init__(self, config, recognizer_cls=None, numpy_module=None):
        self.config = config
        self.engine_options = dict(config.engine_options or {})
        self.np = numpy_module or _load_numpy()
        self.model_path = self._resolve_model_path()
        self.sample_rate = _int_option(self.engine_options, "sample_rate", 16000)
        self.tail_padding_seconds = _float_option(
            self.engine_options,
            "tail_padding_seconds",
            self.engine_options.get("finalization_padding_seconds", 0.66),
        )
        recognizer_cls = recognizer_cls or _load_online_recognizer_class()
        self.recognizer = self._create_recognizer(recognizer_cls)

    def _resolve_model_path(self):
        model_path_value = self.engine_options.get("model_path") or self.engine_options.get("model_file")
        model_dir_value = self.engine_options.get("model_dir")

        if model_path_value:
            path = _maybe_under_download_root(self.config.download_root, model_path_value)
        elif model_dir_value:
            model_dir = _maybe_under_download_root(self.config.download_root, model_dir_value)
            if model_dir.suffix == ".data":
                path = model_dir
            else:
                filename = self.engine_options.get("model_filename") or DEFAULT_KROKO_ONNX_MODEL
                path = model_dir / filename
                if not path.is_file() and model_dir.is_dir():
                    path = _first_existing_data_file(model_dir) or path
        else:
            path = _maybe_under_download_root(
                self.config.download_root,
                self.config.model or DEFAULT_KROKO_ONNX_MODEL,
            )

        if not path.is_file():
            raise TranscriptionEngineError(
                "Missing kroko-onnx model file: %s. Download a .data model from "
                "%s (%s), for example %s, then pass it as model or "
                "engine_options['model_path']." % (
                    path,
                    KROKO_ONNX_MODEL_URL,
                    KROKO_ONNX_HF_REPO,
                    DEFAULT_KROKO_ONNX_MODEL,
                )
            )
        return path

    def _recognizer_kwargs(self):
        options = self.engine_options
        kwargs = {
            "model_path": str(self.model_path),
            "key": options.get("key", ""),
            "referralcode": options.get("referralcode", ""),
            "num_threads": _int_option(options, "num_threads", 1),
            "provider": _provider_from_config(self.config, options),
            "sample_rate": self.sample_rate,
            "feature_dim": _int_option(options, "feature_dim", 80),
            "decoding_method": options.get("decoding_method", "greedy_search"),
            "max_active_paths": _int_option(options, "max_active_paths", 4),
            "hotwords_file": options.get("hotwords_file", ""),
            "hotwords_score": _float_option(options, "hotwords_score", 1.5),
            "blank_penalty": _float_option(options, "blank_penalty", 0.0),
            "enable_endpoint_detection": _bool_option(
                options,
                "enable_endpoint_detection",
                True,
            ),
            "rule1_min_trailing_silence": _float_option(
                options,
                "rule1_min_trailing_silence",
                2.4,
            ),
            "rule2_min_trailing_silence": _float_option(
                options,
                "rule2_min_trailing_silence",
                1.2,
            ),
            "rule3_min_utterance_length": _float_option(
                options,
                "rule3_min_utterance_length",
                20.0,
            ),
        }

        nested = options.get("recognizer", {})
        if nested is None:
            nested = {}
        if not isinstance(nested, dict):
            raise TranscriptionEngineError(
                "kroko-onnx option 'recognizer' must be a JSON object."
            )
        kwargs.update(nested)
        return kwargs

    def _create_recognizer(self, recognizer_cls):
        try:
            return recognizer_cls.from_transducer(**self._recognizer_kwargs())
        except AttributeError as exc:
            raise TranscriptionEngineError(
                "The installed 'kroko-onnx' package does not expose "
                "OnlineRecognizer.from_transducer."
            ) from exc

    def _as_float32_audio(self, audio):
        if hasattr(audio, "values"):
            audio = audio.values
        array = self.np.asarray(audio, dtype=self.np.float32)
        if getattr(array, "ndim", 1) > 1:
            array = array.reshape(-1)
        return array

    def _accept_waveform(self, stream, sample_rate, audio):
        try:
            stream.accept_waveform(sample_rate, audio)
        except TypeError:
            stream.accept_waveform(sample_rate=sample_rate, waveform=audio)

    def _decode_ready_stream(self, stream):
        if not hasattr(self.recognizer, "is_ready"):
            if hasattr(self.recognizer, "decode_stream"):
                self.recognizer.decode_stream(stream)
                return
            raise TranscriptionEngineError(
                "The installed 'kroko-onnx' recognizer exposes neither "
                "is_ready/decode_streams nor decode_stream."
            )

        while self.recognizer.is_ready(stream):
            if hasattr(self.recognizer, "decode_streams"):
                self.recognizer.decode_streams([stream])
            elif hasattr(self.recognizer, "decode_stream"):
                self.recognizer.decode_stream(stream)
            else:
                raise TranscriptionEngineError(
                    "The installed 'kroko-onnx' recognizer cannot decode streams."
                )

    def _result_text(self, stream):
        if hasattr(self.recognizer, "get_result"):
            result = self.recognizer.get_result(stream)
        else:
            result = getattr(stream, "result", "")
        return text_from_output(result)

    def transcribe(self, audio, **params):
        sample_rate = int(params.get("sample_rate", self.sample_rate))
        stream = self.recognizer.create_stream()
        audio = self._as_float32_audio(audio)
        self._accept_waveform(stream, sample_rate, audio)

        if self.tail_padding_seconds > 0:
            tail_padding = self.np.zeros(
                int(self.tail_padding_seconds * sample_rate),
                dtype=self.np.float32,
            )
            self._accept_waveform(stream, sample_rate, tail_padding)

        if hasattr(stream, "input_finished"):
            stream.input_finished()

        self._decode_ready_stream(stream)
        return KrokoOnnxDecodedOutput(
            text=self._result_text(stream),
            language=str(params.get("language") or _language_from_model_path(self.model_path)),
        )


class KrokoOnnxEngine(BaseTranscriptionEngine):
    engine_name = "kroko_onnx"

    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(config)
        self.backend = backend or (backend_cls or KrokoOnnxBackend)(config)

    def transcribe(self, audio, language=None, use_prompt=True):
        audio = self._normalize_audio(audio)
        options = self.config.engine_options or {}
        output = self.backend.transcribe(
            audio,
            language=options.get("language") or language,
        )
        detected_language = output.language or options.get("language") or language
        return TranscriptionResult(
            text=output.text.strip(),
            info=TranscriptionInfo(
                language=detected_language,
                language_probability=1.0 if detected_language else 0.0,
            ),
        )
