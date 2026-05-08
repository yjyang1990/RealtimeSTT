from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_MOONSHINE_MODEL,
)


class MoonshineBackend(_hf.MoonshineBackend):
    pass


class MoonshineEngine(_hf.MoonshineEngine):
    def __init__(self, config, backend=None, backend_cls=None):
        engine_options = config.engine_options or {}
        backend_name = str(engine_options.get("backend", "")).lower().replace("-", "_")
        if backend is None and backend_cls is None and backend_name == "sherpa_onnx":
            from .sherpa_onnx_engine import SherpaOnnxMoonshineBackend

            backend_cls = SherpaOnnxMoonshineBackend
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or MoonshineBackend,
        )


__all__ = [
    "DEFAULT_MOONSHINE_MODEL",
    "MoonshineBackend",
    "MoonshineEngine",
]
