from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_GRANITE_MODEL,
)


class GraniteSpeechBackend(_hf.GraniteSpeechBackend):
    pass


class GraniteSpeechEngine(_hf.GraniteSpeechEngine):
    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or GraniteSpeechBackend,
        )


__all__ = [
    "DEFAULT_GRANITE_MODEL",
    "GraniteSpeechBackend",
    "GraniteSpeechEngine",
]
