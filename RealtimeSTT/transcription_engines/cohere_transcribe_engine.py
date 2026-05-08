from . import hf_transformers_engines as _hf
from .hf_transformers_engines import (
    DEFAULT_COHERE_MODEL,
)


class CohereTranscribeBackend(_hf.CohereTranscribeBackend):
    pass


class CohereTranscribeEngine(_hf.CohereTranscribeEngine):
    def __init__(self, config, backend=None, backend_cls=None):
        super().__init__(
            config,
            backend=backend,
            backend_cls=backend_cls or CohereTranscribeBackend,
        )


__all__ = [
    "DEFAULT_COHERE_MODEL",
    "CohereTranscribeBackend",
    "CohereTranscribeEngine",
]
