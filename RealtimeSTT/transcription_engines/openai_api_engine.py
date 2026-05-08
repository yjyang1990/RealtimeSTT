from .base import BaseTranscriptionEngine, TranscriptionEngineError


class OpenAIAPIEngine(BaseTranscriptionEngine):
    engine_name = "openai_api"

    def __init__(self, config):
        super().__init__(config)
        raise TranscriptionEngineError(
            "The 'openai_api' transcription engine has not been wired yet. "
            "Add API request handling before selecting it."
        )

    def transcribe(self, audio, language=None, use_prompt=True):
        raise TranscriptionEngineError("The 'openai_api' transcription engine is not available.")
