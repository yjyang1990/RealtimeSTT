from faster_whisper import BatchedInferencePipeline
import faster_whisper

from .base import BaseTranscriptionEngine, TranscriptionInfo, TranscriptionResult


class FasterWhisperEngine(BaseTranscriptionEngine):
    engine_name = "faster_whisper"

    def __init__(self, config):
        super().__init__(config)
        model = faster_whisper.WhisperModel(
            model_size_or_path=self.config.model,
            device=self.config.device,
            compute_type=self.config.compute_type,
            device_index=self.config.gpu_device_index,
            download_root=self.config.download_root,
        )
        if self.config.batch_size > 0:
            model = BatchedInferencePipeline(model=model)
        self.model = model

    def transcribe(self, audio, language=None, use_prompt=True):
        audio = self._normalize_audio(audio)
        kwargs = {
            "language": language if language else None,
            "beam_size": self.config.beam_size,
            "initial_prompt": self._get_prompt(use_prompt),
            "suppress_tokens": self.config.suppress_tokens,
            "vad_filter": self.config.vad_filter,
        }
        if self.config.batch_size > 0:
            kwargs["batch_size"] = self.config.batch_size

        segments, info = self.model.transcribe(audio, **kwargs)
        text = " ".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            info=TranscriptionInfo(
                language=getattr(info, "language", None),
                language_probability=getattr(info, "language_probability", 0.0),
            ),
        )
