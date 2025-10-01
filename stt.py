import numpy as np
from faster_whisper import WhisperModel
from audio_utils import resample_to_16k, pcm16_from_f32

class STTEngine:
    def __init__(self, model_size="tiny", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_chunk(self, pcm_f32_8k: np.ndarray, lang="en"):
        # upsample to 16k for Whisper
        pcm16_16k = pcm16_from_f32(resample_to_16k(pcm_f32_8k, 8000))
        # faster-whisper accepts numpy float32 in [-1,1], but we’ll pass f32:
        pcm_f32_16k = pcm16_16k.astype(np.float32) / 32768.0
        segs, info = self.model.transcribe(
            pcm_f32_16k,
            language=lang,
            vad_filter=False,  # we already gated with VAD
            beam_size=1,
            word_timestamps=False,
            condition_on_previous_text=False,
            temperature=[0.0],
        )
        text = "".join(s.text for s in segs).strip()
        return text
