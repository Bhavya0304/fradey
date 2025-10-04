# stt.py
import numpy as np
from faster_whisper import WhisperModel
from audio_utils import resample_to_16k, pcm16_from_f32
import os, warnings

class STTEngine:
    """
    GPU-enabled faster-whisper wrapper tuned for call-quality usage on an RTX GPU.
    Keeps same interface: transcribe_chunk(pcm_f32_8k: np.ndarray, lang="en") -> str
    """
    def __init__(self, model_size="small", device=None, compute_type=None):
        # Default to GPU float16 for a good tradeoff of quality and speed on RTX Ada
        device = device or ("cuda" if os.environ.get("USE_CPU") is None else "cpu")
        compute_type = compute_type or "float16"  # faster + good quality on modern GPUs
        # For multilingual heavy workloads you might pick "small" or "base" -- "small" is a good call-quality step-up from "tiny"
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            self.device = device
            self.compute_type = compute_type
        except Exception as e:
            warnings.warn(f"[STTEngine] GPU model init failed ({e}), falling back to cpu tiny. Error: {e}")
            # fallback to cpu tiny if GPU initialization fails
            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")
            self.device = "cpu"
            self.compute_type = "int8"

    def transcribe_chunk(self, pcm_f32_8k: np.ndarray, lang="hi"):
        # upsample to 16k for Whisper as before
        pcm16_16k = pcm16_from_f32(resample_to_16k(pcm_f32_8k, 8000))
        pcm_f32_16k = pcm16_16k.astype(np.float32) / 32768.0

        # Use single-shot transcribe for short segments (call audio). Keep beam_size small for latency.
        segs, info = self.model.transcribe(
            pcm_f32_16k,
            language=lang,
            vad_filter=False,
            beam_size=1,               # lower latency; increase to 3 for slightly better accuracy at cost of time
            word_timestamps=False,
            condition_on_previous_text=False,
            temperature=[0.0],
            # no timestamps streaming here; returning final text like original
        )
        text = "".join(s.text for s in segs).strip()
        return text
