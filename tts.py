# tts.py
import os, subprocess, tempfile, numpy as np, soundfile as sf
from audio_utils import resample_to_8k
import warnings

# Prefer Piper binary if user set it (keeps your original binary usage)
class PiperTTS:
    def __init__(self):
        self.piper_bin = os.getenv("PIPER_BIN", "")
        self.voice = os.getenv("PIPER_VOICE", "")
        if not (self.piper_bin and os.path.isfile(self.piper_bin)):
            raise RuntimeError("PIPER_BIN or PIPER_VOICE not set/invalid")

    def synth(self, text: str) -> np.ndarray:
        outname = None
        try:
            outf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            outname = outf.name
            outf.close()
            cmd = [self.piper_bin]
            if self.voice:
                cmd += ["-m", self.voice]
            cmd += ["-f", outname, "-q"]
            proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if proc.returncode != 0:
                raise RuntimeError(f"Piper rc={proc.returncode} stderr={proc.stderr[:1000]}")
            pcm, sr = sf.read(outname, dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm[:,0]
            return resample_to_8k(pcm, sr)
        finally:
            try:
                if outname and os.path.exists(outname):
                    os.remove(outname)
            except Exception:
                pass

# If Piper not available, fall back to a GPU-capable Coqui/TTS model (better voice than silence)
class CoquiTTS:
    def __init__(self, model_name: str = "tts_models/en/vctk/vits"):
        # lazy import to avoid heavy import unless used
        try:
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError("Coqui TTS package not installed or failed to load. Install with the provided setup script.") from e
        # choose a relatively lightweight GPU-friendly model; VITS is high-quality and fast on GPU
        self.tts = TTS(model_name, progress_bar=False, gpu=True)
        self.sr = 22050  # model sample rate

    def synth(self, text: str) -> np.ndarray:
        # returns float32 array at 8k sample rate to match your pipeline
        wav = self.tts.tts(text,speaker="p229",language="hi")
        # wav is float32 in [-1,1]
        return resample_to_8k(np.asarray(wav, dtype=np.float32), self.sr)


class FallbackTTS:
    # Keep same return shape: float32 PCM sampled at 8k
    def synth(self, text: str):
        import numpy as np
        # 1 second of silence at 8k
        return np.zeros(8000, dtype=np.float32)

def build_tts():
    """
    Tries Piper (binary) first, then Coqui TTS on GPU, then fallback.
    """
    try:
        return CoquiTTS()
        
    except Exception as e:
        print(f"[TTS] Coqui TTS not ready: {e}. Using fallback silence.")
    try:
        return PiperTTS()
    except Exception as e:
        print(f"[TTS] Piper not ready: {e}. Trying Coqui TTS (gpu)...")
        return FallbackTTS()
