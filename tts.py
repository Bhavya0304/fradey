import os, subprocess, tempfile, numpy as np, soundfile as sf
from audio_utils import resample_to_8k

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
            print(f"[PIPER_DEBUG] running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            print(f"[PIPER_DEBUG] rc={proc.returncode} stdout_len={len(proc.stdout)} stderr_len={len(proc.stderr)}")
            if proc.stderr:
                print(f"[PIPER_DEBUG] stderr snippet: {proc.stderr.decode('utf-8', 'ignore')[:2000]}")
            if proc.returncode != 0:
                raise RuntimeError(f"Piper rc={proc.returncode}")
            pcm, sr = sf.read(outname, dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm[:,0]
            print(f"[PIPER_DEBUG] read wav sr={sr} len={len(pcm)} min={pcm.min():.6f} max={pcm.max():.6f}")
            return resample_to_8k(pcm, sr)
        finally:
            try:
                if outname and os.path.exists(outname):
                    os.remove(outname)
            except Exception:
                pass

class FallbackTTS:
    # Super basic CPU TTS (dev only). Replace ASAP with Piper.
    def synth(self, text: str):
        import pyttsx3
        engine = pyttsx3.init()
        # pyttsx3 can't easily give PCM directly; here we just return silence.
        # Replace with proper TTS if Piper unavailable.
        import numpy as np
        return np.zeros(16000, dtype=np.float32)  # 1s of silence

def build_tts():
    try:
        return PiperTTS()
    except Exception as e:
        print(f"[TTS] Piper not ready: {e}. Using fallback silence.")
        return FallbackTTS()
