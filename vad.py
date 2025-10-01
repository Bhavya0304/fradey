import webrtcvad
import numpy as np

class VADGate:
    """
    Gate speech into segments: returns True while speaking, False when in pause.
    Works on 20ms 16-bit mono frames at 8k or 16k.
    """
    def __init__(self, sample_rate=16000, aggressiveness=2, pause_ms=400):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_ms = 20
        self.pause_ms = pause_ms
        self._silence_run = 0
        self._speaking = False

    def is_speech(self, frame_pcm16: bytes) -> bool:
        return self.vad.is_speech(frame_pcm16, self.sample_rate)

    def update(self, frame_pcm16: bytes):
        speech = self.is_speech(frame_pcm16)
        if speech:
            self._silence_run = 0
            self._speaking = True
        else:
            self._silence_run += self.frame_ms
            if self._silence_run >= self.pause_ms:
                self._speaking = False
        return speech, self._speaking, (self._silence_run >= self.pause_ms)
