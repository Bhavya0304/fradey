import time, threading, queue, numpy as np
from vad import VADGate
from stt import STTEngine
from llm import LLMEngine
from tts import build_tts
import math

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.in_sr = 8000
        self.vad = VADGate(sample_rate=8000, aggressiveness=0, pause_ms=600)
        self.in_q = queue.Queue(maxsize=200)   # incoming PCM f32 frames (20ms @ 8k -> 160 samples)
        self.tts_q = queue.Queue(maxsize=16000)  # outgoing PCM f32 frames (8k)
        self.ctrl_q = queue.Queue()
        self.stop = threading.Event()
        self.stt = STTEngine(model_size="small", device="cuda", compute_type="float16")
        self.llm = LLMEngine(ctx_size=2048, n_gpu_layers=20)
        self.tts = build_tts()
        self.partial_buf = []   # collect active speech
        self.lock = threading.Lock()

    def push_frame(self, pcm_f32_8k: np.ndarray):
        try:
            self.in_q.put_nowait(pcm_f32_8k)
        except queue.Full:
            _ = self.in_q.get_nowait()
            self.in_q.put_nowait(pcm_f32_8k)

    def push_ctrl(self, evt):
        self.ctrl_q.put(evt)

    def run(self, on_tts_ready):
        threading.Thread(target=self._consume_audio, args=(on_tts_ready,), daemon=True).start()

    def _consume_audio(self, on_tts_ready):
        frame_nsamps = int(0.02 * self.in_sr)  # 20ms
        while not self.stop.is_set():
            try:
                frame = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            rms = math.sqrt((frame.astype('float32')**2).mean())
            #print(f"[session {self.session_id}] frame_rms={rms:.5f}")        
            # VAD expects 16-bit bytes for the frame
            frame_pcm16 = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
            speech, speaking, segment_done = self.vad.update(frame_pcm16)
            # DEBUG: print vad state occasionally
            # if len(self.partial_buf) % 50 == 0:  # not to spam, only occasional
            #     print(f"[session {self.session_id}] vad: speech={speech} speaking={speaking} segment_done={segment_done} partial_buf_len={len(self.partial_buf)}")


            if speech:
                self.partial_buf.append(frame)

            if segment_done and len(self.partial_buf) > 0:
                segment = np.concatenate(self.partial_buf, axis=0)
                self.partial_buf.clear()
                # Process segment -> STT -> LLM -> TTS
                threading.Thread(target=self._process_turn, args=(segment, on_tts_ready), daemon=True).start()

    def _process_turn(self, segment_f32_8k: np.ndarray, on_tts_ready):
        text = self.stt.transcribe_chunk(segment_f32_8k, lang="en")
        print(text)
        if not text:
            return
        reply = self.llm.reply(text)
        print(reply)
        # Stream TTS audio
        tts_pcm_8k = self.tts.synth(reply)
        print(tts_pcm_8k)
        # chop into 20ms frames for sender
        hop = int(0.02 * 8000)
        for i in range(0, len(tts_pcm_8k), hop):
            chunk = tts_pcm_8k[i:i+hop]
            if len(chunk) < hop:
                # pad
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            try:
                self.tts_q.put(chunk, timeout=1.0)
            except queue.Full:
                pass
        # signal available
        on_tts_ready()
