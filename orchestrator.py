# orchestrator.py
import time, threading, queue, numpy as np, math
from vad import VADGate
# Note: import heavy libs inside the worker thread below
# from stt import STTEngine
# from llm import LLMEngine
# from tts import build_tts

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.in_sr = 8000
        self.vad = VADGate(sample_rate=8000, aggressiveness=0, pause_ms=600)

        # Queues
        self.in_q = queue.Queue(maxsize=200)    # incoming PCM f32 frames (20ms @ 8k -> 160 samples)
        self.tts_q = queue.Queue(maxsize=16000) # outgoing PCM f32 frames (8k)
        self.ctrl_q = queue.Queue()
        self.proc_q = queue.Queue()             # queue of segments to process (STT->LLM->TTS)

        self.stop = threading.Event()
        self.partial_buf = []
        self.lock = threading.Lock()

        # We'll lazy-init heavy model objects on the dedicated worker thread
        self.stt = None
        self.llm = None
        self.tts = None

        # Start two threads:
        #  - audio consumer (reads in_q, VAD, assembles segments)
        #  - processing worker (initializes models and processes segments sequentially)
        threading.Thread(target=self._consume_audio, args=(None,), daemon=True).start()
        threading.Thread(target=self._worker_thread, daemon=True).start()

    def push_frame(self, pcm_f32_8k: np.ndarray):
        try:
            self.in_q.put_nowait(pcm_f32_8k)
        except queue.Full:
            try:
                _ = self.in_q.get_nowait()
            except Exception:
                pass
            try:
                self.in_q.put_nowait(pcm_f32_8k)
            except Exception:
                pass

    def push_ctrl(self, evt):
        self.ctrl_q.put(evt)

    def run(self, on_tts_ready):
        # For backward compatibility with your server code that expects run(on_tts_ready)
        # simply set an attr that _consume_audio will call into when TTS is produced.
        self._on_tts_ready = on_tts_ready

    def _consume_audio(self, _unused_on_tts):
        """Read frames from in_q, run VAD and assemble segments, then push to proc_q."""
        frame_nsamps = int(0.02 * self.in_sr)  # 20ms frames
        while not self.stop.is_set():
            try:
                frame = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # VAD expects 16-bit bytes for the frame
            try:
                rms = math.sqrt((frame.astype('float32')**2).mean())
            except Exception:
                rms = 0.0

            frame_pcm16 = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
            try:
                speech, speaking, segment_done = self.vad.update(frame_pcm16)
            except Exception as e:
                # If VAD errors, treat as no-speech and continue
                speech, speaking, segment_done = False, False, False
                # optional: log e

            if speech:
                self.partial_buf.append(frame)

            if segment_done and len(self.partial_buf) > 0:
                segment = np.concatenate(self.partial_buf, axis=0)
                self.partial_buf.clear()
                # enqueue segment for processing by the single worker thread
                try:
                    self.proc_q.put_nowait(segment)
                except queue.Full:
                    # drop the segment if worker is overwhelmed; better than blocking
                    try:
                        _ = self.proc_q.get_nowait()
                    except Exception:
                        pass
                    try:
                        self.proc_q.put_nowait(segment)
                    except Exception:
                        pass

    def _worker_thread(self):
        """
        Dedicated worker thread which:
          - initializes STT/LLM/TTS models on this thread
          - processes segments sequentially (STT -> LLM -> TTS)
        Doing everything on one thread avoids cross-thread CUDA/context issues that often cause segfaults.
        """
        # Import/construct heavy objects here (on the worker thread)
        try:
            from stt import STTEngine
            from llm import LLMEngine
            from tts import build_tts
        except Exception as e:
            print(f"[session {self.session_id}] Failed to import model modules in worker: {e}")
            # If imports fail, keep running but do not process segments
            return

        try:
            # Initialize models here — on this thread
            # NOTE: if you want to force CPU for debugging, set device="cpu" here
            self.stt = STTEngine(model_size="small", device="cuda", compute_type="float16")
            self.llm = LLMEngine(ctx_size=2048, n_gpu_layers=20)
            self.tts = build_tts()
            print(f"[session {self.session_id}] models initialized in worker thread")
        except Exception as e:
            print(f"[session {self.session_id}] model init failed: {e}")
            # If model init fails, don't crash the whole server — just stop processing
            return

        hop = int(0.02 * self.in_sr)  # 160 samples @ 8k
        while not self.stop.is_set():
            try:
                segment = self.proc_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # STT
                text = self.stt.transcribe_chunk(segment, lang="en")
            except Exception as e:
                print(f"[session {self.session_id}] STT error: {e}")
                continue

            if not text:
                continue

            try:
                reply = self.llm.reply(text)
            except Exception as e:
                print(f"[session {self.session_id}] LLM error: {e}")
                continue

            try:
                tts_pcm_8k = self.tts.synth(reply)
            except Exception as e:
                print(f"[session {self.session_id}] TTS error: {e}")
                continue

            # chunk and enqueue to tts_q for sender to pick up
            for i in range(0, len(tts_pcm_8k), hop):
                chunk = tts_pcm_8k[i:i+hop]
                if len(chunk) < hop:
                    chunk = np.pad(chunk, (0, hop - len(chunk)))
                try:
                    self.tts_q.put(chunk, timeout=1.0)
                except queue.Full:
                    # if queue full, skip chunk
                    pass

            # signal available frames (preserve backward compat with your server)
            try:
                if hasattr(self, "_on_tts_ready") and callable(self._on_tts_ready):
                    self._on_tts_ready()
            except Exception as e:
                # don't let signaling kill the worker
                print(f"[session {self.session_id}] on_tts_ready error: {e}")

    # optional: cleanup helper
    def shutdown(self):
        self.stop.set()
        # optionally join threads if you saved references
