# orchestrator.py
import time, threading, queue, numpy as np, math
from audio_utils import resample_to_16k
from vad import VADGate
import time
# Note: import heavy libs inside the worker thread below
# from stt import STTEngine
# from llm import LLMEngine
# from tts import build_tts

class Session:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.in_sr = 8000
        self.vad = VADGate(sample_rate=8000, aggressiveness=3, pause_ms=200)

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
        try:
            from stt import STTEngine
            from llm import LLMEngine, LLMEngineGroq
            from tts import build_tts
        except Exception as e:
            print(f"[session {self.session_id}] Failed to import model modules in worker: {e}")
            # If imports fail, keep running but do not process segments
            

        try:
            # Initialize models here — on this thread
            # NOTE: if you want to force CPU for debugging, set device="cpu" here
            self.stt = STTEngine(model_size="small", device="cuda", compute_type="float16")
            self.llm = LLMEngineGroq()
            self.tts = build_tts()
            self._silence_counter = 0
            self._speech_started = False
            self._max_segment_frames = int(10 / 0.02)  # 30s max -> 1500 frames (tweak as needed)
            self._pause_frames = int(0.4 / 0.02)  # matches pause_ms=400 -> 20 frames at 20ms
            print(f"[session {self.session_id}] models initialized in worker thread")
        except Exception as e:
            print(f"[session {self.session_id}] model init failed: {e}")
            # If model init fails, don't crash the whole server — just stop processing
            
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
                #print(f"[{self.session_id}] DEBUG _consume_audio: got frame len={len(frame)} dtype={frame.dtype} rms={rms:.6f} in_q={self.in_q.qsize()} partial_len={len(self.partial_buf)}")
                speech, speaking, segment_done = self.vad.update(frame_pcm16)
                #print(f"[{self.session_id}] DEBUG VAD -> speech={speech} speaking={speaking} segment_done={segment_done}")
            except Exception as e:
                # If VAD errors, treat as no-speech and continue
                speech, speaking, segment_done = False, False, False
                # optional: log e

            if speech:
                # we observed speech in this frame
                self.partial_buf.append(frame)
                self._speech_started = True
                self._silence_counter = 0
            else:
                # no speech in this frame
                if self._speech_started:
                    self._silence_counter += 1
                # if silence persisted beyond pause threshold, force segment end
            # force segment_done if VAD says so OR our silence counter passes threshold OR segment too long
            forced_segment = False
            if getattr(self, "_pause_frames", None) is not None:
                if self._silence_counter >= self._pause_frames and self._speech_started:
                    forced_segment = True

            if segment_done or forced_segment or (len(self.partial_buf) >= self._max_segment_frames):
                if len(self.partial_buf) > 0:
                    segment = np.concatenate(self.partial_buf, axis=0)
                    print(f"[{self.session_id}] DEBUG segment finalized: frames={len(segment)} samples={segment.shape} forced={forced_segment} silence_count={self._silence_counter} partial_buf_before_clear={len(self.partial_buf)}")
                    self.partial_buf.clear()
                    # reset VAD helper state
                    self._speech_started = False
                    self._silence_counter = 0
                    # enqueue for processing
                    try:
                        self.proc_q.put_nowait(segment)
                    except queue.Full:
                        try:
                            _ = self.proc_q.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.proc_q.put_nowait(segment)
                        except Exception:
                            print(f"[{self.session_id}] WARNING: proc_q full, dropped segment")

    def _worker_thread(self):
        """
        Worker: STT -> LLM -> TTS on a single thread to avoid cross-thread CUDA/context issues.
        Produces float32 mono frames at out_sr (16 kHz) in fixed 20 ms chunks (frame_size=320).
        """
        # target output sampling rate and frame size
        out_sr = 16000
        frame_ms = 20
        frame_size = int((frame_ms / 1000.0) * out_sr)  # 320 samples for 20ms @ 16k
        hop = frame_size  # no overlap, send discrete 20ms frames

        # helper accessor for resampling function - assume available in module scope
        # resample_to_16k(pcm_f32, in_sr) should exist (you provided earlier)
        while not self.stop.is_set():
            try:
                segment = self.proc_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # STT
            try:
                STT_start_time = time.perf_counter()
                text = self.stt.transcribe_chunk(segment, lang="en")
                STT_end_time = time.perf_counter()
                print(f"STT Time: {(STT_end_time - STT_start_time):.6f} seconds")
            except Exception as e:
                print(f"[session {self.session_id}] STT error: {e}")
                continue
            print("text: " + text)
            # if not text:
            #     continue

            # LLM
            try:
                LLM_start_time = time.perf_counter()
                reply = self.llm.reply(text)
                LLM_end_time = time.perf_counter()
                print(f"LLM Time: {(LLM_end_time - LLM_start_time):.6f} seconds")
            except Exception as e:
                print(f"[session {self.session_id}] LLM error: {e}")
                continue

            # TTS
            try:
                TTS_start_time = time.perf_counter()
                # Prefer TTS that can be asked to synthesize at 16k. If your TTS supports a sr param, use it.
                # Example: tts_pcm = self.tts.synth(reply, sr=out_sr)
                # Otherwise, we accept whatever it returns and resample below.
                tts_out = self.tts.synth(reply)   # may be np.ndarray float32 mono at unknown sr
                TTS_end_time = time.perf_counter()
                print(f"TTS Time: {(TTS_end_time - TTS_start_time):.6f} seconds")
            except Exception as e:
                print(f"[session {self.session_id}] TTS error: {e}")
                continue

            # Normalize to numpy float32 mono
            try:
                tts_pcm = np.asarray(tts_out, dtype=np.float32).flatten()
            except Exception as e:
                print(f"[session {self.session_id}] TTS output conversion error: {e}")
                continue

            # Determine source sample rate for TTS output if available, fallback heuristic to 8k
            tts_sr = None
            # Try common attributes
            if hasattr(self.tts, "sample_rate"):
                tts_sr = getattr(self.tts, "sample_rate")
            elif hasattr(self.tts, "sr"):
                tts_sr = getattr(self.tts, "sr")
            # If still unknown, try heuristic: if length is small and likely 8k-based
            if tts_sr is None:
                # try to guess: if last chunk length divisible by 160 -> maybe 8k; divisible by 320 -> maybe 16k
                # This is a heuristic only; prefer explicit tts.sample_rate on your TTS class.
                if tts_pcm.size % 320 == 0:
                    tts_sr = out_sr
                elif tts_pcm.size % 160 == 0:
                    tts_sr = 8000
                else:
                    # default guess: 8000 (your previous code used 8k)
                    tts_sr = 8000

            # Resample to out_sr (16k) if needed
            if int(tts_sr) != out_sr:
                try:
                    tts_pcm = resample_to_16k(tts_pcm, int(tts_sr))
                except Exception as e:
                    print(f"[session {self.session_id}] resample_to_16k error: {e}")
                    continue

            # Ensure float32 range is reasonable (-1..1)
            if tts_pcm.dtype != np.float32:
                tts_pcm = tts_pcm.astype(np.float32)
            # If values look large (int16 miscast), normalize:
            if np.max(np.abs(tts_pcm)) > 10.0:
                # likely int16 range; scale down
                tts_pcm = np.clip(tts_pcm / 32768.0, -1.0, 1.0)

            # Chunk into fixed 20ms frames (frame_size) and push to tts_q
            CHUNK_start_time = time.perf_counter()
            total_samples = tts_pcm.shape[0]
            # iterate in frame_size steps and pad the last frame if needed
            for start in range(0, total_samples, hop):
                frame = tts_pcm[start:start + frame_size]
                if frame.shape[0] < frame_size:
                    # pad with zeros to exact frame_size
                    pad = np.zeros(frame_size - frame.shape[0], dtype=np.float32)
                    frame = np.concatenate((frame, pad))
                # Non-blocking push with timeout; drop if full
                try:
                    # push the numpy array (float32, length frame_size). Sender expects float32 arrays.
                    self.tts_q.put(frame, timeout=0.5)
                except queue.Full:
                    # queue full; drop this frame to avoid blocking worker
                    print(f"[session {self.session_id}] tts_q full, dropping frame")
                    continue
            CHUNK_end_time = time.perf_counter()
            print(f"CHUNK Time: {(CHUNK_end_time - CHUNK_start_time):.6f} seconds")

            # signal sender that TTS frames are available
            try:
                if hasattr(self, "_on_tts_ready") and callable(self._on_tts_ready):
                    self._on_tts_ready()
            except Exception as e:
                print(f"[session {self.session_id}] on_tts_ready error: {e}")


    # optional: cleanup helper
    def shutdown(self):
        self.stop.set()
        # optionally join threads if you saved references
