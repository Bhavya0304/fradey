# client_simple.py
import asyncio
import struct
import requests
import queue
import time
import numpy as np
import sounddevice as sd
import websockets
from audio_utils import mulaw_encode, mulaw_decode, resample_to_8k, resample_to_16k

SERVER = "ws://213.173.102.214:15469/stream?session_id="
HANDSHAKE_URL = "http://213.173.102.214:15469/handshake"
FRAME_MS = 20
TARGET_SR = 8000
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)  # 160

# Thread-safe queues for audio <-> network
send_q = queue.Queue(maxsize=1024)   # audio callback -> sender coroutine (blocking get in executor)
recv_q = queue.Queue(maxsize=1024)   # receiver coroutine -> audio callback (get_nowait in audio)

def pick_default_device():
    """Return (in_dev, out_dev) indices (may be None)."""
    try:
        devs = sd.query_devices()
        in_idx = out_idx = None
        for i, d in enumerate(devs):
            if in_idx is None and d.get('max_input_channels', 0) > 0:
                in_idx = 13
            if out_idx is None and d.get('max_output_channels', 0) > 0:
                out_idx = 13
            if in_idx is not None and out_idx is not None:
                break
        return in_idx, out_idx
    except Exception:
        return None, None

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

async def run_client(session_id: str):
    uri = SERVER + session_id
    print("Connecting to", uri)
    async with websockets.connect(uri, max_size=None) as ws:
        print("Connected. Starting audio streams and networking.")

        # pick devices and sample rates
        in_dev, out_dev = pick_default_device()
        if in_dev is None or out_dev is None:
            raise RuntimeError("No input/output devices found.")

        in_info = sd.query_devices(in_dev)
        out_info = sd.query_devices(out_dev)
        in_sr = safe_int(in_info.get('default_samplerate') or 0)
        out_sr = safe_int(out_info.get('default_samplerate') or 0)
        print(f"Input device #{in_dev}: {in_info['name']} sr={in_sr}")
        print(f"Output device #{out_dev}: {out_info['name']} sr={out_sr}")

        # choose a convenience stream sample rate for device callbacks (we'll resample to TARGET_SR)
        # prefer matching rates; otherwise choose input device's rate or system default
        stream_sr = in_sr or out_sr or safe_int(sd.default.samplerate) or 48000
        blocksize = int(stream_sr * FRAME_MS / 1000)

        loop = asyncio.get_event_loop()

        # AUDIO CALLBACKS (must be fast)
        def input_callback(indata, frames, time_info, status):
            if status:
                print("Input status:", status)
            # take first channel
            pcm = indata[:, 0].astype(np.float32)
            # resample to 8k if needed
            if stream_sr != TARGET_SR:
                try:
                    pcm8 = resample_to_8k(pcm, stream_sr)
                except Exception:
                    # fallback naive resample if your function misbehaves
                    pcm8 = np.interp(
                        np.linspace(0, len(pcm), FRAME_SAMPLES, endpoint=False),
                        np.arange(len(pcm)), pcm
                    )
            else:
                pcm8 = pcm
            # ensure exact length
            if len(pcm8) < FRAME_SAMPLES:
                pcm8 = np.pad(pcm8, (0, FRAME_SAMPLES - len(pcm8)))
            else:
                pcm8 = pcm8[:FRAME_SAMPLES]
            # push to thread-safe queue; drop if full (do not block audio thread)
            try:
                send_q.put_nowait(pcm8.astype(np.float32))
            except queue.Full:
                # dropping frame is better than blocking the audio callback
                # minimal diagnostic to avoid spamming
                if send_q.qsize() % 50 == 0:
                    print("send_q full — dropping frames")

        def output_callback(outdata, frames, time_info, status):
            if status:
                print("Output status:", status)
            try:
                out_pcm8 = recv_q.get_nowait()  # expects float32 at TARGET_SR
                # resample to device sample rate if needed
                if out_sr and out_sr != TARGET_SR:
                    if out_sr == 16000:
                        out_pcm = resample_to_16k(out_pcm8, TARGET_SR)
                    else:
                        out_pcm = np.interp(
                            np.linspace(0, len(out_pcm8), int(len(out_pcm8) * out_sr / TARGET_SR), endpoint=False),
                            np.arange(len(out_pcm8)), out_pcm8
                        )
                    # ensure length
                    if len(out_pcm) < frames:
                        out_pcm = np.pad(out_pcm, (0, frames - len(out_pcm)))
                    else:
                        out_pcm = out_pcm[:frames]
                else:
                    out_pcm = out_pcm8
                    if len(out_pcm) < frames:
                        out_pcm = np.pad(out_pcm, (0, frames - len(out_pcm)))
                    else:
                        out_pcm = out_pcm[:frames]
                outdata[:] = out_pcm.reshape(-1, 1)
            except queue.Empty:
                outdata.fill(0)

        # start streams (input and output); prefer separate streams for simplicity
        in_stream = sd.InputStream(device=in_dev, samplerate=stream_sr, blocksize=blocksize,
                                   dtype='float32', channels=1, callback=input_callback)
        out_stream = sd.OutputStream(device=out_dev, samplerate=out_sr or stream_sr, blocksize=blocksize,
                                     dtype='float32', channels=1, callback=output_callback)
        in_stream.start()
        out_stream.start()
        print("Audio streams started (non-blocking).")

        # NETWORK TASKS
        async def sender():
            seq = 0
            sent = 0
            while True:
                # get a frame from the blocking, thread-safe queue without blocking event loop
                pcm8 = await loop.run_in_executor(None, send_q.get)
                try:
                    mu = mulaw_encode(pcm8.astype(np.float32))
                except Exception as e:
                    print("mulaw_encode error:", e)
                    continue
                header = struct.pack(">IBH", seq, 0, FRAME_MS)
                try:
                    await ws.send(header + mu)
                except Exception as e:
                    print("ws.send error:", e)
                    # if connection dies, raise to let outer scope handle reconnection/exit
                    raise
                seq = (seq + 1) & 0xFFFFFFFF
                sent += 1
                if sent % 50 == 0:
                    print(f"[client] sent frames={sent}, last_seq={seq}")

        async def receiver():
            while True:
                try:
                    msg = await ws.recv()
                except Exception as e:
                    print("ws.recv error:", e)
                    raise
                if not msg or len(msg) < 7:
                    continue
                seq, codec, ms = struct.unpack(">IBH", msg[:7])
                payload = msg[7:]
                if codec != 0:
                    # ignore unknown codec
                    continue
                try:
                    pcm8 = mulaw_decode(payload)  # should return numpy float32 or int16-like (we expect float32)
                except Exception as e:
                    print("mulaw_decode error:", e)
                    continue
                # coerce to float32 in [-1,1]
                pcm8 = np.asarray(pcm8, dtype=np.float32)
                if len(pcm8) < FRAME_SAMPLES:
                    pcm8 = np.pad(pcm8, (0, FRAME_SAMPLES - len(pcm8)))
                elif len(pcm8) > FRAME_SAMPLES:
                    pcm8 = pcm8[:FRAME_SAMPLES]
                # push into thread-safe recv queue for audio callback
                try:
                    recv_q.put_nowait(pcm8)
                except queue.Full:
                    # drop if playback queue is full
                    pass

        # run sender & receiver until one fails (then let exception bubble up)
        try:
            await asyncio.gather(sender(), receiver())
        finally:
            in_stream.stop()
            out_stream.stop()
            in_stream.close()
            out_stream.close()
            print("Audio streams stopped.")

def do_handshake_and_run():
    resp = requests.post(HANDSHAKE_URL)
    sid = resp.json().get("session_id")
    print("Got session_id:", sid)
    if not sid:
        raise RuntimeError("Handshake failed: no session_id")
    asyncio.run(run_client(sid))

if __name__ == "__main__":
    try:
        do_handshake_and_run()
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print("Fatal:", e)
