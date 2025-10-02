# client_sd_auto.py
import asyncio
import struct
import requests
import numpy as np
import sounddevice as sd
import websockets
from audio_utils import mulaw_encode, mulaw_decode, resample_to_8k, resample_to_16k

SERVER = "ws://213.173.107.102:14537/stream?session_id="
FRAME_MS = 20
TARGET_SR = 8000
FRAME_SAMPLES = int(TARGET_SR * FRAME_MS / 1000)  # 160

def pick_device():
    devs = sd.query_devices()
    in_idx = out_idx = None
    for i, d in enumerate(devs):
        if in_idx is None and d['max_input_channels'] > 0:
            in_idx = i
        if out_idx is None and d['max_output_channels'] > 0:
            out_idx = i
    return in_idx, out_idx

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

async def run_client(session_id: str):
    uri = SERVER + session_id
    async with websockets.connect(uri, max_size=None) as ws:
        print("Connected to Friday (auto-sr sounddevice client)")

        send_q = asyncio.Queue()   # float32 frames at TARGET_SR that will be sent upstream
        recv_q = asyncio.Queue()   # float32 frames at TARGET_SR received from server to play

        in_dev, out_dev = pick_device()
        if in_dev is None or out_dev is None:
            raise RuntimeError("No input/output audio device found. Run client on machine with mic/speaker.")

        # Determine default samplerates
        in_info = sd.query_devices(in_dev)
        out_info = sd.query_devices(out_dev)
        in_sr = safe_int(in_info.get('default_samplerate') or 0)
        out_sr = safe_int(out_info.get('default_samplerate') or 0)
        print(f"Input device #{in_dev}: {in_info['name']} default_sr={in_sr}")
        print(f"Output device #{out_dev}: {out_info['name']} default_sr={out_sr}")

        # Choose working samplerate strategy
        if in_sr and out_sr and in_sr == out_sr:
            sr = in_sr
            mode = "duplex"
            print(f"Using duplex stream at {sr} Hz")
        else:
            # fallback: use separate streams at their defaults (or system default)
            sr = in_sr or out_sr or safe_int(sd.default.samplerate) or 48000
            mode = "separate"
            print(f"Using separate streams: in_sr={in_sr}, out_sr={out_sr}, chosen sr={sr}")

        # compute blocksize in frames for callback
        blocksize = int(sr * FRAME_MS / 1000)

        loop = asyncio.get_event_loop()

        # Callback(s) produce/consume float32 arrays in [-1,1]
        def duplex_callback(indata, outdata, frames, time, status):
            if status:
                print("SoundDevice status:", status)
            # indata shape (frames, channels)
            if indata is not None:
                pcm = indata[:, 0].astype(np.float32)
                # resample input to 8k if needed
                if sr != TARGET_SR:
                    pcm8 = resample_to_8k(pcm, sr)
                else:
                    pcm8 = pcm
                # ensure length
                if len(pcm8) != FRAME_SAMPLES:
                    if len(pcm8) < FRAME_SAMPLES:
                        pcm8 = np.pad(pcm8, (0, FRAME_SAMPLES - len(pcm8)))
                    else:
                        pcm8 = pcm8[:FRAME_SAMPLES]
                loop.call_soon_threadsafe(send_q.put_nowait, pcm8)

            # Try to get frame to play
            try:
                out_pcm8 = recv_q.get_nowait()
                # resample to device sr if needed
                if sr != TARGET_SR:
                    out_pcm = resample_to_16k(out_pcm8, TARGET_SR) if sr == 16000 else np.interp(
                        np.linspace(0, len(out_pcm8), int(len(out_pcm8) * sr / TARGET_SR), endpoint=False),
                        np.arange(len(out_pcm8)), out_pcm8
                    )
                    # simpler: use resample_to_16k for 16k case; otherwise naive
                    # ensure length
                    if len(out_pcm) < frames:
                        out_pcm = np.pad(out_pcm, (0, frames - len(out_pcm)))
                    else:
                        out_pcm = out_pcm[:frames]
                else:
                    out_pcm = out_pcm8
                outdata[:] = out_pcm.reshape(-1, 1)
            except asyncio.QueueEmpty:
                outdata.fill(0)

        # separate streams approach (input and output callbacks)
        def input_callback(indata, frames, time, status):
            if status:
                print("Input status:", status)
            pcm = indata[:, 0].astype(np.float32)
            if sr != TARGET_SR:
                pcm8 = resample_to_8k(pcm, sr)
            else:
                pcm8 = pcm
            if len(pcm8) != FRAME_SAMPLES:
                if len(pcm8) < FRAME_SAMPLES:
                    pcm8 = np.pad(pcm8, (0, FRAME_SAMPLES - len(pcm8)))
                else:
                    pcm8 = pcm8[:FRAME_SAMPLES]
            loop.call_soon_threadsafe(send_q.put_nowait, pcm8)

        def output_callback(outdata, frames, time, status):
            if status:
                print("Output status:", status)
            try:
                out_pcm8 = recv_q.get_nowait()
                # resample to out_sr if needed
                if out_sr and out_sr != TARGET_SR:
                    out_pcm = resample_to_16k(out_pcm8, TARGET_SR) if out_sr == 16000 else np.interp(
                        np.linspace(0, len(out_pcm8), int(len(out_pcm8) * out_sr / TARGET_SR), endpoint=False),
                        np.arange(len(out_pcm8)), out_pcm8
                    )
                    if len(out_pcm) < frames:
                        out_pcm = np.pad(out_pcm, (0, frames - len(out_pcm)))
                    else:
                        out_pcm = out_pcm[:frames]
                else:
                    out_pcm = out_pcm8
                outdata[:] = out_pcm.reshape(-1, 1)
            except asyncio.QueueEmpty:
                outdata.fill(0)

        # Start stream(s)
        if mode == "duplex":
            try:
                print("tille here")
                stream = sd.Stream(device=4, samplerate=sr, blocksize=1024,
                                   dtype='float32', channels=1, callback=duplex_callback)
                stream.start()
                print("stream started")
            except Exception as e:
                print("Failed to open duplex stream:", e)
                print("Falling back to separate streams.")
                mode = "separate"

        if mode == "separate":
            # input stream with callback pushing to send_q
            in_stream = sd.InputStream(device=4, samplerate=in_sr or sr, blocksize=1024,
                                       dtype='float32', channels=1, callback=input_callback)
            out_stream = sd.OutputStream(device=4, samplerate=out_sr or sr, blocksize=1024,
                                         dtype='float32', channels=1, callback=output_callback)
            in_stream.start()
            out_stream.start()

        # Networking: sender / receiver run concurrently
        async def sender():
            seq = 0
            sent = 0
            while True:
                try:
                    pcm8 = await send_q.get()
                    mu = mulaw_encode(pcm8.astype(np.float32))
                    header = struct.pack(">IBH", seq, 0, FRAME_MS)
                    await ws.send(header + mu)
                    print("send first bytes")
                    seq = (seq + 1) & 0xFFFFFFFF
                    sent += 1      # <-- add
                    if sent % 50 == 0:   # print every ~1s (20ms * 50)
                        print(f"[client] sent frames={sent}, last_seq={seq}")   # <-- add
                except Exception as e:
                    print(e)
                    continue
        async def receiver():
            while True:
                msg = await ws.recv()
                if len(msg) < 7:
                    continue
                seq, codec, ms = struct.unpack(">IBH", msg[:7])
                payload = msg[7:]
                if codec != 0:
                    continue
                pcm8 = mulaw_decode(payload)
                print(pcm8)
                # ensure size
                if len(pcm8) < FRAME_SAMPLES:
                    pcm8 = np.pad(pcm8, (0, FRAME_SAMPLES - len(pcm8)))
                await recv_q.put(pcm8.astype(np.float32))

        await asyncio.gather(sender(), receiver())

if __name__ == "__main__":
    resp = requests.post("http://213.173.107.102:14537/handshake")
    sid = resp.json()["session_id"]
    print("Got session_id:", sid)
    try:
        asyncio.run(run_client(sid))
    except KeyboardInterrupt:
        print("Stopped by user")
