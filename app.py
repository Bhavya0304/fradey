import os, time, uuid, json, asyncio, struct
import numpy as np
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse
from schemas import HandshakeResponse, ControlEvent
from orchestrator import Session
from audio_utils import mulaw_decode, mulaw_encode
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="Friday")
SESSIONS: Dict[str, Dict] = {}

# Simple XOR "crypto" placeholder (replace with real)
CRYPTO = {"algo": "xor8", "key": "7F"}

@app.post("/handshake", response_model=HandshakeResponse)
def handshake():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"session": Session(sid), "clients": set()}
    return HandshakeResponse(session_id=sid, crypto=CRYPTO)

@app.post("/control")
def control(evt: ControlEvent):
    if evt.session_id not in SESSIONS:
        return JSONResponse({"ok": False, "error": "no such session"}, status_code=404)
    sess: Session = SESSIONS[evt.session_id]["session"]
    sess.push_ctrl(evt.dict())
    return {"ok": True}

@app.get("/diag/{session_id}")
def diag(session_id: str):
    return {"ok": session_id in SESSIONS}

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    params = dict(ws.query_params)
    sid = params.get("session_id")
    if not sid or sid not in SESSIONS:
        await ws.close(code=4000)
        return

    sess: Session = SESSIONS[sid]["session"]
    clients = SESSIONS[sid]["clients"]
    clients.add(ws)
    tts_ready = asyncio.Event()
    loop = asyncio.get_running_loop()
    def on_tts_ready():
        try:
            loop.call_soon_threadsafe(tts_ready.set)
        except Exception as e:
            print(e)
            return
    sess.run(on_tts_ready)
    

    # Binary frame schema: {seq:u32, codec:u8, ms:u16, payload:...}
    # We'll assume codec 0 = μ-law 8k

    async def sender():
        seq = 0
        while True:
            await tts_ready.wait()
            tts_ready.clear()
            # drain available TTS frames
            while not sess.tts_q.empty():
                chunk = sess.tts_q.get_nowait()  # f32 mono @ 8k (20ms)
                mu = mulaw_encode(chunk.astype(np.float32))
                header = struct.pack(">IBH", seq, 0, 20)  # big-endian
                try:
                    await ws.send_bytes(header + mu)
                except Exception:
                    return
                seq = (seq + 1) & 0xFFFFFFFF

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            data = await ws.receive_bytes()
            # parse header
            if len(data) < 7:
                continue
            seq, codec, ms = struct.unpack(">IBH", data[:7])
            payload = data[7:]
            # DEBUG: print incoming frame summary
            

            if codec != 0:
                # unsupported codec
                continue

            # decode μ-law -> f32 (mono 8k)
            pcm_f32 = mulaw_decode(payload)
            sess.push_frame(pcm_f32)

    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
        if len(clients) == 0:
            sess.stop.set()
            del SESSIONS[sid]
        try:
            sender_task.cancel()
        except Exception:
            pass
        await ws.close()
