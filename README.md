# Friday (Sprint-1)

Minimal IVR brain in Python:
- `POST /handshake` → `session_id`
- `WS /stream?session_id=...` binary μ-law 8k (20ms frames)
- `POST /control` for DTMF/menu
- VAD-gated turn-taking → STT (faster-whisper) → LLM (llama.cpp) → TTS (piper) → stream back

## 1) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
