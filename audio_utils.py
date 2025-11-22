import numpy as np
from scipy.signal import resample_poly

# # μ-law tables
# MU = 255

# def mulaw_decode(mu_bytes: bytes) -> np.ndarray:
#     x = np.frombuffer(mu_bytes, dtype=np.uint8).astype(np.float32)
#     x = x / 255.0 * 2 - 1
#     mag = (1 / MU) * ((1 + MU) ** np.abs(x) - 1)
#     pcm = np.sign(x) * mag
#     return np.clip(pcm, -1.0, 1.0).astype(np.float32)

# def mulaw_encode(pcm_f32: np.ndarray) -> bytes:
#     x = np.clip(pcm_f32, -1.0, 1.0)
#     s = np.sign(x)
#     y = np.log1p(MU * np.abs(x)) / np.log1p(MU)
#     u = ((s * y) + 1) * 0.5 * 255.0
#     return u.astype(np.uint8).tobytes()

# def pcm16_from_f32(pcm_f32: np.ndarray) -> np.ndarray:
#     return np.clip(pcm_f32 * 32767.0, -32768, 32767).astype(np.int16)

# def f32_from_pcm16(pcm16: np.ndarray) -> np.ndarray:
#     return (pcm16.astype(np.float32) / 32768.0).astype(np.float32)

# def resample_to_16k(pcm_f32: np.ndarray, in_sr: int) -> np.ndarray:
#     if in_sr == 16000:
#         return pcm_f32
#     # up/downsample with poly filter
#     return resample_poly(pcm_f32, 16000, in_sr)

# def resample_to_8k(pcm_f32: np.ndarray, in_sr: int) -> np.ndarray:
#     if in_sr == 8000:
#         return pcm_f32
#     return resample_poly(pcm_f32, 8000, in_sr)
def pcm16_from_f32(pcm_f32: np.ndarray) -> np.ndarray:
    # float32 in -1.0..1.0 => int16
    return np.clip(pcm_f32 * 32767.0, -32768, 32767).astype(np.int16)

def f32_from_pcm16(pcm16: np.ndarray) -> np.ndarray:
    return (pcm16.astype(np.float32) / 32768.0).astype(np.float32)

def resample_to_16k(pcm_f32: np.ndarray, in_sr: int) -> np.ndarray:
    if in_sr == 16000:
        return pcm_f32
    # use polyphase resampling
    gcd = np.gcd(in_sr, 16000)
    up = 16000 // gcd
    down = in_sr // gcd
    return resample_poly(pcm_f32, up, down)

def resample_to_8k(pcm_f32: np.ndarray, in_sr: int) -> np.ndarray:
    if in_sr == 8000:
        return pcm_f32
    gcd = np.gcd(in_sr, 8000)
    up = 8000 // gcd
    down = in_sr // gcd
    return resample_poly(pcm_f32, up, down)