# llm.py
import os
from llama_cpp import Llama

class LLMEngine:
    """
    GPU-enabled Llama wrapper using llama-cpp-python. Keeps same reply(prompt, system) -> str contract.
    Set LLAMA_MODEL env var to the GGUF path.
    """
    def __init__(self, ctx_size=2048, n_gpu_layers=None):
        model_path = os.getenv("LLAMA_MODEL", "")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("Set LLAMA_MODEL to a valid GGUF path (TinyLlama/Phi)")
        # If not provided, choose a reasonable default for RTX 4000 Ada
        # We'll default to 20 GPU layers — adjust if you see OOM.
        if n_gpu_layers is None:
            n_gpu_layers = 20
        # Use deterministic low-temperature settings to reduce nonsense
        self.llm = Llama(model_path=model_path, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers, verbose=False)

    def reply(self, prompt: str, system: str = "You are Friday, a concise IVR assistant. Keep replies short, clear, and natural."):
        full = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        # Use low temperature and constraints to avoid hallucination; adjust max_tokens as needed.
        out = self.llm(full, max_tokens=128, temperature=0.0, top_p=0.95, stop=["<|user|>", "</s>"])
        text = out["choices"][0]["text"].strip()
        return text
