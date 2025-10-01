import os
from llama_cpp import Llama

class LLMEngine:
    """
    Tiny local model for IVR. Use a small GGUF (TinyLlama 1.1B chat or Phi-2 gguf).
    Set env LLAMA_MODEL=/path/to/model.gguf
    """
    def __init__(self, ctx_size=2048, n_gpu_layers=0):
        model_path = os.getenv("LLAMA_MODEL", "")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("Set LLAMA_MODEL to a valid GGUF path (TinyLlama/Phi)")
        self.llm = Llama(model_path=model_path, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers, verbose=False)

    def reply(self, prompt: str, system: str = "You are Friday, a concise IVR assistant. Keep replies short, clear, and natural."):
        full = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        out = self.llm(full, max_tokens=128, temperature=0.2, stop=["<|user|>", "</s>"])
        text = out["choices"][0]["text"].strip()
        return text
