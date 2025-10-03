# llm.py
import os
from llama_cpp import Llama
from groq import Groq

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

    def reply(self, prompt: str, system: str = "You are Friday, Bhavys personal assistent and call handler so someone will call bhavya and you will answer bhavya is sleeping right now if someone asks and ask him to record their messege also if they ask anything about bhavya's profile asnwer him considering he has some achivements like mark zukerberg."):
        full = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        # Use low temperature and constraints to avoid hallucination; adjust max_tokens as needed.
        out = self.llm(full, max_tokens=128, temperature=0.0, top_p=0.95, stop=["<|user|>", "</s>"])
        text = out["choices"][0]["text"].strip()
        return text
    
class LLMEngineGroq:
    """
    GPU-enabled Llama wrapper using llama-cpp-python. Keeps same reply(prompt, system) -> str contract.
    Set LLAMA_MODEL env var to the GGUF path.
    """
    def __init__(self):

        api_key = os.getenv("GROQ_API")
        self.groq = Groq(api_key=api_key)

    def reply(self, prompt: str, system: str = "You are Friday, Bhavys personal assistent and call handler so someone will call bhavya and you will answer bhavya is sleeping right now if someone asks and ask him to record their messege also if they ask anything about bhavya's profile asnwer him considering he has some achivements like mark zukerberg. Also keep convertstaion very small and to the point"):
        chat_completion = self.groq.chat.completions.create(
    messages=[
        # Set an optional system message. This sets the behavior of the
        # assistant and can be used to provide specific instructions for
        # how it should behave throughout the conversation.
        {
            "role": "system",
            "content": system
        },
        # Set a user message for the assistant to respond to.
        {
            "role": "user",
            "content": prompt,
        }
    ],

    # The language model which will generate the completion.
    model="llama-3.3-70b-versatile"
)
        text = chat_completion.choices[0].message.content
        return text
