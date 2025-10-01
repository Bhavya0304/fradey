#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/workspace"
VENV_DIR="$WORKDIR/venv"
PY=python3

echo "== runpod_setup: starting in $WORKDIR =="

# basic apt packages
apt-get update
apt-get install -y git build-essential cmake pkg-config libsndfile1-dev libasound2-dev \
    libssl-dev libffi-dev python3-dev python3-venv wget unzip

# quick GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi present:"
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. Ensure container has GPU access. Continuing..."
fi

# Create venv
$PY -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Install general Python deps
pip install numpy soundfile aiohttp

# Install faster-whisper (python wrapper)
pip install faster-whisper

# Install onnxruntime-gpu if you plan to use onnx TTS or piper runtime expecting ONNX GPU
pip install --upgrade "onnxruntime-gpu"

# Install Coqui TTS (GPU-capable). This may pull large deps (torch)
# Pick torch build that matches CUDA available. We try to auto-detect CUDA version.
CUDA_VERSION=""
if command -v nvcc >/dev/null 2>&1; then
  CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)
  echo "Detected CUDA version: $CUDA_VERSION"
fi

# Install torch with CUDA support if possible. Default to latest stable with CUDA 11.8 fallback.
# Using pip triple-quoted indexes may be necessary for specific cuda versions. We'll attempt generic install,
# and if CPU-only torch is installed, user can choose to install specific wheel manually.
pip install "torch" "torchaudio" --extra-index-url https://download.pytorch.org/whl/cu118 || true

# Install Coqui TTS (this will pull a lot; it needs torch)
pip install TTS

# Install soundfile already done earlier.

# Install llama-cpp-python with CUDA flags (force build if needed)
# This will attempt to build a CUDA-enabled llama-cpp-python if CUDA toolchain present.
echo "Installing llama-cpp-python (CUDA build if possible)..."
# Force a CMake build that tries to enable GGML_CUDA / CUBLAS if available
export FORCE_CMAKE=1
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DGGML_CUDA=on"
pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python || {
  echo "llama-cpp-python pip install failed; retrying without cuda flags..."
  unset CMAKE_ARGS
  unset FORCE_CMAKE
  pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python
}

# Install onnx tools and others used by piper if needed
pip install onnx onnxruntime

# ctranslate2 GPU wheel: try to install if available for your platform; fallback to CPU build
echo "Attempting to install ctranslate2 with cuda support..."
pip install --upgrade ctranslate2 || echo "ctranslate2 install failed or CPU-only wheel installed."

# Clean up and show versions
python - <<'PY'
import sys, importlib, pkgutil
print("Python:", sys.version.splitlines()[0])
for pkg in ("torch","faster_whisper","llama_cpp","TTS","onnxruntime"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: OK, version:", getattr(m, '__version__', None))
    except Exception as e:
        print(f"{pkg}: NOT INSTALLED or failed to import: {e}")
PY

echo "== Setup done. Activate venv with: source $VENV_DIR/bin/activate =="
echo "Place your GGUF model and set LLAMA_MODEL env var, or set PIPER_BIN if you have a GPU Piper binary."


MODEL_DIR="$WORKDIR/models"
mkdir -p "$MODEL_DIR"
MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
MODEL_FILE="$MODEL_DIR/Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
  echo "Downloading Llama-3.1-8B-Instruct-IQ2_M.gguf model (~4.5GB)..."
  wget -c --progress=bar:force:noscroll -O "$MODEL_FILE" "$MODEL_URL"
else
  echo "Model already exists at $MODEL_FILE"
fi

# Export LLAMA_MODEL into bashrc for convenience
if ! grep -q "LLAMA_MODEL=" ~/.bashrc; then
  echo "export LLAMA_MODEL=$MODEL_FILE" >> ~/.bashrc
  echo "Added LLAMA_MODEL env var to ~/.bashrc"
fi

echo "== Setup done. Activate venv with: source $VENV_DIR/bin/activate =="
echo "Llama model ready at $MODEL_FILE"
