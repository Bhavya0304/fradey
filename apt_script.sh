set -euo pipefail
WORKDIR="/workspace/fradey"
VENV_DIR="$WORKDIR/venv"
PY=python3

echo "== runpod_setup: starting in $WORKDIR =="

apt update && sudo apt upgrade -y

apt-get install -y git build-essential cmake pkg-config libsndfile1-dev libasound2-dev \
    libssl-dev libffi-dev python3-dev python3-venv wget unzip
apt-get install portaudio19-dev

apt-get install espeak