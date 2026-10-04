#!/bin/bash
# UniVideo (KlingTeam, Apache-2.0): Qwen2.5-VL-7B encoder + HunyuanVideo MMDiT.
# Uses the "hidden" checkpoint variant (the one the upstream README runs for every task).
# The UniVideo checkpoint holds the full MMDiT, so only the HunyuanVideo transformer
# config, VAE and scheduler are downloaded (not its 25 GB transformer or text encoders).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="univideo-v2v"
UPSTREAM_REF="893c9da04b87fdf5099e8e0e9ef164bca6753327"

clone_model_repo "https://github.com/KlingTeam/UniVideo.git" "UniVideo" "$UPSTREAM_REF"

create_model_venv "$MODEL"
activate_model_venv "$MODEL"
# Upstream is tested with torch 2.4.1 + CUDA 12.1, diffusers 0.34.0, transformers 4.51.3.
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# python-dotenv + httpx: run.py and its output probe when run in-process inside this venv.
pip install -q -r "${REPOS_DIR}/UniVideo/requirements.txt" pyyaml python-dotenv httpx
HF_BIN="$(get_model_venv_path "$MODEL")/bin/hf"
deactivate

print_download "UniVideo hidden variant (~26 GB)"
"$HF_BIN" download KlingTeam/UniVideo --include "univideo_qwen2p5vl7b_hidden_hunyuanvideo/*" \
    --local-dir "${WEIGHTS_DIR}/UniVideo"
print_download "HunyuanVideo VAE + scheduler + transformer config (~0.5 GB)"
"$HF_BIN" download hunyuanvideo-community/HunyuanVideo \
    --include "vae/*" "scheduler/*" "transformer/config.json" "model_index.json" \
    --local-dir "${WEIGHTS_DIR}/HunyuanVideo"
print_download "Qwen2.5-VL-7B-Instruct (~17 GB)"
"$HF_BIN" download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "${WEIGHTS_DIR}/Qwen2.5-VL-7B-Instruct"

print_success "${MODEL} setup complete"
