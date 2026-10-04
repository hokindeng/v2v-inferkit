#!/bin/bash
# Self-Forcing (Wan2.1-T2V-1.3B, DMD, 4-step autoregressive) used as a video continuation model.
# Code: github.com/guandeh17/Self-Forcing (Apache-2.0). Weights: gdhe17/Self-Forcing (Apache-2.0) —
# only checkpoints/self_forcing_dmd.pt is fetched (the HF repo holds ~45 GB of other checkpoints) —
# plus the Wan2.1-T2V-1.3B base (text encoder, VAE, model config; Apache-2.0).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="self-forcing-extend-v2v"
REPO="Self-Forcing"
clone_model_repo "https://github.com/guandeh17/Self-Forcing.git" "$REPO" "33593df3e81fa3ec10239271dd2c100facac6de1"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
# Minimal subset of upstream requirements.txt for the inference path (no TensorRT/ONNX/GUI/training deps).
# flash-attn is optional upstream (falls back to torch SDPA), so it is not installed.
pip install -q "diffusers==0.31.0" "transformers>=4.49,<4.57" "tokenizers>=0.20.3" "accelerate>=1.1.1" \
    "numpy<2" omegaconf einops easydict ftfy regex tqdm imageio imageio-ffmpeg pillow \
    python-dotenv "huggingface_hub[cli]"
HF_BIN="$(get_model_venv_path "$MODEL")/bin/hf"
deactivate
download_hf_checkpoint "Wan-AI/Wan2.1-T2V-1.3B" "Wan2.1-T2V-1.3B" "~17 GB"
SF_DIR="${WEIGHTS_DIR}/Self-Forcing"
if [[ -f "${SF_DIR}/checkpoints/self_forcing_dmd.pt" ]]; then
    print_skip "Checkpoint exists: Self-Forcing/checkpoints/self_forcing_dmd.pt"
else
    print_download "Downloading gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt (~5.7 GB)"
    mkdir -p "$SF_DIR"
    "$HF_BIN" download "gdhe17/Self-Forcing" "checkpoints/self_forcing_dmd.pt" --local-dir "$SF_DIR"
fi
# Upstream code loads the base model from the relative path wan_models/Wan2.1-T2V-1.3B.
mkdir -p "${REPOS_DIR}/${REPO}/wan_models"
ln -sfn "${WEIGHTS_DIR}/Wan2.1-T2V-1.3B" "${REPOS_DIR}/${REPO}/wan_models/Wan2.1-T2V-1.3B"
print_success "${MODEL} setup complete"
