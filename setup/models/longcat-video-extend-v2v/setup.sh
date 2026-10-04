#!/bin/bash
# LongCat-Video (13.6B, MIT) video continuation: official repo + pinned deps + weights.
# The 720p refinement LoRA is skipped (480p distill path only); the rest is ~80 GB.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="longcat-video-extend-v2v"
REPO="${REPOS_DIR}/LongCat-Video"
clone_model_repo "https://github.com/meituan-longcat/LongCat-Video.git" "LongCat-Video" "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# upstream requirements minus flash-attn (prebuilt wheel below) and streamlit (demo UI only)
pip install -q -r <(grep -v -E '^(flash-attn|streamlit|torch)==' "${REPO}/requirements.txt")
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# transformers 4.41 refuses huggingface_hub >= 1.0 at import time
pip install -q regex accelerate safetensors python-dotenv "huggingface_hub[cli]>=0.34,<1.0"
deactivate
DEST="${WEIGHTS_DIR}/LongCat-Video"
print_download "Downloading meituan-longcat/LongCat-Video (~80 GB, refinement LoRA skipped)"
mkdir -p "$DEST"
"${ENVS_DIR}/${MODEL}/bin/hf" download meituan-longcat/LongCat-Video --local-dir "$DEST" \
    --exclude "lora/refinement_lora.safetensors" "assets/*"
print_success "Checkpoint ready: LongCat-Video"
print_success "${MODEL} setup complete"
