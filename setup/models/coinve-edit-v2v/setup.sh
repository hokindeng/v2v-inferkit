#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="coinve-edit-v2v"
clone_model_repo "https://github.com/coinve200k/CoinVE-200K.git" "CoinVE-200K"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install -q -e "${REPOS_DIR}/CoinVE-200K/CoinVE-Edit" "qwen-vl-utils" "huggingface_hub[cli]"
pip install -q flash-attn==2.8.3 --no-build-isolation
deactivate
download_hf_checkpoint "Wan-AI/Wan2.1-T2V-14B" "Wan-AI/Wan2.1-T2V-14B" "base model"
download_hf_checkpoint "Qwen/Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Instruct" "required MLLM"
download_hf_checkpoint "FireCRT/CoinVE-Edit" "CoinVE-Edit"
print_success "${MODEL} setup complete"
