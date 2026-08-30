#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="sama-14b-v2v"
clone_model_repo "https://github.com/Cynthiazxy123/SAMA.git" "SAMA"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q -r <(grep -v '^flash_attn==' "${REPOS_DIR}/SAMA/requirements.txt") \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q flash-attn==2.5.8 --no-build-isolation
pip install -q "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "Wan-AI/Wan2.1-T2V-14B" "Wan2.1-T2V-14B" "base model"
download_hf_checkpoint "syxbb/SAMA-14B" "SAMA-14B"
print_success "${MODEL} setup complete"
