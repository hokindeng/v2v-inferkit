#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="bernini-r-1.3b-v2v"
clone_model_repo "https://github.com/bytedance/Bernini.git" "Bernini"
create_model_venv "$MODEL" "${BERNINI_PYTHON:-python3.11}"
activate_model_venv "$MODEL"
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -q -e "${REPOS_DIR}/Bernini" "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "ByteDance/Bernini-R-1.3B-Diffusers" "Bernini-R-1.3B-Diffusers"
print_success "${MODEL} setup complete"
