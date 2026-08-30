#!/bin/bash
# Ditto/Editto weights are CC-BY-NC-SA-4.0 and are not for commercial use.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="editto-v2v"
clone_model_repo "https://github.com/EzioBy/Ditto.git" "Ditto"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q -e "${REPOS_DIR}/Ditto" "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "Wan-AI/Wan2.1-VACE-14B" "Wan2.1-VACE-14B" "base model"
download_hf_checkpoint "QingyanBai/Ditto_models" "Ditto_models" "CC-BY-NC-SA-4.0"
mkdir -p "${REPOS_DIR}/Ditto/models/Wan-AI"
ln -sfn "${WEIGHTS_DIR}/Wan2.1-VACE-14B" "${REPOS_DIR}/Ditto/models/Wan-AI/Wan2.1-VACE-14B"
print_warning "Ditto/Editto code and weights are non-commercial (CC-BY-NC-SA-4.0)."
print_success "${MODEL} setup complete"
