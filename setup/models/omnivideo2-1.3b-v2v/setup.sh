#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="omnivideo2-1.3b-v2v"
clone_model_repo "https://github.com/SAIS-FUXI/Omni-Video.git" "Omni-Video"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q -r <(grep -v '^flash_attn==' "${REPOS_DIR}/Omni-Video/requirements.txt")
pip install -q flash-attn==2.8.3 --no-build-isolation
pip install -q "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "Fudan-FUXI/OmniVideo2-1.3B" "OmniVideo2-1.3B"
download_hf_checkpoint "Qwen/Qwen3-VL-30B-A3B-Instruct" "Qwen3-VL-30B-A3B-Instruct" "required VLM"
print_warning "OmniVideo2 upstream references a LICENSE file that is not currently published; treat use rights as unspecified."
print_success "${MODEL} setup complete"
