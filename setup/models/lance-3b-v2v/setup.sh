#!/bin/bash
# Lance 3B (code + weights Apache-2.0). Video editing needs only Lance_3B_Video, the Qwen2.5-VL ViT
# and the Wan2.2 VAE (~18 GB of the 31 GB repo).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="lance-3b-v2v"
LANCE_REF="4baeee086648996f6ab12e673cbe461b0b149997"
clone_model_repo "https://github.com/bytedance/Lance.git" "Lance" "$LANCE_REF"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
# Upstream-tested stack: torch 2.8.0 + cu126 + flash-attn 2.8.3 (prebuilt wheel, no source build).
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -q -r <(grep -vE '^(gradio|gradio-client|spaces|wandb|sk-video|nvidia-ml-py)==' "${REPOS_DIR}/Lance/requirements.txt")
PYTAG="cp$(python -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
pip install -q --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-${PYTAG}-${PYTAG}-linux_x86_64.whl"
pip install -q python-dotenv
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="bytedance-research/Lance", local_dir="${WEIGHTS_DIR}/Lance",
                  allow_patterns=["Lance_3B_Video/*", "Qwen2.5-VL-ViT/*", "Wan2.2_VAE.pth"])
PY
deactivate
print_success "${MODEL} setup complete"
