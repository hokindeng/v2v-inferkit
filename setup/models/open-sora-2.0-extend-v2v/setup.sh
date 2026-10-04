#!/bin/bash
# Open-Sora 2.0 (11B) video extension. Code and weights: Apache-2.0.
# Only the video model, the Hunyuan VAE and the two text encoders are downloaded
# (~45 GB); flux1-dev*.safetensors (~24 GB) is only used by the text->image->video
# pipeline and is skipped.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="open-sora-2.0-extend-v2v"
clone_model_repo "https://github.com/hpcaitech/Open-Sora.git" "Open-Sora" "7ad6a96a135feb81f755c84fb391818718f6beb2"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
# Upstream pins torch 2.4.0. diffusers 0.29 (pinned by colossalai) still imports
# huggingface_hub.cached_download, which was removed in hub 0.26 -> pin hub 0.25.2.
CONSTRAINTS="$(mktemp)"
printf 'torch==2.4.0\ntorchvision==0.19.0\nhuggingface_hub==0.25.2\n' > "$CONSTRAINTS"
pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -q xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
# Prebuilt flash-attn wheel (torch 2.4 / CUDA 12 / cp310); no source build.
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install -q -c "$CONSTRAINTS" "colossalai==0.4.9" -r "${REPOS_DIR}/Open-Sora/requirements.txt"
pip install -q -c "$CONSTRAINTS" --no-deps -e "${REPOS_DIR}/Open-Sora"
pip install -q -c "$CONSTRAINTS" "huggingface_hub[cli]==0.25.2" python-dotenv
rm -f "$CONSTRAINTS"
python -c "import torch, flash_attn, liger_kernel, colossalai, opensora; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
DEST="${WEIGHTS_DIR}/Open-Sora-v2"
if [[ -f "${DEST}/Open_Sora_v2.safetensors" && -f "${DEST}/hunyuan_vae.safetensors" ]]; then
    print_skip "Checkpoint exists: Open-Sora-v2"
else
    print_download "Downloading hpcai-tech/Open-Sora-v2 (~45 GB, without flux1-dev)"
    mkdir -p "$DEST"
    huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir "$DEST" --exclude "flux1-dev*"
fi
deactivate
print_success "${MODEL} setup complete"
