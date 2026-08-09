#!/bin/bash
# Tencent HY-OmniWeaving --task editing — single GPU with offloading, 480p.
# The model dir needs the HF snapshot PLUS four extra components (see
# download-checkpoint.md in the repo). FLUX.1-Redux siglip is GATED — accept
# its license on huggingface.co and set HF_TOKEN before running.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
load_env_file

MODEL="hy-omniweaving-v2v"
MODEL_DIR="${WEIGHTS_DIR}/HY-OmniWeaving"

print_section "Repo"
clone_model_repo "https://github.com/Tencent-Hunyuan/OmniWeaving.git" "OmniWeaving"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q -r "${REPOS_DIR}/OmniWeaving/requirements.txt"
pip install -q "huggingface_hub[cli]" modelscope
# flash-attn optional (SDPA fallback); skip the source build by default.

deactivate

print_section "Checkpoints (main snapshot + 4 extra components)"
download_hf_checkpoint "tencent/HY-OmniWeaving" "HY-OmniWeaving" "~55GB"

activate_model_venv "$MODEL"
if [[ ! -d "${MODEL_DIR}/text_encoder/llm" ]] || [[ -z "$(ls -A "${MODEL_DIR}/text_encoder/llm" 2>/dev/null)" ]]; then
    print_download "Qwen2.5-VL-7B-Instruct (~17GB)"
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "${MODEL_DIR}/text_encoder/llm"
else
    print_skip "Qwen2.5-VL-7B-Instruct exists"
fi
if [[ ! -d "${MODEL_DIR}/text_encoder/byt5-small" ]] || [[ -z "$(ls -A "${MODEL_DIR}/text_encoder/byt5-small" 2>/dev/null)" ]]; then
    print_download "byt5-small"
    hf download google/byt5-small --local-dir "${MODEL_DIR}/text_encoder/byt5-small"
else
    print_skip "byt5-small exists"
fi
if [[ ! -d "${MODEL_DIR}/text_encoder/Glyph-SDXL-v2" ]] || [[ -z "$(ls -A "${MODEL_DIR}/text_encoder/Glyph-SDXL-v2" 2>/dev/null)" ]]; then
    print_download "Glyph-SDXL-v2 (ModelScope)"
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir "${MODEL_DIR}/text_encoder/Glyph-SDXL-v2"
else
    print_skip "Glyph-SDXL-v2 exists"
fi
# The pipeline loads SiglipVisionModel.from_pretrained(siglip_path,
# subfolder='image_encoder') — so the FLUX repo's directory layout must be
# preserved under vision_encoder/siglip/. Download only the needed subfolders
# (separate --include flags; two patterns after one flag get misparsed as
# positional filenames). Check for the actual model file, not a non-empty
# dir: a pre-license 403 leaves a husk (gated repos expose README/LICENSE
# publicly) that fools an ls -A check.
if [[ ! -f "${MODEL_DIR}/vision_encoder/siglip/image_encoder/model.safetensors" ]]; then
    print_download "FLUX.1-Redux siglip (GATED — needs HF_TOKEN + accepted/approved license)"
    rm -rf "${MODEL_DIR}/vision_encoder/siglip"
    if ! hf download black-forest-labs/FLUX.1-Redux-dev \
        --include "image_encoder/*" --include "feature_extractor/*" \
        --local-dir "${MODEL_DIR}/vision_encoder/siglip"; then
        print_warning "siglip download failed — gated access not granted yet. Request access at"
        print_warning "https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev, wait for approval"
        print_warning "(check https://huggingface.co/settings/gated-repos), then re-run."
        deactivate
        exit 1
    fi
else
    print_skip "siglip exists"
fi
deactivate

print_section "Checkpoint sanity check (issue #3: early snapshot had zeroed mm_in weights)"
print_info "After setup, verify: python -c \"from safetensors import safe_open; import glob;\\
f=glob.glob('${MODEL_DIR}/transformer/*.safetensors')[0]; print('spot-check mm_in in', f)\""

print_success "${MODEL} setup complete"
