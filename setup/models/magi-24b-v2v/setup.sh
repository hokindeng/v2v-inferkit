#!/bin/bash
# MAGI-1 24B base — video CONTINUATION (prefix v2v). Needs 4x80GB+ GPUs.
# Uses MAGI-1 (2025) base checkpoint: the MAGI-1.1 base checkpoint is
# currently incompatible with the public inference code (issue #123).
# Docker (sandai/magi:latest) sidesteps the flash-attn build; this venv path
# follows the repo's conda recipe with pip.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
load_env_file

MODEL="magi-24b-v2v"
MODEL_DIR="${WEIGHTS_DIR}/MAGI-1"

print_section "Repo"
clone_model_repo "https://github.com/SandAI-org/MAGI-1.git" "MAGI-1"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
# flash-attn 2.4.2 is an old pin that often fails to build on new stacks
# (issues #65/#118/#92). Try the wheel; on failure fall back to source build
# with a warning. MagiAttention is Hopper-only — NOT needed on A100.
pip install -q ninja  # without ninja the source build compiles SERIALLY (hours)
if ! MAX_JOBS="$(nproc)" pip install -q flash-attn==2.4.2 --no-build-isolation; then
    print_warning "flash-attn 2.4.2 install failed — MAGI requires it; consider the sandai/magi docker image instead"
    deactivate
    exit 1
fi
pip install -q flashinfer-python==0.2.0.post2 -i https://flashinfer.ai/whl/cu124/torch2.4/
pip install -q transformers==4.42.3 diffusers==0.29.2 accelerate==0.32.1 \
    "numpy==1.26.4" ffmpeg-python easydict omegaconf "huggingface_hub[cli]"

deactivate

print_section "Checkpoints (t5 + vae + 24B_base, ~55GB total)"
if [[ -d "${MODEL_DIR}/ckpt/magi/24B_base" ]] && [[ -n "$(ls -A "${MODEL_DIR}/ckpt/magi/24B_base" 2>/dev/null)" ]]; then
    print_skip "MAGI-1 24B_base exists"
else
    activate_model_venv "$MODEL"
    hf download sand-ai/MAGI-1 --include "ckpt/t5/*" "ckpt/vae/*" "ckpt/magi/24B_base/*" \
        --local-dir "${MODEL_DIR}"
    deactivate
fi

print_warning "MAGI needs 4x80GB GPUs (A100/H100) — it will NOT run on this box if it is a single-GPU instance."
print_success "${MODEL} setup complete"
