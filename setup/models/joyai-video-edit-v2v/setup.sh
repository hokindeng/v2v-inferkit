#!/bin/bash
# JoyAI's fused CUDA op requires nvcc >=12.8. Set JOYAI_NO_FP8=1 to build
# the light BF16-only variant; the installer records that choice for the wrapper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"

MODEL="joyai-video-edit-v2v"
clone_model_repo "https://github.com/jd-opensource/JoyAI-Video-Edit.git" "JoyAI-Video-Edit"
clone_model_repo "https://github.com/NVIDIA/cutlass.git" "cutlass" "dcf215af"

create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q -r "${REPOS_DIR}/JoyAI-Video-Edit/deploy/requirements.txt"
pip install -q websocket-client "huggingface_hub[cli]"

if ! command -v nvcc >/dev/null 2>&1; then
    print_error "JoyAI joyomni_ops requires nvcc >=12.8. Install a CUDA toolkit in this environment."
    deactivate
    exit 1
fi

if [[ "${JOYAI_NO_FP8:-0}" == "1" ]]; then
    JOYOMNI_OPS_NO_FP8=1 pip install -q --no-build-isolation "${REPOS_DIR}/JoyAI-Video-Edit/deploy/joyomni_ops"
    touch "${ENVS_DIR}/${MODEL}/.joyai_no_fp8"
else
    JOYOMNI_ARCH="$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}{cc[1]}a' if cc[0] >= 10 else f'{cc[0]}{cc[1]}')")"
    JOYOMNI_OPS_CUDA_ARCHS="$JOYOMNI_ARCH" \
    JOYOMNI_OPS_CUTLASS_DIR="${REPOS_DIR}/cutlass" \
        pip install -q --no-build-isolation "${REPOS_DIR}/JoyAI-Video-Edit/deploy/joyomni_ops"
fi
deactivate

download_hf_checkpoint "jdopensource/JoyAI-Video-Edit" "JoyAI-Video-Edit" "DiT + xVAE"
download_hf_checkpoint "XiaomiMiMo/MiMo-VL-7B-RL-2508" "MiMo-VL-7B-RL-2508" "required encoder"

if [[ "${JOYAI_NO_FP8:-0}" == "1" ]]; then
    print_warning "BF16-only build recorded; JoyAI wrapper will disable both FP8 paths."
fi
print_success "${MODEL} setup complete"
