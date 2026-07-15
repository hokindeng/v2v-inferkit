#!/bin/bash
# NVIDIA Cosmos3-Super video transfer. Needs 4+ GPUs / 128GB aggregate VRAM
# and ~150GB disk for the 132.7GB checkpoint (auto-downloaded to $HF_HOME on
# first inference; gated — accept the nvidia/Cosmos3-Super license and set
# HF_TOKEN). The cosmos-framework repo manages its own uv venv; the kit venv
# here is a thin shim so the runner's venv-detection dispatches the wrapper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
load_env_file

MODEL="cosmos3-super-v2v"

print_section "System packages (guardrail OpenCV deps + media)"
sudo apt-get install -y -q --no-install-recommends curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev libxcb1

print_section "Repo"
clone_model_repo "https://github.com/NVIDIA/cosmos-framework.git" "cosmos-framework"

print_section "uv + framework venv (prebuilt wheels: torch 2.10.0+cu128, natten, flash-attn)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
cd "${REPOS_DIR}/cosmos-framework"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group=cu128-train
cd "${V2V_ROOT}"

print_section "Kit shim venv (runner dispatch)"
create_model_venv "$MODEL"

print_info "Checkpoint (132.7GB) auto-downloads to \$HF_HOME on first run."
print_info "Prefetch: ${REPOS_DIR}/cosmos-framework/.venv/bin/python -m cosmos_framework.scripts.prefetch_hf_checkpoints"
print_warning "Point HF_HOME at a volume with >=150GB free before first inference."
print_warning "Cosmos3-Super needs 4x80GB GPUs — it will NOT run on a single-GPU box."
print_success "${MODEL} setup complete"
