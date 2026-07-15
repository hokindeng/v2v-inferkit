#!/bin/bash
# Wan2.1-VACE-14B via diffusers WanVACEPipeline — single 48GB GPU, 480p.
# NOTE: needs the *-diffusers checkpoint; the raw Wan-AI/Wan2.1-VACE-14B
# checkpoint has no model_index.json and cannot be loaded by diffusers.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"

MODEL="wan-vace-14b-v2v"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# WanVACEPipeline added in diffusers 0.34.0; flash-attn not needed (SDPA fallback)
pip install -q "diffusers>=0.34.0" "transformers>=4.49.0" accelerate ftfy \
    imageio imageio-ffmpeg "numpy>=1.23.5,<2" pillow opencv-python-headless \
    "huggingface_hub[cli]"

deactivate

print_section "Checkpoints"
download_hf_checkpoint "Wan-AI/Wan2.1-VACE-14B-diffusers" "Wan2.1-VACE-14B-diffusers" "~32GB"

print_success "${MODEL} setup complete"
