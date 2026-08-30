#!/bin/bash
# Wan2.1-VACE-1.3B via Diffusers, local 480p V2V editing.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"

MODEL="wan-vace-1.3b-v2v"
print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q "diffusers>=0.34.0" "transformers>=4.49.0" accelerate ftfy imageio imageio-ffmpeg "numpy>=1.23.5,<2" pillow opencv-python-headless "huggingface_hub[cli]"
deactivate

print_section "Checkpoint"
download_hf_checkpoint "Wan-AI/Wan2.1-VACE-1.3B-diffusers" "Wan2.1-VACE-1.3B-diffusers" "~6GB"
print_success "${MODEL} setup complete"
