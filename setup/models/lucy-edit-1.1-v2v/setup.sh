#!/bin/bash
# Lucy Edit Dev weights are licensed for non-commercial use only.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="lucy-edit-1.1-v2v"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -q "git+https://github.com/huggingface/diffusers.git" transformers accelerate imageio imageio-ffmpeg pillow "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "decart-ai/Lucy-Edit-1.1-Dev" "Lucy-Edit-1.1-Dev" "non-commercial"
print_warning "Lucy-Edit-1.1-Dev is governed by the Lucy non-commercial model license."
print_success "${MODEL} setup complete"
