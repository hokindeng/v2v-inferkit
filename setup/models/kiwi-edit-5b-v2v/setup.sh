#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="kiwi-edit-5b-v2v"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q diffusers decord einops accelerate transformers==4.57.0 opencv-python-headless av imageio imageio-ffmpeg "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "linyq/kiwi-edit-5b-instruct-reference-diffusers" "kiwi-edit-5b-instruct-reference-diffusers"
print_warning "Kiwi code is MIT; the selected checkpoint does not publish an explicit model license."
print_success "${MODEL} setup complete"
