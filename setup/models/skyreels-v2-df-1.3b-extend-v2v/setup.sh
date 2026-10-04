#!/bin/bash
# SkyReels-V2 Diffusion Forcing 1.3B (540P) used as a video continuation model through the
# diffusers SkyReelsV2DiffusionForcingVideoToVideoPipeline (same algorithm as the official
# generate_video_df.py --video_path path). Weights: Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers
# (~29 GB; Skywork Community License).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
MODEL="skyreels-v2-df-1.3b-extend-v2v"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q "diffusers>=0.35.1" "transformers>=4.49,<5" accelerate sentencepiece ftfy regex \
    imageio imageio-ffmpeg pillow "numpy<2" python-dotenv "huggingface_hub[cli]"
deactivate
download_hf_checkpoint "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers" "SkyReels-V2-DF-1.3B-540P-Diffusers" "~29 GB"
print_warning "SkyReels-V2 weights are governed by the Skywork Community License."
print_success "${MODEL} setup complete"
