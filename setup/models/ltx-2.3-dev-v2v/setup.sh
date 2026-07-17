#!/bin/bash
# LTX-2.3 IC-LoRA video conditioning. Official v2v is DISTILLED-only
# (docs/pipelines.md) — pending team ruling; both distilled-1.1 and dev
# checkpoints are downloaded so either can be pointed at via
# LTX2_IC_LORA_CHECKPOINT. Gemma text encoder is GATED — accept the license
# for google/gemma-3-12b-it-qat-q4_0-unquantized and set HF_TOKEN.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"
load_env_file

MODEL="ltx-2.3-dev-v2v"
MODEL_DIR="${WEIGHTS_DIR}/LTX-2.3"

print_section "Repo"
clone_model_repo "https://github.com/Lightricks/LTX-2.git" "LTX-2"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
# Pin the full torch triplet on ONE CUDA index — torchvision/torchaudio left
# unpinned resolve from the default index and can land CUDA-13 builds beside
# cu126 torch (libcudart.so.13 crash, hit live 2026-07-17).
pip install -q "torch~=2.7.0" "torchvision~=0.22.0" "torchaudio~=2.7.0" \
    --index-url https://download.pytorch.org/whl/cu126
# install the pipelines package editable from the monorepo (issue #216: plain
# pip missed the multigpu module — current main is fixed, keep -e)
pip install -q -e "${REPOS_DIR}/LTX-2/packages/ltx-core"
pip install -q -e "${REPOS_DIR}/LTX-2/packages/ltx-pipelines"
pip install -q "huggingface_hub[cli]" sentencepiece

deactivate

print_section "Checkpoints"
mkdir -p "${MODEL_DIR}"
activate_model_venv "$MODEL"
for f in \
    "ltx-2.3-22b-distilled-1.1.safetensors" \
    "ltx-2.3-22b-dev.safetensors" \
    "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" \
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"; do
    if [[ -f "${MODEL_DIR}/$f" ]]; then
        print_skip "$f exists"
    else
        print_download "$f"
        hf download Lightricks/LTX-2.3 "$f" --local-dir "${MODEL_DIR}"
    fi
done

if [[ -d "${MODEL_DIR}/gemma-3-12b" ]] && [[ -n "$(ls -A "${MODEL_DIR}/gemma-3-12b" 2>/dev/null)" ]]; then
    print_skip "gemma-3-12b exists"
else
    print_download "Gemma 3 12B text encoder (GATED — needs HF_TOKEN + accepted license)"
    if ! hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir "${MODEL_DIR}/gemma-3-12b"; then
        print_warning "Gemma download failed — accept the license at"
        print_warning "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized and re-run."
        deactivate
        exit 1
    fi
fi
deactivate

print_success "${MODEL} setup complete"
