#!/bin/bash
##############################################################################
# Install and test models (individual or all open-source)
#
# A model directory may ship a setup.sh (repo clones, multi-part checkpoints,
# custom pip steps) — when present it is used instead of the bare
# requirements.txt venv flow.
#
# Usage:
#   ./setup/install_model.sh --list
#   ./setup/install_model.sh --model wan-vace-14b-v2v
#   ./setup/install_model.sh --model wan-vace-14b-v2v --validate
#   ./setup/install_model.sh --opensource
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/share.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--model <name>|--opensource] [--validate]

Options:
  --model <name>       Model name (install single model)
  --opensource         Install all open-source models
  --list               List all available models
  --validate           Test model(s) after installation (runs the turntable
                       smoke task — needs GPU + downloaded weights)
  -h, --help           Show this help

Environment:
  V2V_WEIGHTS_DIR      Where checkpoints live (default: <repo>/weights).
                       Point at an existing download dir to reuse it.
USAGE
}

list_models() {
    print_header "Available Models"

    echo "OPEN-SOURCE MODELS (${#OPENSOURCE_MODELS[@]}) — install with this script:"
    echo ""
    for model in "${OPENSOURCE_MODELS[@]}"; do
        echo "  • ${model}"
    done

    echo ""
    echo "COMMERCIAL MODELS (${#COMMERCIAL_MODELS[@]}) — no install, just API keys in .env:"
    echo ""
    for model in "${COMMERCIAL_MODELS[@]}"; do
        local api_key
        api_key=$(get_commercial_env_var "$model")
        echo "  • ${model} (requires ${api_key})"
    done
    echo ""
}

MODEL=""
INSTALL_OPENSOURCE=false
VALIDATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --opensource)
            INSTALL_OPENSOURCE=true
            shift
            ;;
        --list)
            list_models
            exit 0
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" && "$INSTALL_OPENSOURCE" == "false" ]]; then
                MODEL="$1"
                shift
            else
                print_error "Unknown argument: $1"
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$MODEL" && "$INSTALL_OPENSOURCE" == "false" ]]; then
    usage
    exit 1
fi

MODELS_TO_INSTALL=()
if [[ "$INSTALL_OPENSOURCE" == "true" ]]; then
    print_header "Installing all open-source models"
    MODELS_TO_INSTALL=("${OPENSOURCE_MODELS[@]}")
else
    if is_commercial_model "$MODEL"; then
        print_info "${MODEL} is a commercial API model — nothing to install."
        print_info "Set $(get_commercial_env_var "$MODEL") in .env and run it directly."
        exit 0
    fi
    if ! is_opensource_model "$MODEL"; then
        print_error "Unknown model: ${MODEL}"
        exit 1
    fi
    MODELS_TO_INSTALL=("$MODEL")
fi

FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for model in "${MODELS_TO_INSTALL[@]}"; do
    print_header "Installing: ${model}"

    SETUP_SCRIPT="${SCRIPT_DIR}/models/${model}/setup.sh"
    REQUIREMENTS_FILE="${SCRIPT_DIR}/models/${model}/requirements.txt"

    if [[ -f "$SETUP_SCRIPT" ]]; then
        if bash "$SETUP_SCRIPT"; then
            print_success "${model} installed successfully"
            SUCCESSFUL_MODELS+=("${model}")
        else
            print_error "${model} installation failed"
            FAILED_MODELS+=("${model}")
        fi
    elif [[ -f "$REQUIREMENTS_FILE" ]]; then
        create_model_venv "$model"
        activate_model_venv "$model"
        if pip install -q -r "$REQUIREMENTS_FILE"; then
            deactivate
            print_success "${model} installed successfully"
            SUCCESSFUL_MODELS+=("${model}")
        else
            deactivate
            print_error "${model} installation failed"
            FAILED_MODELS+=("${model}")
        fi
    else
        print_error "No setup.sh or requirements.txt for ${model}"
        FAILED_MODELS+=("${model}")
    fi
done

if [[ "$VALIDATE" == "true" ]]; then
    print_header "Validation Phase"

    VALIDATION_FAILED=()
    VALIDATION_PASSED=()

    for model in "${SUCCESSFUL_MODELS[@]}"; do
        print_section "Validating: ${model}"
        if validate_model "$model"; then
            VALIDATION_PASSED+=("${model}")
        else
            VALIDATION_FAILED+=("${model}")
        fi
    done

    print_header "Validation Summary"
    if [[ ${#VALIDATION_PASSED[@]} -gt 0 ]]; then
        print_success "Passed (${#VALIDATION_PASSED[@]}):"
        for model in "${VALIDATION_PASSED[@]}"; do
            echo "      ✓ ${model}"
        done
    fi
    if [[ ${#VALIDATION_FAILED[@]} -gt 0 ]]; then
        echo ""
        print_error "Failed (${#VALIDATION_FAILED[@]}):"
        for model in "${VALIDATION_FAILED[@]}"; do
            echo "      ✗ ${model}"
        done
    fi
fi

print_header "Installation Summary"

if [[ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]]; then
    print_success "Installed (${#SUCCESSFUL_MODELS[@]}):"
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        echo "      ✓ ${model}"
    done
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    print_error "Failed (${#FAILED_MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "      ✗ ${model}"
    done
    exit 1
fi

print_header "✅ All installations completed successfully"
