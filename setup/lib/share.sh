#!/bin/bash
##############################################################################
# v2v-inferkit Setup - Shared Library
# Ported from VBVR-InferKit setup/lib/share.sh, adapted for the v2v model set.
##############################################################################

set -euo pipefail

# Project root - dynamically determine from script location
SHARE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V2V_ROOT="$(cd "${SHARE_LIB_DIR}/../.." && pwd)"
export ENVS_DIR="${V2V_ROOT}/envs"
export REPOS_DIR="${V2V_ROOT}/repos"
export LOGS_DIR="${V2V_ROOT}/logs"
export TESTS_DIR="${V2V_ROOT}/examples"
# Weights live under the repo by default; point V2V_WEIGHTS_DIR at an existing
# download location (e.g. ~/models) to reuse checkpoints already on disk.
export WEIGHTS_DIR="${V2V_WEIGHTS_DIR:-${V2V_ROOT}/weights}"

# ============================================================================
# MODEL REGISTRY
# ============================================================================

declare -a OPENSOURCE_MODELS=(
    "wan-vace-14b-v2v"
    "hy-omniweaving-v2v"
    "magi-24b-v2v"
    "ltx-2.3-dev-v2v"
    "cosmos3-super-v2v"
)

declare -a COMMERCIAL_MODELS=(
    "runway-aleph-v2v"
    "kling-v2-6-v2v"
    "luma-ray-3.2-v2v"
    "wan-2.7-video-edit"
    "gemini-omni-flash-video-edit"
)

# Commercial API keys lookup (bash 3.2 compatible - no associative arrays)
_get_api_key_for_model() {
    case "$1" in
        runway-aleph-v2v) echo "RUNWAYML_API_SECRET" ;;
        kling-v2-6-v2v) echo "KLING_API_KEY" ;;
        luma-ray-3.2-v2v) echo "LUMA_AGENTS_API_KEY" ;;
        wan-2.7-video-edit|gemini-omni-flash-video-edit) echo "WAVESPEED_API_KEY" ;;
        *) echo "" ;;
    esac
}

# ============================================================================
# MODEL HELPERS
# ============================================================================

is_opensource_model() {
    local target="$1"
    for model in "${OPENSOURCE_MODELS[@]}"; do
        [[ "$model" == "$target" ]] && return 0
    done
    return 1
}

is_commercial_model() {
    local target="$1"
    for model in "${COMMERCIAL_MODELS[@]}"; do
        [[ "$model" == "$target" ]] && return 0
    done
    return 1
}

get_commercial_env_var() {
    _get_api_key_for_model "$1"
}

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "$1"
    echo "────────────────────────────────────────────────────────────────"
}

print_success() { echo "   ✅ $1"; }
print_error()   { echo "   ❌ $1"; }
print_warning() { echo "   ⚠️  $1"; }
print_skip()    { echo "   ⏭️  $1"; }
print_info()    { echo "   📌 $1"; }
print_step()    { echo "🔧 $1"; }
print_download(){ echo "📥 $1"; }

# ============================================================================
# VENV FUNCTIONS
# ============================================================================

get_model_venv_path() {
    echo "${ENVS_DIR}/$1"
}

model_venv_exists() {
    [[ -f "$(get_model_venv_path "$1")/bin/python" ]]
}

activate_model_venv() {
    source "$(get_model_venv_path "$1")/bin/activate"
}

create_model_venv() {
    local model="$1"
    local venv_path
    venv_path="$(get_model_venv_path "$model")"

    # Always start fresh - remove existing environment if present
    if [[ -d "$venv_path" ]]; then
        print_step "Removing existing environment: ${model}"
        rm -rf "$venv_path"
        print_success "Old environment removed"
    fi

    print_step "Creating virtual environment: ${model}"
    mkdir -p "${ENVS_DIR}"
    python3 -m venv "$venv_path"

    source "${venv_path}/bin/activate"
    pip install -q --upgrade pip setuptools wheel
    deactivate

    print_success "Virtual environment created: ${model}"
}

# ============================================================================
# REPO + CHECKPOINT FUNCTIONS
# ============================================================================

clone_model_repo() {
    # clone_model_repo <git-url> <dest-name> [ref]
    local url="$1"
    local dest="${REPOS_DIR}/$2"
    local ref="${3:-}"

    if [[ -d "$dest/.git" ]]; then
        print_skip "Repo exists: $2"
    else
        print_download "Cloning $2"
        mkdir -p "${REPOS_DIR}"
        git clone "$url" "$dest"
        print_success "Repo ready: $2"
    fi

    if [[ -n "$ref" ]]; then
        git -C "$dest" fetch --quiet origin "$ref" || true
        git -C "$dest" checkout --quiet "$ref"
        print_info "Checked out: $ref"
    fi
}

download_hf_checkpoint() {
    # download_hf_checkpoint <hf-repo-id> <dest-dir-name> [size-desc]
    # Idempotent: skips when the destination is already populated.
    local repo_id="$1"
    local dest="${WEIGHTS_DIR}/$2"
    local size_desc="${3:-}"

    if [[ -d "$dest" ]] && [[ -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
        print_skip "Checkpoint exists: $2"
        return 0
    fi

    print_download "Downloading ${repo_id} ${size_desc:+- ${size_desc}}"
    mkdir -p "$dest"
    hf download "$repo_id" --local-dir "$dest"
    print_success "Checkpoint ready: $2"
}

# ============================================================================
# COMMERCIAL API FUNCTIONS
# ============================================================================

check_api_key() {
    local value="${!1:-}"
    [[ -n "$value" ]]
}

load_env_file() {
    local env_file="${V2V_ROOT}/.env"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
    fi
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

validate_model() {
    local model="$1"
    local test_output="${V2V_ROOT}/test_outputs"
    local timeout_seconds=10800  # 3 hours (open-source flagship models are slow)

    print_step "Validating ${model} on the turntable smoke test... (timeout: ${timeout_seconds}s)"
    echo ""

    set +e
    timeout "$timeout_seconds" python3 "${V2V_ROOT}/run.py" \
        --questions-dir "${TESTS_DIR}" \
        --output-dir "$test_output" \
        --model "$model"
    local exit_code=$?
    set -e

    local video_count
    video_count=$(find "${test_output}/${model}" -name "*.mp4" 2>/dev/null | wc -l)

    if [[ $exit_code -eq 0 ]] && [[ $video_count -ge 1 ]]; then
        print_success "${model}: ${video_count} video(s) generated ✓"
        return 0
    elif [[ $exit_code -eq 124 ]]; then
        print_warning "${model}: TIMEOUT (>${timeout_seconds}s)"
        return 1
    else
        print_error "${model}: FAILED - see output above"
        return 1
    fi
}

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

ensure_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        print_success "ffmpeg present"
        return 0
    fi
    print_step "Installing ffmpeg (requires sudo)..."
    sudo apt-get update -q
    sudo apt-get install -y -q ffmpeg
    print_success "ffmpeg installed"
}

cd "${V2V_ROOT}"
