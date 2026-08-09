"""Model registry — V2V (video-to-video) models.

Every model here has modality v2v: it consumes a conditioning video
(first_video.mp4) plus a text prompt, and returns an edited/generated video.

Commercial API entries need no local weights, venvs, or GPU; open-source
entries run locally in their model-specific venv.
"""
from typing import Any, Dict

# Luma AI (LUMA_AGENTS_API_KEY or LUMA_API_KEY; input uploaded via FAL_KEY)
LUMA_MODELS = {
    "luma-ray-3.2-v2v": {
        "wrapper_module": "v2vinferkit.models.luma_inference",
        "wrapper_class": "LumaWrapper",
        "service_class": "LumaInference",
        "model": "ray-3.2",
        "modality": "v2v",
        "description": "Luma Ray 3.2 Video-to-Video (video_edit via Agents API)",
        "family": "Luma"
    },
}

# Kling AI (KLING_API_KEY, or KLING_ACCESS_KEY + KLING_SECRET_KEY for JWT;
# input uploaded via FAL_KEY)
KLING_MODELS = {
    "kling-v2-6-v2v": {
        "wrapper_module": "v2vinferkit.models.kling_inference",
        "wrapper_class": "KlingWrapper",
        "service_class": "KlingService",
        "model": "kling-video-o1",
        "modality": "v2v",
        "description": "Kling Omni V2V via /v1/videos/omni-video endpoint",
        "family": "Kling AI"
    },
}

# Runway ML (RUNWAYML_API_SECRET; input via Runway's own ephemeral upload)
RUNWAY_MODELS = {
    "runway-aleph-v2v": {
        "wrapper_module": "v2vinferkit.models.runway_inference",
        "wrapper_class": "RunwayWrapper",
        "service_class": "RunwayService",
        "model": "aleph2",
        "modality": "v2v",
        "description": "Runway Aleph 2 - Video-to-video (text + video -> video)",
        "family": "Runway ML"
    },
}

# WAN 2.7 via WaveSpeed (WAVESPEED_API_KEY; input inlined as base64 data-URI)
WAN27_MODELS = {
    "wan-2.7-video-edit": {
        "wrapper_module": "v2vinferkit.models.wan27_inference",
        "wrapper_class": "Wan27Wrapper",
        "service_class": "Wan27Service",
        "model": "wan-2.7",
        "modality": "v2v",
        "description": "WAN 2.7 Video Edit - Prompt-driven video editing (720p/1080p)",
        "family": "WAN 2.7"
    },
}

# Gemini Omni Flash Video Edit via WaveSpeed (WAVESPEED_API_KEY)
GEMINI_OMNI_MODELS = {
    "gemini-omni-flash-video-edit": {
        "wrapper_module": "v2vinferkit.models.gemini_omni_inference",
        "wrapper_class": "GeminiOmniWrapper",
        "service_class": "GeminiOmniService",
        "model": "gemini-omni-flash",
        "modality": "v2v",
        "description": "Gemini Omni Flash Video Edit - Natural-language video editing",
        "family": "Gemini Omni"
    },
}

# ---------------------------------------------------------------------------
# Open-source models (local GPU; each runs in its own venv under envs/<name>,
# installed via setup/install_model.sh --model <name>; weights under
# $V2V_WEIGHTS_DIR, default <repo>/weights)
# ---------------------------------------------------------------------------

# Wan2.1-VACE-14B via diffusers WanVACEPipeline (repo CLI OOMs on <80GB GPUs)
WAN_VACE_MODELS = {
    "wan-vace-14b-v2v": {
        "wrapper_module": "v2vinferkit.models.vace_inference",
        "wrapper_class": "VaceWrapper",
        "service_class": "VaceService",
        "model": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "modality": "v2v",
        "description": "Wan2.1-VACE-14B - open-source V2V editing (diffusers, 1x48GB GPU, 480p)",
        "family": "Wan-VACE (open-source)"
    },
}

# Tencent HY-OmniWeaving --task editing (HunyuanVideo-1.5 backbone)
OMNIWEAVING_MODELS = {
    "hy-omniweaving-v2v": {
        "wrapper_module": "v2vinferkit.models.omniweaving_inference",
        "wrapper_class": "OmniWeavingWrapper",
        "service_class": "OmniWeavingService",
        "model": "tencent/HY-OmniWeaving",
        "modality": "v2v",
        "description": "HY-OmniWeaving - open-source V2V editing (1 GPU w/ offload, 480p only)",
        "family": "Hunyuan OmniWeaving (open-source)"
    },
}

# MAGI-1 24B base. NOTE: MAGI v2v = prefix-video CONTINUATION, not editing —
# the prompt steers only newly generated frames. Kept for benchmark
# completeness pending a team ruling on whether continuation counts as V2V.
MAGI_MODELS = {
    "magi-24b-v2v": {
        "wrapper_module": "v2vinferkit.models.magi_inference",
        "wrapper_class": "MagiWrapper",
        "service_class": "MagiService",
        "model": "sand-ai/MAGI-1:24B_base",
        "modality": "v2v",
        "description": "MAGI-1 24B - video CONTINUATION (not editing); needs 4x80GB+ GPUs",
        "family": "MAGI (open-source)"
    },
}

# LTX-2.3 via ICLoraPipeline. NOTE: the only official v2v pipeline is
# distilled-checkpoint-only (docs/pipelines.md) — collides with the
# flagship-only rule; checkpoint variant recorded in result metadata pending
# a team ruling. Override via LTX2_IC_LORA_CHECKPOINT.
LTX23_MODELS = {
    "ltx-2.3-dev-v2v": {
        "wrapper_module": "v2vinferkit.models.ltx23_inference",
        "wrapper_class": "Ltx23Wrapper",
        "service_class": "Ltx23Service",
        "model": "Lightricks/LTX-2.3:ic-lora",
        "modality": "v2v",
        "description": "LTX-2.3 IC-LoRA video conditioning (official v2v is distilled-only - pending ruling)",
        "family": "LTX (open-source)"
    },
}

# NVIDIA Cosmos3-Super video transfer (edge control derived from source video)
COSMOS3_MODELS = {
    "cosmos3-super-v2v": {
        "wrapper_module": "v2vinferkit.models.cosmos3_inference",
        "wrapper_class": "Cosmos3Wrapper",
        "service_class": "Cosmos3Service",
        "model": "Cosmos3-Super",
        "modality": "v2v",
        "description": "Cosmos3-Super video transfer - open-source V2V (needs 4x80GB GPUs, 132GB ckpt)",
        "family": "NVIDIA Cosmos (open-source)"
    },
}

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    **LUMA_MODELS,
    **KLING_MODELS,
    **RUNWAY_MODELS,
    **WAN27_MODELS,
    **GEMINI_OMNI_MODELS,
    **WAN_VACE_MODELS,
    **OMNIWEAVING_MODELS,
    **MAGI_MODELS,
    **LTX23_MODELS,
    **COSMOS3_MODELS,
}

MODEL_FAMILIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Luma": LUMA_MODELS,
    "Kling AI": KLING_MODELS,
    "Runway ML": RUNWAY_MODELS,
    "WAN 2.7": WAN27_MODELS,
    "Gemini Omni": GEMINI_OMNI_MODELS,
    "Wan-VACE (open-source)": WAN_VACE_MODELS,
    "Hunyuan OmniWeaving (open-source)": OMNIWEAVING_MODELS,
    "MAGI (open-source)": MAGI_MODELS,
    "LTX (open-source)": LTX23_MODELS,
    "NVIDIA Cosmos (open-source)": COSMOS3_MODELS,
}


def add_model_family(family_name: str, models: Dict[str, Dict[str, Any]]) -> None:
    """Register an extra model family at runtime (upstream-compatible hook)."""
    MODEL_FAMILIES[family_name] = models
    AVAILABLE_MODELS.update(models)
