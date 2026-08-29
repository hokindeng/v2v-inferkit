"""Model registry — V2V (video-to-video) models.

Every model here has modality v2v: it consumes a conditioning video
(first_video.mp4) plus a text prompt, and returns an edited/generated video.

Commercial API entries need no local weights, venvs, or GPU; open-source
entries run locally in their model-specific venv.
"""
from typing import Any, Dict

# Hosted APIs use one FAL_KEY for upload and inference unless noted otherwise.
LUMA_MODELS = {
    "luma-ray-3.2-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "ray-3.2",
        "modality": "v2v",
        "description": "Luma Ray 3.2 video editing via fal.ai",
        "family": "Luma",
        "args": {
            "endpoint": "luma/agent/ray/v3.2/video-to-video",
            "profile": "luma_ray_3_2",
        },
    },
}

# Kling AI via fal.ai (FAL_KEY)
KLING_MODELS = {
    "kling-v2-6-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "kling-video-o1",
        "modality": "v2v",
        "description": "Kling O1 video editing via fal.ai (legacy model ID)",
        "family": "Kling AI",
        "args": {
            "endpoint": "fal-ai/kling-video/o1/video-to-video/edit",
            "profile": "kling_o1_edit",
        },
    },
    "kling-o3-pro-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "kling-o3-pro",
        "modality": "v2v",
        "description": "Kling O3 Pro video editing via fal.ai",
        "family": "Kling AI",
        "args": {
            "endpoint": "fal-ai/kling-video/o3/pro/video-to-video/edit",
            "profile": "kling_o3_edit",
        },
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

# WAN 2.7 via fal.ai (FAL_KEY)
WAN27_MODELS = {
    "wan-2.7-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "wan-2.7",
        "modality": "v2v",
        "description": "WAN 2.7 prompt-driven video editing via fal.ai",
        "family": "WAN 2.7",
        "args": {
            "endpoint": "fal-ai/wan/v2.7/edit-video",
            "profile": "wan27_edit",
        },
    },
}

# Wan 3.0 reference-to-video tiers via fal.ai (FAL_KEY)
WAN3_MODELS = {
    "wan-3.0-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "wan-3.0",
        "modality": "v2v",
        "description": "Wan 3.0 reference-video editing via fal.ai (720p default)",
        "family": "Wan 3.0",
        "args": {
            "endpoint": "alibaba/wan-3.0/reference-to-video",
            "profile": "wan3",
        },
    },
    "wan-3.0-prime-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "wan-3.0-prime",
        "modality": "v2v",
        "description": "Wan 3.0 Prime reference-video editing via fal.ai (720p default)",
        "family": "Wan 3.0",
        "args": {
            "endpoint": "alibaba/wan-3.0-prime/reference-to-video",
            "profile": "wan3",
        },
    },
}

# Gemini Omni Flash video editing via fal.ai (FAL_KEY)
GEMINI_OMNI_MODELS = {
    "gemini-omni-flash-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "gemini-omni-flash",
        "modality": "v2v",
        "description": "Gemini Omni Flash natural-language video editing via fal.ai",
        "family": "Gemini Omni",
        "args": {
            "endpoint": "google/gemini-omni-flash/edit",
            "profile": "gemini_omni",
        },
    },
    "gemini-omni-flash-1.1-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "gemini-omni-flash-1.1",
        "modality": "v2v",
        "description": "Gemini Omni Flash 1.1 natural-language video editing via fal.ai",
        "family": "Gemini Omni",
        "args": {
            "endpoint": "google/gemini-omni-flash/v1.1/edit",
            "profile": "gemini_omni_edit",
        },
    },
}

# MiniMax H3 multimodal reference-to-video via fal.ai (FAL_KEY)
MINIMAX_H3_MODELS = {
    "minimax-h3-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "minimax-h3",
        "modality": "v2v",
        "description": "MiniMax H3 multimodal reference-to-video via fal.ai (768P default)",
        "family": "MiniMax H3",
        "args": {
            "endpoint": "minimax/h3/reference-to-video",
            "profile": "minimax_h3",
        },
    },
}

# ByteDance Seedance reference-to-video versions and performance tiers (FAL_KEY)
SEEDANCE_MODELS = {
    "seedance-2.0-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "seedance-2.0",
        "modality": "v2v",
        "description": "Seedance 2.0 reference-video generation and editing via fal.ai",
        "family": "Seedance",
        "args": {
            "endpoint": "bytedance/seedance-2.0/reference-to-video",
            "profile": "seedance_2",
        },
    },
    "seedance-2.0-fast-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "seedance-2.0-fast",
        "modality": "v2v",
        "description": "Seedance 2.0 Fast reference-video generation and editing via fal.ai",
        "family": "Seedance",
        "args": {
            "endpoint": "bytedance/seedance-2.0/fast/reference-to-video",
            "profile": "seedance_2",
        },
    },
    "seedance-2.0-mini-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "seedance-2.0-mini",
        "modality": "v2v",
        "description": "Seedance 2.0 Mini reference-video generation and editing via fal.ai",
        "family": "Seedance",
        "args": {
            "endpoint": "bytedance/seedance-2.0/mini/reference-to-video",
            "profile": "seedance_2",
        },
    },
    "seedance-2.5-v2v": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "seedance-2.5",
        "modality": "v2v",
        "description": "Seedance 2.5 reference-video generation and editing via fal.ai",
        "family": "Seedance",
        "args": {
            "endpoint": "bytedance/seedance-2.5/reference-to-video",
            "profile": "seedance_2_5",
        },
    },
}

# Alibaba Happy Horse video editor via fal.ai (FAL_KEY)
HAPPY_HORSE_MODELS = {
    "happy-horse-1.0-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "happy-horse-1.0",
        "modality": "v2v",
        "description": "Happy Horse 1.0 natural-language video editing via fal.ai",
        "family": "Happy Horse",
        "args": {
            "endpoint": "alibaba/happy-horse/video-edit",
            "profile": "happy_horse_edit",
        },
    },
}

# xAI Grok Imagine Video editor via fal.ai (FAL_KEY)
GROK_MODELS = {
    "grok-imagine-video-edit": {
        "wrapper_module": "v2vinferkit.models.fal_v2v_inference",
        "wrapper_class": "FalV2VWrapper",
        "service_class": "FalV2VService",
        "model": "grok-imagine-video",
        "modality": "v2v",
        "description": "Grok Imagine natural-language video editing via fal.ai",
        "family": "Grok Imagine",
        "args": {
            "endpoint": "xai/grok-imagine-video/edit-video",
            "profile": "grok_edit",
        },
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
    **WAN3_MODELS,
    **GEMINI_OMNI_MODELS,
    **MINIMAX_H3_MODELS,
    **SEEDANCE_MODELS,
    **HAPPY_HORSE_MODELS,
    **GROK_MODELS,
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
    "Wan 3.0": WAN3_MODELS,
    "Gemini Omni": GEMINI_OMNI_MODELS,
    "MiniMax H3": MINIMAX_H3_MODELS,
    "Seedance": SEEDANCE_MODELS,
    "Happy Horse": HAPPY_HORSE_MODELS,
    "Grok Imagine": GROK_MODELS,
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
