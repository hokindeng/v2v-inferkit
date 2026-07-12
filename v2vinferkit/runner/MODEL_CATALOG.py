"""Model registry — the V2V (video-to-video) subset of VBVR-InferKit.

Entries keep the exact upstream schema (Video-Reason/VBVR-InferKit
vbvrinferkit/runner/MODEL_CATALOG.py) so they can be diffed and synced against
upstream. Every model here has modality v2v: it consumes a conditioning video
(first_video.mp4) plus a text prompt, and returns an edited/generated video.

All five are commercial APIs — no local weights, venvs, or GPU needed.
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

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    **LUMA_MODELS,
    **KLING_MODELS,
    **RUNWAY_MODELS,
    **WAN27_MODELS,
    **GEMINI_OMNI_MODELS,
}

MODEL_FAMILIES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Luma": LUMA_MODELS,
    "Kling AI": KLING_MODELS,
    "Runway ML": RUNWAY_MODELS,
    "WAN 2.7": WAN27_MODELS,
    "Gemini Omni": GEMINI_OMNI_MODELS,
}


def add_model_family(family_name: str, models: Dict[str, Dict[str, Any]]) -> None:
    """Register an extra model family at runtime (upstream-compatible hook)."""
    MODEL_FAMILIES[family_name] = models
    AVAILABLE_MODELS.update(models)
