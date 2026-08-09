"""Lazy imports so loading one provider never pulls in the others' SDKs."""

import importlib

__all__ = [
    "LumaInference", "LumaWrapper",
    "KlingService", "KlingWrapper",
    "RunwayService", "RunwayWrapper",
    "Wan27Service", "Wan27Wrapper",
    "GeminiOmniService", "GeminiOmniWrapper",
    "VaceService", "VaceWrapper",
    "OmniWeavingService", "OmniWeavingWrapper",
    "MagiService", "MagiWrapper",
    "Ltx23Service", "Ltx23Wrapper",
    "Cosmos3Service", "Cosmos3Wrapper",
]

_MODULE_MAP = {
    "luma_inference": ["LumaInference", "LumaWrapper"],
    "kling_inference": ["KlingService", "KlingWrapper"],
    "runway_inference": ["RunwayService", "RunwayWrapper"],
    "wan27_inference": ["Wan27Service", "Wan27Wrapper"],
    "gemini_omni_inference": ["GeminiOmniService", "GeminiOmniWrapper"],
    "vace_inference": ["VaceService", "VaceWrapper"],
    "omniweaving_inference": ["OmniWeavingService", "OmniWeavingWrapper"],
    "magi_inference": ["MagiService", "MagiWrapper"],
    "ltx23_inference": ["Ltx23Service", "Ltx23Wrapper"],
    "cosmos3_inference": ["Cosmos3Service", "Cosmos3Wrapper"],
}


def __getattr__(name: str):
    for module_name, symbols in _MODULE_MAP.items():
        if name in symbols:
            module = importlib.import_module(f".{module_name}", __name__)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
