"""Lazy imports so loading one provider never pulls in the others' SDKs."""

import importlib

__all__ = [
    "LumaInference", "LumaWrapper",
    "KlingService", "KlingWrapper",
    "RunwayService", "RunwayWrapper",
    "Wan27Service", "Wan27Wrapper",
    "GeminiOmniService", "GeminiOmniWrapper",
]

_MODULE_MAP = {
    "luma_inference": ["LumaInference", "LumaWrapper"],
    "kling_inference": ["KlingService", "KlingWrapper"],
    "runway_inference": ["RunwayService", "RunwayWrapper"],
    "wan27_inference": ["Wan27Service", "Wan27Wrapper"],
    "gemini_omni_inference": ["GeminiOmniService", "GeminiOmniWrapper"],
}


def __getattr__(name: str):
    for module_name, symbols in _MODULE_MAP.items():
        if name in symbols:
            module = importlib.import_module(f".{module_name}", __name__)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
