"""Lazy imports so loading one provider never pulls in the others' SDKs."""

import importlib

__all__ = [
    "RunwayService", "RunwayWrapper",
    "FalV2VService", "FalV2VWrapper",
    "VaceService", "VaceWrapper",
    "OmniWeavingService", "OmniWeavingWrapper",
    "MagiService", "MagiWrapper",
    "Ltx23Service", "Ltx23Wrapper",
    "Cosmos3Service", "Cosmos3Wrapper",
    "JoyAIService", "JoyAIWrapper",
    "BerniniService", "BerniniWrapper",
    "KiwiService", "KiwiWrapper",
    "EdittoService", "EdittoWrapper",
    "SamaService", "SamaWrapper",
    "OmniVideo2Service", "OmniVideo2Wrapper",
    "CoinVEService", "CoinVEWrapper",
    "LucyService", "LucyWrapper",
    "LanceService", "LanceWrapper",
    "LongCatExtendService", "LongCatExtendWrapper",
    "OpenSora2ExtendService", "OpenSora2ExtendWrapper",
    "UniVideoService", "UniVideoWrapper",
    "SkyReelsDFExtendService", "SkyReelsDFExtendWrapper",
    "SelfForcingExtendService", "SelfForcingExtendWrapper",
]

_MODULE_MAP = {
    "runway_inference": ["RunwayService", "RunwayWrapper"],
    "fal_v2v_inference": ["FalV2VService", "FalV2VWrapper"],
    "vace_inference": ["VaceService", "VaceWrapper"],
    "omniweaving_inference": ["OmniWeavingService", "OmniWeavingWrapper"],
    "magi_inference": ["MagiService", "MagiWrapper"],
    "ltx23_inference": ["Ltx23Service", "Ltx23Wrapper"],
    "cosmos3_inference": ["Cosmos3Service", "Cosmos3Wrapper"],
    "joyai_inference": ["JoyAIService", "JoyAIWrapper"],
    "bernini_inference": ["BerniniService", "BerniniWrapper"],
    "kiwi_inference": ["KiwiService", "KiwiWrapper"],
    "editto_inference": ["EdittoService", "EdittoWrapper"],
    "sama_inference": ["SamaService", "SamaWrapper"],
    "omnivideo2_inference": ["OmniVideo2Service", "OmniVideo2Wrapper"],
    "coinve_inference": ["CoinVEService", "CoinVEWrapper"],
    "lucy_inference": ["LucyService", "LucyWrapper"],
    "lance_inference": ["LanceService", "LanceWrapper"],
    "longcat_inference": ["LongCatExtendService", "LongCatExtendWrapper"],
    "opensora2_inference": ["OpenSora2ExtendService", "OpenSora2ExtendWrapper"],
    "univideo_inference": ["UniVideoService", "UniVideoWrapper"],
    "skyreels_inference": ["SkyReelsDFExtendService", "SkyReelsDFExtendWrapper"],
    "self_forcing_inference": ["SelfForcingExtendService", "SelfForcingExtendWrapper"],
}


def __getattr__(name: str):
    for module_name, symbols in _MODULE_MAP.items():
        if name in symbols:
            module = importlib.import_module(f".{module_name}", __name__)
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
