from v2vinferkit.runner.MODEL_CATALOG import AVAILABLE_MODELS


NEW_FAL_MODELS = {
    "wan-3.0-video-edit": "alibaba/wan-3.0/reference-to-video",
    "wan-3.0-prime-video-edit": "alibaba/wan-3.0-prime/reference-to-video",
    "minimax-h3-v2v": "minimax/h3/reference-to-video",
    "seedance-2.0-v2v": "bytedance/seedance-2.0/reference-to-video",
    "seedance-2.0-fast-v2v": "bytedance/seedance-2.0/fast/reference-to-video",
    "seedance-2.0-mini-v2v": "bytedance/seedance-2.0/mini/reference-to-video",
    "seedance-2.5-v2v": "bytedance/seedance-2.5/reference-to-video",
    "gemini-omni-flash-1.1-video-edit": "google/gemini-omni-flash/v1.1/edit",
    "kling-o3-pro-video-edit": "fal-ai/kling-video/o3/pro/video-to-video/edit",
    "happy-horse-1.0-video-edit": "alibaba/happy-horse/video-edit",
    "grok-imagine-video-edit": "xai/grok-imagine-video/edit-video",
}


def test_catalog_contains_old_and_new_models():
    assert len(AVAILABLE_MODELS) == 21
    assert "wan-2.7-video-edit" in AVAILABLE_MODELS
    assert "gemini-omni-flash-video-edit" in AVAILABLE_MODELS
    assert "kling-v2-6-v2v" in AVAILABLE_MODELS


def test_new_fal_models_use_shared_wrapper_and_expected_endpoint():
    for model_id, endpoint in NEW_FAL_MODELS.items():
        config = AVAILABLE_MODELS[model_id]
        assert config["wrapper_module"] == "v2vinferkit.models.fal_v2v_inference"
        assert config["wrapper_class"] == "FalV2VWrapper"
        assert config["service_class"] == "FalV2VService"
        assert config["args"]["endpoint"] == endpoint
        assert config["args"]["profile"]
        assert config["modality"] == "v2v"


def test_each_new_base_model_has_one_hosted_endpoint():
    endpoints = [AVAILABLE_MODELS[model_id]["args"]["endpoint"] for model_id in NEW_FAL_MODELS]
    assert len(endpoints) == len(set(endpoints))

