"""v2v-inferkit inference runner — direct-import dispatch for API models.

Slimmed from VBVR-InferKit's runner: every model in this kit is a commercial
API, so the per-model venv + subprocess path was removed. Wrappers are loaded
dynamically from MODEL_CATALOG and called in-process.
"""

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
from ..models.base import ModelWrapper

_UNKNOWN_DOMAIN = "unknown_task"
_UNKNOWN_TASK_ID = "unknown"


def _extract_domain_and_task(question_data: Optional[Dict[str, Any]]) -> tuple:
    """Extract output domain folder and task id from optional question metadata."""
    if not question_data:
        return _UNKNOWN_DOMAIN, _UNKNOWN_TASK_ID

    domain = question_data.get("domain_dir") or question_data.get("domain") or _UNKNOWN_DOMAIN
    task_id = question_data.get("id", _UNKNOWN_TASK_ID)
    return domain, task_id


def _build_wrapper_init_kwargs(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Build constructor kwargs for a model wrapper from catalog config."""
    model_config = AVAILABLE_MODELS[model_name]
    init_kwargs: Dict[str, Any] = {
        "model": model_config["model"],
        "output_dir": str(output_dir),
    }
    if "args" in model_config:
        init_kwargs.update(model_config["args"])
    return init_kwargs


def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    """Load wrapper class dynamically from catalog."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    config = AVAILABLE_MODELS[model_name]
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])

    return wrapper_class


def run_inference(
    model_name: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str = "./outputs",
    question_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run one inference with the specified model (standalone helper)."""
    generation_kwargs = dict(kwargs)
    if question_data:
        generation_kwargs["question_data"] = question_data

    wrapper_class = _load_model_wrapper(model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_id = generation_kwargs.pop('inference_id', f"{model_name}_{timestamp}")
    inference_dir = Path(output_dir) / inference_id
    video_dir = inference_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = _build_wrapper_init_kwargs(model_name, video_dir)
    wrapper = wrapper_class(**init_kwargs)

    result = wrapper.generate(image_path, text_prompt, **generation_kwargs)

    result["inference_dir"] = str(inference_dir)
    result["question_data"] = question_data
    return result


class InferenceRunner:
    """Inference runner with dynamic model loading and wrapper caching."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._wrapper_cache = {}

    def _get_or_create_wrapper(self, model_name: str) -> ModelWrapper:
        """Lazily initialize and cache wrappers."""
        if model_name not in self._wrapper_cache:
            wrapper_class = _load_model_wrapper(model_name)
            init_kwargs = _build_wrapper_init_kwargs(model_name, self.output_dir)
            self._wrapper_cache[model_name] = wrapper_class(**init_kwargs)
            print(f"Loaded model: {model_name}")
        return self._wrapper_cache[model_name]

    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference and save video as {task_id}.mp4 under domain folder."""
        domain_dir_name, task_id = _extract_domain_and_task(question_data)

        domain_dir = self.output_dir / domain_dir_name
        domain_dir.mkdir(parents=True, exist_ok=True)
        generation_kwargs = dict(kwargs)
        if question_data:
            generation_kwargs['question_data'] = question_data

        wrapper = self._get_or_create_wrapper(model_name)
        wrapper.output_dir = domain_dir
        result = wrapper.generate(image_path, text_prompt, **generation_kwargs)

        self._rename_video_to_task_id(domain_dir, task_id, result)
        print(f"\nInference complete: {domain_dir / f'{task_id}.mp4'}")
        return result

    def _rename_video_to_task_id(self, domain_dir: Path, task_id: str, result: Dict[str, Any]):
        """Rename generated video to {task_id}.mp4."""
        video_path = result.get("video_path")
        if not video_path:
            return

        video_path = Path(video_path)
        if not video_path.exists():
            return

        target_path = domain_dir / f"{task_id}.mp4"
        if video_path != target_path:
            video_path.rename(target_path)
            result["video_path"] = str(target_path)

    def list_models(self) -> Dict[str, str]:
        """List available models and their descriptions."""
        return {
            name: config["description"]
            for name, config in AVAILABLE_MODELS.items()
        }

    def list_models_by_family(self) -> Dict[str, Dict[str, str]]:
        """List models organized by family."""
        return {
            family_name: {
                name: config["description"]
                for name, config in family_models.items()
            }
            for family_name, family_models in MODEL_FAMILIES.items()
        }

    def get_model_families(self) -> Dict[str, int]:
        """Get model family statistics."""
        return {
            family_name: len(family_models)
            for family_name, family_models in MODEL_FAMILIES.items()
        }


from .MODEL_CATALOG import add_model_family  # noqa: E402,F401  (upstream-compatible re-export)
