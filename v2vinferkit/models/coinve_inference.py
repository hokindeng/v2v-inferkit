"""CoinVE multi-instruction video editing via its official single-video CLI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, run_command, success_result, weights_path


class CoinVEService:
    def __init__(self, model: str = "FireCRT/CoinVE-Edit"):
        self.model = model
        self.repo = repo_path("CoinVE-200K", "COINVE_REPO_PATH")
        self.checkpoint_root = Path(os.environ.get("COINVE_WEIGHTS_PATH") or str(weights_path("CoinVE-Edit")))
        self.base_root = Path(os.environ.get("COINVE_BASE_MODELS_PATH") or str(weights_path()))
        self.qwen = Path(os.environ.get("COINVE_QWEN_PATH") or str(weights_path("Qwen3-VL-8B-Instruct")))

    def _checkpoint(self) -> Path:
        override = os.environ.get("COINVE_CHECKPOINT")
        if override:
            return require_file(override, "CoinVE checkpoint")
        root = require_dir(self.checkpoint_root, "CoinVE checkpoint directory")
        candidates = sorted(root.rglob("coinve_edit_composite*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No CoinVE composite checkpoint found under {root}")
        return candidates[0]

    def generate_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_path: Path,
        *,
        instructions: Optional[List[str]] = None,
        num_frames: int = 49,
        max_pixels: int = 921600,
        num_inference_steps: int = 50,
        seed: int = 0,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        repo = require_dir(self.repo, "CoinVE repository")
        project = require_dir(repo / "CoinVE-Edit", "CoinVE-Edit project")
        source = require_file(video_path, "source video")
        checkpoint = self._checkpoint()
        require_dir(self.base_root / "Wan-AI" / "Wan2.1-T2V-14B", "Wan2.1-T2V-14B base model")
        qwen = require_dir(self.qwen, "Qwen3-VL-8B checkpoint")
        prompts = [item.strip() for item in (instructions or [prompt]) if item and item.strip()]
        if not prompts:
            raise ValueError("At least one CoinVE edit instruction is required")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, project / "infer_coinve_single.py",
            "--src_video", source,
            "--prompts", *prompts,
            "--composite_checkpoint", checkpoint,
            "--output_dir", output_path.parent,
            "--output_name", output_path.stem,
            "--local_model_path", self.base_root,
            "--mllm_model", qwen,
            "--eval_max_pixels", str(max_pixels),
            "--eval_max_frame", str(num_frames),
            "--num_inference_steps", str(num_inference_steps),
            "--seed", str(seed),
            "--no_side_by_side",
        ]
        run_command(command, cwd=project, timeout=timeout)
        require_file(output_path, "CoinVE output video")
        return {"checkpoint": str(checkpoint), "instructions": prompts, "num_frames": num_frames, "max_pixels": max_pixels, "num_inference_steps": num_inference_steps, "seed": seed}


class CoinVEWrapper(ModelWrapper):
    def __init__(self, model="FireCRT/CoinVE-Edit", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = CoinVEService(model)

    def generate(self, image_path, text_prompt, duration=5.0, output_filename=None, video_path=None, **kwargs):
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        kwargs.pop("question_data", None)
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, **kwargs)
        except Exception as exc:
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "coinve", metadata)
