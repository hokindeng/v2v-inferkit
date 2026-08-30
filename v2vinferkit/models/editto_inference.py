"""Ditto/Editto VACE LoRA video editing via the official DiffSynth script."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, run_command, success_result, weights_path


class EdittoService:
    def __init__(self, model: str = "QingyanBai/Ditto_models"):
        self.model = model
        self.repo = repo_path("Ditto", "EDITTO_REPO_PATH")
        self.lora_root = Path(os.environ.get("EDITTO_WEIGHTS_PATH") or str(weights_path("Ditto_models")))

    def _lora_path(self) -> Path:
        override = os.environ.get("EDITTO_LORA_PATH")
        if override:
            return require_file(override, "Ditto LoRA")
        root = require_dir(self.lora_root, "Ditto weights")
        preferred = root / "models" / "ditto_global.safetensors"
        if preferred.is_file():
            return preferred
        candidates = sorted((root / "models").glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No Ditto .safetensors file found under {root}")
        return candidates[0]

    def generate_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_path: Path,
        *,
        num_frames: int = 73,
        height: int = 480,
        width: int = 832,
        seed: int = 1,
        fps: int = 20,
        lora_alpha: float = 1.0,
        device_id: int = 0,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        repo = require_dir(self.repo, "Ditto repository")
        source = require_file(video_path, "source video")
        lora = self._lora_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_frames = max(1, min(int(num_frames), 73))
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        command = [
            sys.executable, repo / "inference" / "infer_ditto.py",
            "--input_video", source,
            "--output_video", output_path,
            "--prompt", prompt,
            "--lora_path", lora,
            "--lora_alpha", str(lora_alpha),
            "--device_id", str(device_id),
            "--height", str(height),
            "--width", str(width),
            "--num_frames", str(num_frames),
            "--seed", str(seed),
            "--fps", str(fps),
        ]
        run_command(command, cwd=repo, timeout=timeout)
        require_file(output_path, "Ditto output video")
        return {"checkpoint": str(lora), "num_frames": num_frames, "height": height, "width": width, "seed": seed, "fps": fps, "lora_alpha": lora_alpha}


class EdittoWrapper(ModelWrapper):
    def __init__(self, model: str = "QingyanBai/Ditto_models", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = EdittoService(model)

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
        return success_result(self.model, text_prompt, start, output, video_path, "editto", metadata)
