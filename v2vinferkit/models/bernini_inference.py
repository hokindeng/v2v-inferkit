"""Bernini-R 1.3B/14B local video editing via the official single-GPU CLI."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, run_command, success_result, weights_path


class BerniniService:
    def __init__(self, model: str):
        self.model = model
        self.repo = repo_path("Bernini", "BERNINI_REPO_PATH")
        default_name = model.rsplit("/", 1)[-1]
        self.checkpoint = Path(os.environ.get("BERNINI_WEIGHTS_PATH") or str(weights_path(default_name)))

    def generate_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_path: Path,
        *,
        reference_image_path: Optional[Union[str, Path]] = None,
        num_frames: int = 81,
        height: int = 480,
        width: int = 848,
        num_inference_steps: int = 40,
        seed: int = 42,
        fps: int = 16,
        timeout: int = 7200,
    ) -> Dict[str, Any]:
        repo = require_dir(self.repo, "Bernini repository")
        checkpoint = require_dir(self.checkpoint, "Bernini checkpoint")
        source = require_file(video_path, "source video")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        num_frames = max(1, min(int(num_frames), 81))
        num_frames = ((num_frames - 1) // 4) * 4 + 1

        task_type = "rv2v" if reference_image_path else "v2v"
        command = [
            sys.executable,
            repo / "infer_single_gpu.py",
            "--config", checkpoint,
            "--prompt", prompt,
            "--task_type", task_type,
            "--guidance_mode", task_type,
            "--video", source,
            "--output", output_path,
            "--num_frames", str(num_frames),
            "--height", str(height),
            "--width", str(width),
            "--num_inference_steps", str(num_inference_steps),
            "--seed", str(seed),
            "--fps", str(fps),
        ]
        if reference_image_path:
            command.extend(["--images", require_file(reference_image_path, "reference image")])
        run_command(command, cwd=repo, timeout=timeout)
        require_file(output_path, "Bernini output video")
        return {
            "task_type": task_type,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "fps": fps,
            "checkpoint": str(checkpoint),
        }


class BerniniWrapper(ModelWrapper):
    def __init__(self, model: str, output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = BerniniService(model)

    def generate(
        self,
        image_path: Optional[Union[str, Path]],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        kwargs.pop("question_data", None)
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, **kwargs)
        except Exception as exc:
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "bernini", metadata)
