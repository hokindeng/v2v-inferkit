"""SAMA-14B semantic video editing via its official single-video CLI."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, run_command, success_result, weights_path


class SamaService:
    def __init__(self, model: str = "syxbb/SAMA-14B"):
        self.model = model
        self.repo = repo_path("SAMA", "SAMA_REPO_PATH")
        self.base_model = Path(os.environ.get("SAMA_BASE_MODEL_PATH") or str(weights_path("Wan2.1-T2V-14B")))
        self.checkpoint_root = Path(os.environ.get("SAMA_WEIGHTS_PATH") or str(weights_path("SAMA-14B")))

    def _checkpoint(self) -> Path:
        override = os.environ.get("SAMA_CHECKPOINT")
        if override:
            return require_file(override, "SAMA checkpoint")
        root = require_dir(self.checkpoint_root, "SAMA checkpoint directory")
        candidates = sorted(root.rglob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No SAMA .safetensors checkpoint found under {root}")
        return candidates[0]

    def generate_video(self, video_path, prompt, output_path: Path, *, num_frames=81, height=480, width=832, seed=1, fps=20, device="cuda:0", timeout=7200) -> Dict[str, Any]:
        repo = require_dir(self.repo, "SAMA repository")
        source = require_file(video_path, "source video")
        base_model = require_dir(self.base_model, "Wan2.1-T2V-14B base model")
        checkpoint = self._checkpoint()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sama_v2v_") as tmp:
            command = [
                sys.executable, repo / "infer_sh" / "inference.py",
                "--src-video", source, "--prompt", prompt,
                "--output-dir", tmp, "--model-root", base_model,
                "--state-dict", checkpoint, "--device", device,
                "--height", str(height), "--width", str(width),
                "--max-frames", str(num_frames), "--fps", str(fps),
                "--seed", str(seed), "--tiled", "--prompt-prefix", "--overwrite",
            ]
            run_command(command, cwd=repo, timeout=timeout)
            generated = require_file(Path(tmp) / source.name, "SAMA output video")
            shutil.copy2(generated, output_path)
        return {"checkpoint": str(checkpoint), "base_model": str(base_model), "num_frames": num_frames, "height": height, "width": width, "seed": seed, "fps": fps}


class SamaWrapper(ModelWrapper):
    def __init__(self, model="syxbb/SAMA-14B", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = SamaService(model)

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
        return success_result(self.model, text_prompt, start, output, video_path, "sama", metadata)
