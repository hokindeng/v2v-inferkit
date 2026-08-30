"""Kiwi-Edit 5B instruction/reference video editing through Diffusers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, require_file, success_result, weights_path


class KiwiService:
    def __init__(self, model: str = "linyq/kiwi-edit-5b-instruct-reference-diffusers"):
        self.model = model
        self.checkpoint = Path(os.environ.get("KIWI_WEIGHTS_PATH") or str(weights_path("kiwi-edit-5b-instruct-reference-diffusers")))
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import DiffusionPipeline
        source = str(self.checkpoint) if self.checkpoint.is_dir() else self.model
        self.pipe = DiffusionPipeline.from_pretrained(source, trust_remote_code=True)
        if os.environ.get("V2V_NO_OFFLOAD") == "1":
            self.pipe.to("cuda", dtype=torch.bfloat16)
        else:
            self.pipe.enable_model_cpu_offload()

    def generate_video(self, video_path, prompt, output_path: Path, *, reference_image_path=None, num_frames=81, max_pixels=921600, num_inference_steps=50, guidance_scale=5.0, seed=0, fps=15) -> Dict[str, Any]:
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video, load_video
        source = require_file(video_path, "source video")
        self._load()
        frames = load_video(str(source))[:num_frames]
        if not frames:
            raise ValueError(f"No frames decoded from {source}")
        width, height = frames[0].size
        scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
        width = max(16, int(width * scale) // 16 * 16)
        height = max(16, int(height * scale) // 16 * 16)
        frames = [frame.resize((width, height), Image.Resampling.LANCZOS) for frame in frames]
        reference = None
        if reference_image_path:
            reference = [Image.open(require_file(reference_image_path, "reference image")).convert("RGB")]
        result = self.pipe(prompt=prompt, source_video=frames, ref_image=reference, height=height, width=width, num_frames=len(frames), num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed, tiled=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result, str(output_path), fps=fps)
        return {"checkpoint": str(self.checkpoint if self.checkpoint.is_dir() else self.model), "reference_image": str(reference_image_path) if reference_image_path else None, "num_frames": len(frames), "height": height, "width": width, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "seed": seed, "fps": fps}


class KiwiWrapper(ModelWrapper):
    def __init__(self, model="linyq/kiwi-edit-5b-instruct-reference-diffusers", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = KiwiService(model)

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
        return success_result(self.model, text_prompt, start, output, video_path, "kiwi", metadata)
