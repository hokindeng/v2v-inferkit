"""Lucy-Edit 1.1 Dev instruction-guided video editing via Diffusers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, require_file, success_result, weights_path


class LucyService:
    def __init__(self, model="decart-ai/Lucy-Edit-1.1-Dev"):
        self.model = model
        self.checkpoint = Path(os.environ.get("LUCY_WEIGHTS_PATH") or str(weights_path("Lucy-Edit-1.1-Dev")))
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan, LucyEditPipeline
        source = str(self.checkpoint) if self.checkpoint.is_dir() else self.model
        vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = LucyEditPipeline.from_pretrained(source, vae=vae, torch_dtype=torch.bfloat16)
        if os.environ.get("V2V_NO_OFFLOAD") == "1":
            self.pipe.to("cuda")
        else:
            self.pipe.enable_model_cpu_offload()

    def generate_video(self, video_path, prompt, output_path: Path, *, num_frames=81, height=480, width=832, num_inference_steps=50, guidance_scale=5.0, seed=42, negative_prompt="", fps=24) -> Dict[str, Any]:
        import torch
        from PIL import Image
        from diffusers.utils import export_to_video, load_video
        source = require_file(video_path, "source video")
        frames = load_video(str(source))[:num_frames]
        if not frames:
            raise ValueError(f"No frames decoded from {source}")
        count = ((len(frames) - 1) // 4) * 4 + 1
        frames = [frame.resize((width, height), Image.Resampling.LANCZOS) for frame in frames[:count]]
        self._load()
        result = self.pipe(prompt=prompt, video=frames, negative_prompt=negative_prompt, height=height, width=width, num_frames=len(frames), num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=torch.Generator().manual_seed(seed)).frames[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result, str(output_path), fps=fps)
        return {"checkpoint": str(self.checkpoint if self.checkpoint.is_dir() else self.model), "num_frames": len(frames), "height": height, "width": width, "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale, "seed": seed, "fps": fps}


class LucyWrapper(ModelWrapper):
    def __init__(self, model="decart-ai/Lucy-Edit-1.1-Dev", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = LucyService(model)

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
        return success_result(self.model, text_prompt, start, output, video_path, "lucy", metadata)
