"""Wan2.1-VACE-14B video-to-video inference (open-source, local GPU).

Runs the diffusers WanVACEPipeline rather than the ali-vilab/VACE repo CLI:
the repo CLI keeps DiT + doubled VACE context latents + CFG batch resident and
OOMs 14B on 48GB-class GPUs even with --offload_model/--t5_cpu (VACE issues
#55/#56), while enable_model_cpu_offload() bounds peak memory near the 28GB
bf16 transformer (~30-35GB at 480p/81 frames).

V2V semantics: source frames + an all-white mask = "regenerate every pixel,
conditioned on the source video" — equivalent to the repo's raw --src_video
path (no preprocessing). NOTE diffusers mask colors are inverted vs the repo:
white = generate, black = keep.

Checkpoint: Wan-AI/Wan2.1-VACE-14B-diffusers (the raw-format Wan2.1-VACE-14B
checkpoint has no model_index.json and cannot be loaded by diffusers).

Runs inside its model venv (envs/wan-vace-14b-v2v) via _subprocess_worker.py.

VACE prompt guidance (UserGuide §3.1): descriptive prompts of the desired
output work much better than imperative edit instructions. Benchmark prompts
are passed through as-is for now — instruction→description rewriting is a
team-level design decision.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper

# 4n+1 frame counts, 16 fps, ~5s max — Wan2.1 VACE native settings
_MAX_FRAMES = 81
_OUTPUT_FPS = 16
_SIZES = {"landscape": (832, 480), "portrait": (480, 832)}


def _default_weights_dir() -> Path:
    root = Path(__file__).parent.parent.parent
    weights = os.environ.get("V2V_WEIGHTS_DIR", str(root / "weights"))
    return Path(weights) / "Wan2.1-VACE-14B-diffusers"


class VaceService:
    """Loads WanVACEPipeline once and serves v2v generations."""

    def __init__(self, model: str = "Wan-AI/Wan2.1-VACE-14B-diffusers"):
        self.model_id = model
        self.pipe = None

    def _load_model(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan, WanVACEPipeline, UniPCMultistepScheduler

        local_dir = _default_weights_dir()
        source = str(local_dir) if local_dir.exists() else self.model_id

        # fp32 VAE is deliberate (quality); transformer in bf16
        vae = AutoencoderKLWan.from_pretrained(
            source, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanVACEPipeline.from_pretrained(
            source, vae=vae, torch_dtype=torch.bfloat16
        )
        # flow_shift 3.0 for 480p (5.0 for 720p)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=3.0
        )
        if os.environ.get("V2V_NO_OFFLOAD") == "1":
            # 80GB-class GPUs: whole pipeline resident, much faster
            self.pipe.to("cuda")
        else:
            # Key to fitting 14B on a 48GB GPU
            self.pipe.enable_model_cpu_offload()

    def _prepare_frames(self, video_path: Union[str, Path], num_frames: Optional[int]):
        """Load, orient, resize (aspect-preserving + center crop), cap to 4n+1."""
        from PIL import Image
        from diffusers.utils import load_video

        frames = load_video(str(video_path))
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")

        w0, h0 = frames[0].size
        width, height = _SIZES["landscape" if w0 >= h0 else "portrait"]

        n = min(len(frames), num_frames or _MAX_FRAMES, _MAX_FRAMES)
        n = ((n - 1) // 4) * 4 + 1  # snap down to 4n+1
        frames = frames[:n]

        scale = max(width / w0, height / h0)
        rw, rh = round(w0 * scale), round(h0 * scale)
        left, top = (rw - width) // 2, (rh - height) // 2

        prepared = []
        for f in frames:
            f = f.resize((rw, rh), Image.Resampling.LANCZOS)
            prepared.append(f.crop((left, top, left + width, top + height)))

        mask = [Image.new("L", (width, height), 255)] * len(prepared)  # white = generate
        return prepared, mask, width, height

    def generate_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        import torch
        from diffusers.utils import export_to_video

        start_time = time.time()
        self._load_model()

        frames, mask, width, height = self._prepare_frames(video_path, num_frames)

        result = self.pipe(
            video=frames,
            mask=mask,
            prompt=text_prompt,
            height=height,
            width=width,
            num_frames=len(frames),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result.frames[0], str(output_path), fps=_OUTPUT_FPS)

        return {
            "video_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "status": "success",
            "metadata": {
                "width": width,
                "height": height,
                "num_frames": len(frames),
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "fps": _OUTPUT_FPS,
            },
        }


class VaceWrapper(ModelWrapper):
    """v2v-inferkit wrapper for Wan2.1-VACE-14B (diffusers pipeline)."""

    def __init__(
        self,
        model: str = "Wan-AI/Wan2.1-VACE-14B-diffusers",
        output_dir: str = "./outputs",
        **kwargs,
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = VaceService(model=model)

    def generate(
        self,
        image_path: Optional[Union[str, Path]],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if video_path is None:
            return {
                "success": False,
                "video_path": None,
                "error": "wan-vace-14b-v2v is video-to-video only: video_path is required",
                "duration_seconds": 0.0,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt},
            }

        num_frames = kwargs.pop("num_frames", None)
        kwargs.pop("question_data", None)
        output_path = self.output_dir / (output_filename or "video.mp4")

        try:
            result = self.service.generate_video(
                video_path=video_path,
                text_prompt=text_prompt,
                output_path=output_path,
                num_frames=num_frames,
                **kwargs,
            )
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "modality": "v2v",
                    "video_path_input": str(video_path),
                },
            }

        return {
            "success": True,
            "video_path": result["video_path"],
            "error": None,
            "duration_seconds": result["duration_seconds"],
            "generation_id": f"vace_{int(start_time)}",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v",
                "video_path_input": str(video_path),
                **result["metadata"],
            },
        }
