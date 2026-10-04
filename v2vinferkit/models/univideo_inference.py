"""UniVideo instruction-guided video editing (official v2v_edit task, "hidden" variant).

Upstream: https://github.com/KlingTeam/UniVideo (Apache-2.0). The pipeline encodes the
instruction and two frames of the source with Qwen2.5-VL-7B, concatenates the source
latents to the noise latents on the time axis, and denoises with a HunyuanVideo MMDiT
using 3-pass guidance (text 7.0, image 2.0).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from .base import ModelWrapper
from .local_utils import LocalInferenceError, failed_result, repo_path, require_file, success_result, weights_path

VARIANT = "univideo_qwen2p5vl7b_hidden_hunyuanvideo"
QWEN_TXT_DIM = 3584
# Upstream default negative prompt (univideo_inference.py).
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, oversharpening, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, walking backwards, computer-generated environment, weak dynamics, distorted and "
    "erratic motions, unstable framing and a disorganized composition."
)


def _sample_indices(src_count: int, max_frames: int, temporal_unit: int = 4):
    """Evenly spaced frame indices across the whole clip; count is 1 + k*temporal_unit."""
    count = max(1, (min(src_count, max_frames) - 1) // temporal_unit * temporal_unit + 1)
    return [round(i * (src_count - 1) / max(count - 1, 1)) for i in range(count)]


def _read_cond_video_even(video_path, height, width, num_frames, vae_spatial_scale_factor=8,
                          spatial_patch_size=2, vae_temporal_scale_factor=4, temporal_patch_size=1):
    """Drop-in for upstream utils.read_and_preprocess_cond_video.

    Same area-preserving resize and divisible crop, but frames are sampled evenly across
    the whole clip instead of taking the first num_frames.
    """
    import cv2
    import decord
    import torch

    unit = vae_spatial_scale_factor * spatial_patch_size
    cap = cv2.VideoCapture(video_path)
    orig_h, orig_w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    if orig_h <= 0 or orig_w <= 0:
        raise RuntimeError(f"Could not read video resolution: {video_path}")
    aspect = orig_w / orig_h
    resize_h = math.sqrt(height * width / aspect)
    resize_w = int(round(resize_h * aspect))
    resize_h = int(round(resize_h))
    decord.bridge.set_bridge("torch")
    reader = decord.VideoReader(video_path, ctx=decord.cpu(0), height=resize_h, width=resize_w)
    idx = _sample_indices(len(reader), int(num_frames), vae_temporal_scale_factor * temporal_patch_size)
    frames = reader.get_batch(idx).float().permute(0, 3, 1, 2)
    h, w = frames.shape[2], frames.shape[3]
    crop_h, crop_w = (h // unit) * unit, (w // unit) * unit
    frames = frames[:, :, :crop_h, :crop_w].contiguous()
    uint8 = frames.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    meta = {"fps": float(reader.get_avg_fps()), "original_resolution": (int(orig_h), int(orig_w)),
            "post_divisible_resolution": (crop_h, crop_w), "used_num_frames": len(idx)}
    return (frames - 127.5) / 127.5, uint8, meta


class UniVideoService:
    def __init__(self, model="KlingTeam/UniVideo"):
        self.model = model
        self.repo = repo_path("UniVideo", "UNIVIDEO_REPO")
        self.ckpt = Path(os.environ.get("UNIVIDEO_CKPT") or str(weights_path("UniVideo", VARIANT, "model.ckpt")))
        self.hunyuan = Path(os.environ.get("UNIVIDEO_HUNYUAN_PATH") or str(weights_path("HunyuanVideo")))
        self.qwen = Path(os.environ.get("UNIVIDEO_QWEN_PATH") or str(weights_path("Qwen2.5-VL-7B-Instruct")))
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        import yaml
        from accelerate import init_empty_weights

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformer_univideo_hunyuan_video import HunyuanVideoTransformer3DModel, TwoLayerMLP
        from mllm_encoder import MLLMInContext, MLLMInContextConfig
        import pipeline_univideo
        from pipeline_univideo import UniVideoPipeline, UniVideoPipelineConfig

        pipeline_univideo.read_and_preprocess_cond_video = _read_cond_video_even
        ckpt = require_file(self.ckpt, "UniVideo checkpoint")
        with open(self.repo / "configs" / f"{VARIANT}.yaml") as f:
            raw = yaml.safe_load(f)
        raw["mllm_config"]["mllm_id"] = str(self.qwen)
        raw["pipeline_config"]["hunyuan_model_id"] = str(self.hunyuan)

        mllm = MLLMInContext(MLLMInContextConfig(**raw["mllm_config"]))
        mllm.requires_grad_(False).eval()
        vae = AutoencoderKLHunyuanVideo.from_pretrained(str(self.hunyuan), subfolder="vae").eval()

        # Upstream builds the HunyuanVideo transformer from its pretrained weights, swaps in
        # a fresh Qwen connector, then overwrites every parameter from the UniVideo
        # checkpoint. Building from config on the meta device and assigning the checkpoint
        # gives the same weights without reading the 25 GB base transformer.
        config = HunyuanVideoTransformer3DModel.load_config(str(self.hunyuan), subfolder="transformer")
        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModel.from_config(config, text_embed_dim=QWEN_TXT_DIM)
            transformer.qwen_project_in = TwoLayerMLP(QWEN_TXT_DIM, QWEN_TXT_DIM * 4, 4096)
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True, mmap=True)
        except RuntimeError:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        state = {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in state.items()}
        missing, _unexpected = transformer.load_state_dict(state, strict=False, assign=True)
        still_meta = [n for n, p in transformer.named_parameters() if p.is_meta]
        if missing or still_meta:
            raise LocalInferenceError(f"UniVideo checkpoint does not cover the transformer: {sorted(set(missing) | set(still_meta))[:10]}")
        transformer.eval()

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(self.hunyuan), subfolder="scheduler")
        self.pipe = UniVideoPipeline(
            transformer=transformer, vae=vae, scheduler=scheduler, mllm_encoder=mllm,
            univideo_config=UniVideoPipelineConfig(**raw["pipeline_config"]),
        ).to(device="cuda", dtype=torch.bfloat16)

    def generate_video(self, video_path, prompt, output_path: Path, *, max_frames=61, height=480, width=854,
                       num_inference_steps=30, guidance_scale=7.0, image_guidance_scale=2.0, timestep_shift=7.0,
                       seed=42) -> Dict[str, Any]:
        import decord
        import numpy as np
        from diffusers.utils import export_to_video

        source = str(require_file(video_path, "source video"))
        probe = decord.VideoReader(source, ctx=decord.cpu(0))
        src_count, src_fps = len(probe), float(probe.get_avg_fps())
        del probe
        if src_count < 1:
            raise LocalInferenceError(f"No frames decoded from {source}")
        count = len(_sample_indices(src_count, max_frames))
        self._load()
        out = self.pipe(
            prompts=[prompt], negative_prompt=NEGATIVE_PROMPT, cond_video_path=source,
            height=height, width=width, num_frames=count, num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, seed=seed,
            timestep_shift=timestep_shift, task="v2v_edit",
        ).frames[0]
        frames = np.asarray(out, dtype=np.float32)
        # Output spans the source duration: the count frames were sampled across the clip.
        fps = max(1.0, frames.shape[0] * src_fps / src_count) if src_fps > 0 else 24.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(list(frames), str(output_path), fps=fps)
        return {
            "checkpoint": f"{self.model}/{VARIANT}", "task": "v2v_edit", "source_frames": src_count,
            "source_fps": src_fps, "num_frames": int(frames.shape[0]), "height": int(frames.shape[1]),
            "width": int(frames.shape[2]), "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale, "image_guidance_scale": image_guidance_scale,
            "timestep_shift": timestep_shift, "seed": seed, "fps": fps, "frame_sampling": "even_whole_clip",
        }


class UniVideoWrapper(ModelWrapper):
    def __init__(self, model="KlingTeam/UniVideo", output_dir="./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = UniVideoService(model)

    def generate(self, image_path, text_prompt, duration=5.0, output_filename=None, video_path=None, **kwargs):
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        kwargs.pop("question_data", None)
        target_frames = kwargs.pop("num_frames", None)
        allowed = {"max_frames", "height", "width", "num_inference_steps", "guidance_scale",
                   "image_guidance_scale", "timestep_shift", "seed"}
        overrides = {k: v for k, v in kwargs.items() if k in allowed}
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, **overrides)
        except Exception as exc:
            return failed_result(self.model, text_prompt, start, exc, video_path)
        metadata["target_num_frames"] = target_frames
        metadata["reference_used"] = False  # v2v_edit takes no reference video
        return success_result(self.model, text_prompt, start, output, video_path, "univideo", metadata)
