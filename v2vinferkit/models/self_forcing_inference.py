"""
Self-Forcing (Wan2.1-T2V-1.3B, chunk-wise causal, 4-step DMD) as a video continuation model.

Upstream has no video-continuation entry point, but its CausalInferencePipeline accepts an
`initial_latent` ("if num_input_frames is greater than 1, perform video extension"): the latent
frames are written into the KV cache as clean context and generation continues after them.

Here the input clip is resampled to 16 fps and resized to the model's fixed 832x480 grid (the
KV cache and RoPE layout are built for 60x104 latents, so other sizes are not supported; the
frame is stretched, not cropped, so the whole field of view is kept). Its last 9 frames are
encoded with the Wan VAE as a fresh clip (3 latent frames = one 3-latent block, matching how
the model sees the start of a clip), then N new latent frames are generated (N a multiple of
3, covering the target duration from ground_truth.mp4). Context + new latents are decoded
together (continuity at the seam) and the 9 reproduced context frames are dropped, so the
saved mp4 holds only the continuation, trimmed to the target duration, at 16 fps.

The released checkpoint was trained on 21 latent frames (81 frames, ~5 s). Up to 21 latent
frames in total (context + new, i.e. 4.5 s of new video) generation stays inside that horizon;
longer targets roll the KV cache over the last 21 latent frames (local_attn_size=21, the
upstream rolling mechanism; identical behaviour within the horizon) and are capped at 10 s
(SELF_FORCING_MAX_SECONDS). The output is rescaled to the input's aspect ratio (long side
832). Weights are loaded once per Service and stay on the GPU between tasks.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_file, success_result, weights_path

FPS = 16
MODEL_W, MODEL_H = 832, 480
LATENT_H, LATENT_W = 60, 104
BLOCK = 3  # num_frame_per_block of the released checkpoint
TRAINED_LATENTS = 21


def _probe(path: Union[str, Path]) -> Dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    n = int(s.get("nb_read_frames") or 0)
    dur = float((d.get("format") or {}).get("duration") or (n / fps if fps else 0))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "frames": n, "duration": dur}


def _decode_frames(path: Union[str, Path], fps: int, width: int, height: int):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={width}:{height}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8)
    return frames.reshape(-1, height, width, 3)


def _write_video(frames, output_path: Path, fps: int, out_w: int, out_h: int) -> None:
    t, h, w, _ = frames.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
           "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an", str(output_path)]
    subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=True)


def _output_dims(w: int, h: int, long_side: int = MODEL_W) -> tuple:
    if w >= h:
        return long_side, max(2, int(round(long_side * h / w / 2)) * 2)
    return max(2, int(round(long_side * w / h / 2)) * 2), long_side


class SelfForcingExtendService:
    def __init__(self):
        self.repo = repo_path("Self-Forcing", "SELF_FORCING_REPO_PATH")
        self.checkpoint = Path(os.environ.get("SELF_FORCING_CHECKPOINT")
                               or str(weights_path("Self-Forcing", "checkpoints", "self_forcing_dmd.pt")))
        self.config_path = os.environ.get("SELF_FORCING_CONFIG", "configs/self_forcing_dmd.yaml")
        self.context_latents = int(os.environ.get("SELF_FORCING_CONTEXT_LATENTS", "3"))
        self.max_seconds = float(os.environ.get("SELF_FORCING_MAX_SECONDS", "10"))
        self.local_attn = int(os.environ.get("SELF_FORCING_LOCAL_ATTN", str(TRAINED_LATENTS)))
        self.pipe = None
        self.ckpt_key = None
        if self.context_latents % BLOCK:
            raise ValueError(f"SELF_FORCING_CONTEXT_LATENTS must be a multiple of {BLOCK}")

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from omegaconf import OmegaConf
        require_file(self.checkpoint, "Self-Forcing checkpoint")
        base = self.repo / "wan_models" / "Wan2.1-T2V-1.3B"
        if not (base / "Wan2.1_VAE.pth").exists():
            raise FileNotFoundError(f"{base} missing — run setup/install_model.sh --model self-forcing-extend-v2v")
        repo = str(self.repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        cwd = os.getcwd()
        os.chdir(repo)  # upstream loads the base model from the relative path wan_models/...
        try:
            from pipeline.causal_inference import CausalInferencePipeline
            config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(self.config_path))
            config.model_kwargs = OmegaConf.merge(config.get("model_kwargs", {}),
                                                  {"local_attn_size": self.local_attn, "sink_size": 0})
            torch.set_grad_enabled(False)
            pipe = CausalInferencePipeline(config, device=torch.device("cuda"))
            state = torch.load(str(self.checkpoint), map_location="cpu", weights_only=False)
            self.ckpt_key = "generator_ema" if "generator_ema" in state else "generator"
            pipe.generator.load_state_dict(state[self.ckpt_key])
            del state
            pipe = pipe.to(dtype=torch.bfloat16)
            pipe.text_encoder.to("cuda")
            pipe.generator.to("cuda")
            pipe.vae.to("cuda")
        finally:
            os.chdir(cwd)
        self.pipe = pipe

    def generate_video(self, video_path, prompt: str, output_path: Path, target_seconds: Optional[float],
                       seed: int = 42) -> Dict[str, Any]:
        import numpy as np
        import torch
        source = require_file(video_path, "source video")
        info = _probe(source)
        tgt_s = float(target_seconds) if target_seconds else info["duration"]
        need = max(1, int(math.ceil(tgt_s * FPS - 1e-6)))
        cap = int(self.max_seconds * FPS)
        capped = need > cap
        want = min(need, cap)
        n_new = int(math.ceil(want / 4 / BLOCK)) * BLOCK
        while n_new * 4 > cap and n_new > BLOCK:
            n_new -= BLOCK
        ctx_frames = 1 + 4 * (self.context_latents - 1)

        frames = _decode_frames(source, FPS, MODEL_W, MODEL_H)
        if len(frames) == 0:
            raise RuntimeError("input decoded to zero frames")
        tail = frames[-ctx_frames:]
        if len(tail) < ctx_frames:  # very short clip: repeat its first frame
            tail = np.concatenate([np.repeat(tail[:1], ctx_frames - len(tail), axis=0), tail], axis=0)

        self._load()
        pipe = self.pipe
        torch.manual_seed(seed)
        pixel = torch.from_numpy(np.ascontiguousarray(tail)).to("cuda").permute(3, 0, 1, 2)  # C,T,H,W
        pixel = (pixel.float() / 127.5 - 1.0).unsqueeze(0).to(torch.bfloat16)
        initial_latent = pipe.vae.encode_to_latent(pixel).to(device="cuda", dtype=torch.bfloat16)
        if initial_latent.shape[1] != self.context_latents:
            raise RuntimeError(f"expected {self.context_latents} context latents, got {initial_latent.shape[1]}")
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noise = torch.randn([1, n_new, 16, LATENT_H, LATENT_W], device="cuda", dtype=torch.bfloat16, generator=gen)
        t0 = time.time()
        video = pipe.inference(noise=noise, text_prompts=[prompt], initial_latent=initial_latent)
        pipe.vae.model.clear_cache()
        gen_s = time.time() - t0
        new = video[0, ctx_frames:]  # T,C,H,W in [0,1]; drop the reproduced context
        new = new[:want]
        out = (new.permute(0, 2, 3, 1).float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        ow, oh = _output_dims(info["width"], info["height"])
        _write_video(out, output_path, FPS, ow, oh)
        del video, new, noise, initial_latent, pixel
        torch.cuda.empty_cache()
        total_latents = self.context_latents + n_new
        return {
            "checkpoint": f"gdhe17/Self-Forcing checkpoints/{self.checkpoint.name} ({self.ckpt_key})",
            "base_model": "Wan-AI/Wan2.1-T2V-1.3B",
            "pipeline": "CausalInferencePipeline + initial_latent (input tail as clean KV context)",
            "denoising_steps": 4, "fps": FPS, "model_resolution": f"{MODEL_W}x{MODEL_H}",
            "output_resolution": f"{ow}x{oh}", "input_resize": "stretched to 832x480 (no crop)",
            "context_frames": ctx_frames, "context_latents": self.context_latents,
            "generated_latents": n_new, "generated_frames": int(out.shape[0]),
            "target_seconds": round(tgt_s, 3), "target_frames": need, "capped": capped,
            "max_seconds": self.max_seconds,
            "rolling_kv_cache": total_latents > TRAINED_LATENTS, "local_attn_size": self.local_attn,
            "generation_seconds": round(gen_s, 2), "seed": seed, "audio_stripped": True,
        }


class SelfForcingExtendWrapper(ModelWrapper):
    def __init__(self, model: str = "gdhe17/Self-Forcing", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = SelfForcingExtendService()

    def generate(self, image_path, text_prompt, duration: Optional[float] = None,
                 output_filename: Optional[str] = None, video_path: Optional[Union[str, Path]] = None,
                 **kwargs) -> Dict[str, Any]:
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        qd = kwargs.get("question_data") or {}
        gt = qd.get("ground_truth_video")
        target_seconds = None
        if duration:  # explicit --duration override
            target_seconds = float(duration)
        elif gt and Path(gt).exists():
            target_seconds = _probe(gt)["duration"]
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            metadata = self.service.generate_video(video_path, text_prompt, output, target_seconds,
                                                   seed=int(kwargs.get("seed", 42) or 42))
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "selfforcing", metadata)
