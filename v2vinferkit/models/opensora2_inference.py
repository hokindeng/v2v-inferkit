"""
Open-Sora 2.0 (11B) video extension, run in-process with the official sampling code.

Continuation from the END of the input clip:
- the input is resampled to 24 fps, resized-and-center-cropped to the 256px bucket closest
  to its aspect ratio, and its last 33 frames (32 + 1 for the causal VAE = 9 latent frames,
  the official minimum for v2v) become the reference clip; clips shorter than that are
  padded by holding their first frame (recorded as `padded_frames`);
- sampling uses the official `api_fn` with cond_type `v2v_head`: the 9 reference latents
  are given as masked channel conditioning at the START of the generated video, and the
  model generates what follows. (Upstream `v2v_tail` places the reference at the end of
  the output, i.e. it generates what came before; feeding the input's last frames to
  `v2v_head` is the forward continuation of the input.)
- generated length = 33 + 4k frames, covering the target clip's duration (ground_truth.mp4)
  up to the model's 129-frame limit (96 new frames = 4.0 s at 24 fps; `capped: true` when
  the target is longer);
- the 33 reproduced reference frames are dropped and the continuation is trimmed to the
  target duration, saved at 24 fps without audio.

Settings follow configs/diffusion/inference/256px.py: 50 steps, guidance 7.5, image guidance
3.0 with oscillation, flow shift on, prompt suffixed with "24 FPS." and "4 motion score."
(upstream defaults; no LLM prompt refinement). T5 shardformer (a fused-kernel speed-up that
needs colossalai's shard runtime) is off. Seed 42 seeds the initial noise.
Models stay loaded on the Service object across tasks (V2V_IN_PROCESS=1).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_dir, require_file, success_result, weights_path

FPS = 24
REF_FRAMES = 33  # 32 + 1 (causal VAE) = 9 latent frames, official v2v minimum
MAX_FRAMES = 129  # 4k+1, model limit
RESOLUTION = "256px"
MOTION_SCORE = "4"


def _probe(path: Union[str, Path]) -> Dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
         "stream=width,height,r_frame_rate,nb_read_frames:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    n = int(s.get("nb_read_frames") or 0)
    dur = float((d.get("format") or {}).get("duration") or (n / fps if fps else 0))
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "frames": n, "duration": dur}


def _stub_tensornvme():
    """opensora/utils/ckpt.py imports tensornvme (async checkpoint *writing*, training only) at
    module level; it is a compiled extension we do not need for inference."""
    try:
        import tensornvme.async_file_io  # noqa: F401
    except ImportError:
        import types
        pkg, mod = types.ModuleType("tensornvme"), types.ModuleType("tensornvme.async_file_io")
        mod.AsyncFileWriter = object
        pkg.async_file_io = mod
        sys.modules["tensornvme"], sys.modules["tensornvme.async_file_io"] = pkg, mod


class OpenSora2ExtendService:
    def __init__(self, model: str = "hpcai-tech/Open-Sora-v2"):
        self.model = model
        self.repo = Path(os.environ.get("OPENSORA_REPO_PATH") or str(repo_path("Open-Sora")))
        self.ckpt_dir = Path(os.environ.get("OPENSORA_WEIGHTS_PATH") or str(weights_path("Open-Sora-v2")))
        self.num_steps = int(os.environ.get("OPENSORA_STEPS", "50"))
        self.max_frames = int(os.environ.get("OPENSORA_MAX_FRAMES", str(MAX_FRAMES)))
        self.api_fn = None
        self.cfg = None

    def _load(self):
        if self.api_fn is not None:
            return
        require_dir(self.repo / "opensora", "Open-Sora repo (setup/install_model.sh --model open-sora-2.0-extend-v2v)")
        for name in ("Open_Sora_v2.safetensors", "hunyuan_vae.safetensors"):
            require_file(self.ckpt_dir / name, f"Open-Sora-v2 {name}")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        _stub_tensornvme()
        import torch
        from mmengine.config import Config
        from opensora.utils.sampling import prepare_api, prepare_models

        cfg = Config.fromfile(str(self.repo / "configs" / "diffusion" / "inference" / "256px.py"))
        cfg.model.from_pretrained = str(self.ckpt_dir / "Open_Sora_v2.safetensors")
        cfg.ae.from_pretrained = str(self.ckpt_dir / "hunyuan_vae.safetensors")
        cfg.t5.from_pretrained = str(self.ckpt_dir / "google" / "t5-v1_1-xxl")
        cfg.t5.shardformer = False
        cfg.clip.from_pretrained = str(self.ckpt_dir / "openai" / "clip-vit-large-patch14")
        torch.set_grad_enabled(False)
        model, model_ae, model_t5, model_clip, optional = prepare_models(cfg, "cuda", torch.bfloat16)
        self.cfg = cfg
        self.api_fn = prepare_api(model, model_ae, model_t5, model_clip, optional)

    @staticmethod
    def _bucket(w: int, h: int):
        from opensora.datasets.aspect import get_aspect_ratios_dict, get_closest_ratio, get_num_pexels_from_name
        ratios = get_aspect_ratios_dict(get_num_pexels_from_name(RESOLUTION), training=False)
        ar = get_closest_ratio(h, w, ratios)
        return ar

    def generate_video(self, video_path, prompt: str, output_path: Path, target_seconds: Optional[float],
                       seed: int = 42) -> Dict[str, Any]:
        import numpy as np
        import torch
        self._load()
        from opensora.utils.inference import add_fps_info_to_text, add_motion_score_to_text
        from opensora.utils.sampling import SamplingOption, sanitize_sampling_option

        info = _probe(video_path)
        ar = self._bucket(info["width"], info["height"])
        so = dict(self.cfg.sampling_option)
        so.update(resolution=RESOLUTION, aspect_ratio=ar, num_steps=self.num_steps, seed=seed)
        tgt_s = target_seconds if target_seconds else info["duration"]
        ext_frames = max(1, int(math.ceil(tgt_s * FPS - 1e-6)))
        total = REF_FRAMES + 4 * int(math.ceil(ext_frames / 4))
        capped = total > self.max_frames
        total = min(total, self.max_frames)
        so["num_frames"] = total
        opt = sanitize_sampling_option(SamplingOption(**so))
        H, W = opt.height, opt.width

        workdir = Path(tempfile.mkdtemp(prefix="opensora2ext_"))
        try:
            # input at 24 fps, resize-to-fill (bicubic) + center crop to the bucket, as upstream resize_crop
            vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,"
                  f"crop={W}:{H}")
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-vf", vf, "-an",
                                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
            if len(frames) == 0:
                raise RuntimeError("input decoded to zero frames")
            tail = frames[-REF_FRAMES:]
            padded = REF_FRAMES - len(tail)
            if padded > 0:
                tail = np.concatenate([np.repeat(tail[:1], padded, axis=0), tail], axis=0)
            ref = workdir / "ref.mp4"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
                            str(ref)], input=np.ascontiguousarray(tail).tobytes(), capture_output=True, check=True)

            text = add_fps_info_to_text([prompt], fps=FPS)
            text = add_motion_score_to_text(text, MOTION_SCORE)
            t_gen = time.time()
            x = self.api_fn(opt, "v2v_head", seed=seed, text=text, ref=[str(ref)],
                            patch_size=self.cfg.get("patch_size", 2), channel=self.cfg.model.in_channels).cpu()
            gen_seconds = time.time() - t_gen
            video = x[0]  # C T H W in [-1, 1]
            new = video[:, REF_FRAMES:REF_FRAMES + ext_frames]
            if new.shape[1] == 0:
                raise RuntimeError(f"model returned {video.shape[1]} frames, nothing after the reference")
            arr = ((new.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).numpy()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an",
                            str(output_path)], input=np.ascontiguousarray(arr).tobytes(), capture_output=True, check=True)
            out = _probe(output_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return {
            "checkpoint": f"{self.model} (Open_Sora_v2.safetensors)",
            "pipeline": "official opensora api_fn, cond_type v2v_head on the input's last 33 frames (forward continuation)",
            "resolution": RESOLUTION, "aspect_ratio": ar, "requested": f"{W}x{H}",
            "num_steps": opt.num_steps, "guidance": opt.guidance, "guidance_img": opt.guidance_img,
            "text_osci": opt.text_osci, "image_osci": opt.image_osci, "motion_score": MOTION_SCORE,
            "reference_frames": REF_FRAMES, "padded_frames": max(0, padded), "input_frames_24fps": int(len(frames)),
            "generated_frames": total, "prefix_frames_dropped": REF_FRAMES, "output_frames": int(new.shape[1]),
            "target_seconds": round(tgt_s, 3), "target_frames_24fps": ext_frames, "capped": capped,
            "fps": FPS, "seed": seed, "t5_shardformer": False, "sampling_seconds": round(gen_seconds, 1),
            "output_geometry": f"{out['width']}x{out['height']}@{out['fps']:.3f}fps/{out['frames']}f",
            "audio_stripped": True, "reference_video_used": False,
        }


class OpenSora2ExtendWrapper(ModelWrapper):
    def __init__(self, model: str = "hpcai-tech/Open-Sora-v2", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = OpenSora2ExtendService(model)

    def generate(self, image_path, text_prompt, duration: float = 5.0, output_filename: Optional[str] = None,
                 video_path: Optional[Union[str, Path]] = None, **kwargs) -> Dict[str, Any]:
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "open-sora-2.0-extend needs video_path")
        qd = kwargs.get("question_data") or {}
        gt = qd.get("ground_truth_video")
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            target_seconds = _probe(gt)["duration"] if gt and Path(gt).exists() else None
            metadata = self.service.generate_video(video_path, text_prompt, output, target_seconds,
                                                   seed=int(kwargs.get("seed", 42) or 42))
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "opensora2ext", metadata)
