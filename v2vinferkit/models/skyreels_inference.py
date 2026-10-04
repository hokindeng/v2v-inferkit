"""
SkyReels-V2 Diffusion Forcing 1.3B (540P) video continuation via diffusers.

Uses SkyReelsV2DiffusionForcingVideoToVideoPipeline (the diffusers port of the official
generate_video_df.py --video_path extension): the last `overlap_history` = 17 frames of the
input (resampled to 24 fps, the model's native rate) are VAE-encoded as clean prefix latents
and the model denoises the following latents with diffusion forcing; longer targets run in
97-frame windows that each re-use the last 17 frames as history. Settings follow the official
extension recipe (base_num_frames 97, overlap_history 17, addnoise_condition 20, synchronous
ar_step 0, shift 8, guidance 6, official negative prompt) with 30 sampling steps.

Resolution is 540p-class with the input's aspect ratio kept (pixel area of 960x544, sides
multiples of 16). The generated length covers the target clip's duration (ground_truth.mp4),
capped at 10 s (257 frames including the 17-frame prefix, SKYREELS_MAX_SECONDS). The pipeline
returns input + decoded(prefix + new); the input frames and the 17 reproduced prefix frames are
dropped, so the saved mp4 holds only the continuation, trimmed to the target duration, at 24
fps without audio. Weights are loaded once per Service and stay on the GPU between tasks.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, require_file, success_result, weights_path

FPS = 24
OVERLAP = 17
BASE_FRAMES = 97
NATIVE_AREA = 960 * 544
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，"
    "三条腿，背景人很多，倒着走"
)


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


def _dims(w: int, h: int) -> tuple:
    aspect = min(max(w / h, 0.5), 2.0)
    out_h = math.sqrt(NATIVE_AREA / aspect)
    out_w = out_h * aspect
    return max(256, int(round(out_w / 16)) * 16), max(256, int(round(out_h / 16)) * 16)


def _decode_tail(path: Union[str, Path], n: int, width: int, height: int):
    import numpy as np
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={width}:{height}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(frames) == 0:
        raise RuntimeError("input decoded to zero frames")
    tail = frames[-n:]
    if len(tail) < n:  # very short clip: repeat its first frame
        tail = np.concatenate([np.repeat(tail[:1], n - len(tail), axis=0), tail], axis=0)
    return tail


def _write_video(frames, output_path: Path, fps: int) -> None:
    t, h, w, _ = frames.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", "-an",
           str(output_path)]
    subprocess.run(cmd, input=frames.tobytes(), capture_output=True, check=True)


class SkyReelsDFExtendService:
    def __init__(self, model: str = "Skywork/SkyReels-V2-DF-1.3B-540P"):
        self.model = model
        self.checkpoint = Path(os.environ.get("SKYREELS_WEIGHTS_PATH")
                               or str(weights_path("SkyReels-V2-DF-1.3B-540P-Diffusers")))
        self.steps = int(os.environ.get("SKYREELS_STEPS", "30"))
        self.max_seconds = float(os.environ.get("SKYREELS_MAX_SECONDS", "10"))
        self.guidance = float(os.environ.get("SKYREELS_GUIDANCE", "6.0"))
        self.shift = float(os.environ.get("SKYREELS_SHIFT", "8.0"))
        self.addnoise = float(os.environ.get("SKYREELS_ADDNOISE", "20"))
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoencoderKLWan, SkyReelsV2DiffusionForcingVideoToVideoPipeline, UniPCMultistepScheduler
        source = str(self.checkpoint) if self.checkpoint.is_dir() else f"{self.model}-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32)
        pipe = SkyReelsV2DiffusionForcingVideoToVideoPipeline.from_pretrained(source, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=self.shift)
        pipe.set_progress_bar_config(disable=True)
        if os.environ.get("SKYREELS_OFFLOAD") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def generate_video(self, video_path, prompt: str, output_path: Path, target_seconds: Optional[float],
                       seed: int = 42) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image
        source = require_file(video_path, "source video")
        info = _probe(source)
        width, height = _dims(info["width"], info["height"])
        tgt_s = float(target_seconds) if target_seconds else info["duration"]
        need = max(1, int(math.ceil(tgt_s * FPS - 1e-6)))
        cap_new = ((int(self.max_seconds * FPS)) // 4) * 4
        capped = need > cap_new
        want = min(need, cap_new)
        num_frames = OVERLAP + int(math.ceil(want / 4)) * 4  # 4k+1 since OVERLAP = 4*4+1
        tail = _decode_tail(source, OVERLAP, width, height)
        frames_in = [Image.fromarray(f) for f in tail]

        self._load()
        torch.manual_seed(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        t0 = time.time()
        out = self.pipe(
            video=frames_in, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, height=height, width=width,
            num_frames=num_frames, num_inference_steps=self.steps, guidance_scale=self.guidance,
            generator=gen, overlap_history=OVERLAP, addnoise_condition=self.addnoise,
            base_num_frames=BASE_FRAMES, ar_step=0, fps=FPS, output_type="np",
        ).frames[0]
        gen_s = time.time() - t0
        # out = input tail (17) + decoded(prefix 17 + new); keep only the new frames
        new = np.asarray(out)[len(frames_in) + OVERLAP:][:want]
        if len(new) == 0:
            raise RuntimeError(f"pipeline returned {len(out)} frames, none after the prefix")
        new = (np.clip(new, 0, 1) * 255).round().astype(np.uint8)
        _write_video(new, output_path, FPS)
        torch.cuda.empty_cache()
        return {
            "checkpoint": f"{self.model} (diffusers format)",
            "pipeline": "diffusers SkyReelsV2DiffusionForcingVideoToVideoPipeline",
            "num_inference_steps": self.steps, "guidance_scale": self.guidance, "flow_shift": self.shift,
            "overlap_history": OVERLAP, "base_num_frames": BASE_FRAMES, "addnoise_condition": self.addnoise,
            "ar_step": 0, "fps": FPS, "resolution": f"{width}x{height}",
            "num_frames_incl_prefix": num_frames, "generated_frames": int(len(new)),
            "prefix_frames_dropped": OVERLAP, "target_seconds": round(tgt_s, 3), "target_frames": need,
            "capped": capped, "max_seconds": self.max_seconds, "generation_seconds": round(gen_s, 2),
            "seed": seed, "audio_stripped": True,
        }


class SkyReelsDFExtendWrapper(ModelWrapper):
    def __init__(self, model: str = "Skywork/SkyReels-V2-DF-1.3B-540P", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = SkyReelsDFExtendService(model)

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
            import traceback
            detail = f"{type(exc).__name__}: {exc} | {traceback.format_exc()[-1200:]}"
            return failed_result(self.model, text_prompt, start, detail, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "skyreelsdf", metadata)
