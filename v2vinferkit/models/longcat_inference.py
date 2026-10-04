"""
LongCat-Video (13.6B, MIT) video continuation, loaded in-process.

The input clip is resampled to 15 fps (the model's rate) and its last 13 frames condition
`LongCatVideoPipeline.generate_vc` with the cfg-step distill LoRA (16 steps, guidance 1.0,
no enhance_hf), 480p aspect bucket chosen by the pipeline from the input's aspect ratio.
The target length comes from the task's ground_truth.mp4 (or the input's duration when
absent): N = ceil(seconds * 15) new frames, capped at LONGCAT_MAX_NEW_FRAMES (default
150 = 10 s, `capped` recorded). One window adds up to 80 frames (93 total, 13 reproduced);
longer targets chain windows, each conditioned on the last 13 frames of the previous one
(as run_demo_long_video.py does). Only newly generated frames are saved, at 15 fps,
without audio.

The pipeline (tokenizer, UMT5 text encoder, VAE, DiT in bf16, distill LoRA enabled) is
loaded once on the Service and reused across tasks (V2V_IN_PROCESS=1). Single GPU, no
torch.distributed: context-parallel globals default to size 1.

Weights/repo: setup/models/longcat-video-extend-v2v (weights/LongCat-Video, repos/LongCat-Video).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import ModelWrapper
from .local_utils import failed_result, repo_path, require_file, success_result, weights_path

FPS = 15.0
COND_FRAMES = 13
WINDOW_FRAMES = 93  # 13 conditioning + 80 new
STEPS = 16


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


def _decode_15fps(path: Path, workdir: Path) -> List[Any]:
    from PIL import Image
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", f"fps={FPS}", "-f", "image2",
                    str(workdir / "f_%05d.png")], capture_output=True, check=True)
    files = sorted(workdir.glob("f_*.png"))
    if not files:
        raise RuntimeError(f"no frames decoded from {path}")
    return [Image.open(p).convert("RGB") for p in files[-COND_FRAMES:]]


class LongCatExtendService:
    def __init__(self, model: str = "meituan-longcat/LongCat-Video"):
        self.model = model
        self.checkpoint = Path(os.environ.get("LONGCAT_WEIGHTS_PATH") or str(weights_path("LongCat-Video")))
        self.repo = repo_path("LongCat-Video", "LONGCAT_REPO_PATH")
        self.max_new = int(os.environ.get("LONGCAT_MAX_NEW_FRAMES", "150"))
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        for req, hint in [(self.repo / "longcat_video", "repo clone"), (self.checkpoint / "dit", "checkpoint"),
                          (self.checkpoint / "lora" / "cfg_step_lora.safetensors", "distill LoRA")]:
            if not req.exists():
                raise FileNotFoundError(f"{req} missing ({hint}) — run setup/install_model.sh --model longcat-video-extend-v2v")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel
        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

        ck = str(self.checkpoint)
        bf16 = torch.bfloat16
        tokenizer = AutoTokenizer.from_pretrained(ck, subfolder="tokenizer", torch_dtype=bf16)
        text_encoder = UMT5EncoderModel.from_pretrained(ck, subfolder="text_encoder", torch_dtype=bf16)
        vae = AutoencoderKLWan.from_pretrained(ck, subfolder="vae", torch_dtype=bf16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ck, subfolder="scheduler", torch_dtype=bf16)
        dit = LongCatVideoTransformer3DModel.from_pretrained(ck, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=bf16)
        pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler, dit=dit)
        pipe.to("cuda")
        pipe.dit.load_lora(str(self.checkpoint / "lora" / "cfg_step_lora.safetensors"), "cfg_step_lora")
        pipe.dit.enable_loras(["cfg_step_lora"])
        self.pipe = pipe

    def generate_video(self, video_path, prompt: str, output_path: Path, target_seconds: Optional[float],
                       seed: int = 42) -> Dict[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        source = require_file(video_path, "source video")
        info = _probe(source)
        tgt_s = target_seconds if target_seconds else info["duration"]
        wanted = max(1, int(math.ceil(tgt_s * FPS)))
        new_total = min(wanted, self.max_new)
        t_load = time.time()
        self._load()
        load_s = time.time() - t_load

        with tempfile.TemporaryDirectory(prefix="longcat_") as tmp:
            cond = _decode_15fps(source, Path(tmp))
        cond_available = len(cond)
        while len(cond) < COND_FRAMES:  # very short inputs: repeat the first frame
            cond = [cond[0]] + cond

        generator = torch.Generator(device="cuda").manual_seed(seed)
        new_frames: List[np.ndarray] = []
        windows = []
        while len(new_frames) < new_total:
            remaining = new_total - len(new_frames)
            n_new = min(WINDOW_FRAMES - COND_FRAMES, 4 * math.ceil(remaining / 4))  # num_frames must be 4k+1
            out = self.pipe.generate_vc(
                video=cond, prompt=prompt, resolution="480p", num_frames=COND_FRAMES + n_new,
                num_cond_frames=COND_FRAMES, num_inference_steps=STEPS, use_distill=True, guidance_scale=1.0,
                generator=generator, use_kv_cache=True, offload_kv_cache=False, enhance_hf=False,
            )[0]
            frames = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)  # (T, H, W, C)
            gen = frames[COND_FRAMES:]
            windows.append(int(gen.shape[0]))
            new_frames.extend(list(gen))
            cond = [Image.fromarray(f) for f in frames[-COND_FRAMES:]]
            del out
        new_frames = new_frames[:new_total]
        torch.cuda.empty_cache()

        h, w = new_frames[0].shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
             "-i", "-", "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(output_path)],
            input=np.stack(new_frames).tobytes(), capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='replace')[-600:]}")
        out_info = _probe(output_path)
        return {
            "checkpoint": str(self.checkpoint), "lora": "cfg_step_lora (distill)", "num_inference_steps": STEPS,
            "guidance_scale": 1.0, "enhance_hf": False, "resolution": f"{w}x{h} (480p bucket)",
            "input_geometry": f"{info['width']}x{info['height']}@{info['fps']:.3f}fps/{info['frames']}f",
            "cond_frames": COND_FRAMES, "cond_frames_available_at_15fps": cond_available, "fps": FPS,
            "target_seconds": round(tgt_s, 3), "new_frames_wanted": wanted, "new_frames": new_total,
            "windows_new_frames": windows, "capped": new_total < wanted, "seed": seed,
            "model_load_seconds": round(load_s, 1), "audio_stripped": True,
            "output_geometry": f"{out_info['width']}x{out_info['height']}@{out_info['fps']:.3f}fps/{out_info['frames']}f",
        }


class LongCatExtendWrapper(ModelWrapper):
    def __init__(self, model: str = "meituan-longcat/LongCat-Video", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = LongCatExtendService(model)

    def generate(self, image_path, text_prompt, duration: float = 5.0, output_filename: Optional[str] = None,
                 video_path: Optional[Union[str, Path]] = None, **kwargs) -> Dict[str, Any]:
        start = time.time()
        if video_path is None:
            return failed_result(self.model, text_prompt, start, "video_path is required")
        qd = kwargs.get("question_data") or {}
        gt = qd.get("ground_truth_video")
        output = self.output_dir / (output_filename or "video.mp4")
        try:
            target_seconds = _probe(gt)["duration"] if gt and Path(gt).exists() else None
            metadata = self.service.generate_video(video_path, text_prompt, output, target_seconds,
                                                   seed=int(kwargs.get("seed", 42) or 42))
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.model, text_prompt, start, exc, video_path)
        return success_result(self.model, text_prompt, start, output, video_path, "longcat", metadata)
