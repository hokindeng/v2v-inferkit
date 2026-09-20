"""
LTX-2.3 video extension with the local distilled checkpoint (no hosted API).

Continuation by keyframe conditioning: the input clip is resampled to 24 fps, its last
9 frames are written as stills and passed to `python -m ltx_pipelines.distilled` as
`--image <frame> <idx> 1.0` for idx 0..8 (one latent group at LTX's 8x temporal
compression), and the pipeline generates 9 + N frames at 24 fps where N covers the
target clip's duration (ground_truth.mp4; rounded up to 8k+1, capped by
LTX2_EXTEND_MAX_FRAMES, default 257). The first 9 output frames are the reproduced
prefix and are dropped, so the saved mp4 is the extension only (nothing to trim before
scoring). Output resolution follows the input's aspect ratio with the long side at
LTX2_EXTEND_STAGE1_LONG_SIDE (default 768; the two-stage pipeline renders at half and
upsamples x2 to this size), rounded to multiples of 32. Audio is stripped from the output.

Weights and repo are the ones ltx-2.3-dev-v2v installs (setup/models/ltx-2.3-dev-v2v);
the venv is shared (catalog venv_id = ltx-2.3-dev-v2v).
"""
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper
from .ltx23_inference import _kit_root, _repo_dir, _weights_dir

PREFIX_FRAMES = 9
FPS = 24.0


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


def _stage1_dims(w: int, h: int, long_side: int) -> tuple:
    scale = long_side / max(w, h)
    sw, sh = max(256, int(round(w * scale / 32)) * 32), max(256, int(round(h * scale / 32)) * 32)
    return sw, sh


class Ltx23ExtendService:
    def __init__(self):
        self.repo_path = Path(os.environ.get("LTX2_REPO_PATH") or str(_repo_dir()))
        self.weights_path = Path(os.environ.get("LTX2_WEIGHTS_PATH") or str(_weights_dir()))
        self.gemma_root = Path(os.environ.get("LTX2_GEMMA_ROOT", str(self.weights_path / "gemma-3-12b")))
        self.checkpoint = os.environ.get("LTX2_IC_LORA_CHECKPOINT", "ltx-2.3-22b-distilled-1.1.safetensors")
        self.upsampler = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        self.max_frames = int(os.environ.get("LTX2_EXTEND_MAX_FRAMES", "257"))
        self.long_side = int(os.environ.get("LTX2_EXTEND_STAGE1_LONG_SIDE", "768"))

    def generate_video(self, video_path, text_prompt, output_path: Path, target_seconds: Optional[float],
                       seed: int = 42) -> Dict[str, Any]:
        t0 = time.time()
        ckpt = self.weights_path / self.checkpoint
        for req, hint in [(self.repo_path / "packages", "repo clone missing"), (ckpt, "checkpoint missing"),
                          (self.gemma_root, "gemma text encoder missing")]:
            if not req.exists():
                raise FileNotFoundError(f"{req} not found ({hint}) — run setup/install_model.sh --model ltx-2.3-dev-v2v")
        info = _probe(video_path)
        sw, sh = _stage1_dims(info["width"], info["height"], self.long_side)
        ow, oh = sw, sh  # --height/--width are the final output dims of the two-stage pipeline
        tgt_s = target_seconds if target_seconds else info["duration"]
        ext_frames = int(math.ceil(tgt_s * FPS))
        total = PREFIX_FRAMES + ext_frames
        total = min(self.max_frames, ((total - 1 + 7) // 8) * 8 + 1)  # 8k+1, >= prefix+extension when under the cap
        workdir = Path(tempfile.mkdtemp(prefix="ltx23ext_"))
        try:
            # last 9 frames of the 24 fps, output-resolution version of the input
            frames_dir = workdir / "prefix"
            frames_dir.mkdir()
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
                            "-vf", f"fps={FPS},scale={ow}:{oh}:flags=area", "-f", "image2",
                            str(workdir / "all_%05d.png")], capture_output=True, check=True)
            allf = sorted(workdir.glob("all_*.png"))
            if not allf:
                raise RuntimeError("input decoded to zero frames")
            tail = allf[-PREFIX_FRAMES:]
            while len(tail) < PREFIX_FRAMES:  # very short clips: repeat the first frame
                tail = [tail[0]] + tail
            image_args = []
            for i, p in enumerate(tail):
                dst = frames_dir / f"p{i}.png"
                shutil.copyfile(p, dst)
                image_args += ["--image", str(dst), str(i), "1.0"]
            raw = workdir / "raw.mp4"
            python = str(_kit_root() / "envs" / "ltx-2.3-dev-v2v" / "bin" / "python")
            if not Path(python).exists():
                python = "python3"
            cmd = [python, "-m", "ltx_pipelines.distilled",
                   "--distilled-checkpoint-path", str(ckpt),
                   "--spatial-upsampler-path", str(self.weights_path / self.upsampler),
                   "--gemma-root", str(self.gemma_root),
                   "--prompt", text_prompt, "--seed", str(seed),
                   "--height", str(sh), "--width", str(sw), "--num-frames", str(total),
                   "--frame-rate", str(FPS), "--output-path", str(raw)] + image_args
            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            proc = subprocess.run(cmd, cwd=str(self.repo_path), env=env, capture_output=True, text=True, timeout=7200)
            if proc.returncode != 0:
                raise RuntimeError(f"ltx_pipelines.distilled failed (exit {proc.returncode}): {proc.stderr[-1200:]}")
            if not raw.exists():
                raise RuntimeError(f"pipeline exited 0 but produced no video: {proc.stdout[-400:]}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-vf", f"select='gte(n,{PREFIX_FRAMES})'",
                            "-vsync", "0", "-r", str(FPS), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-an", str(output_path)], capture_output=True, check=True)
            out = _probe(output_path)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        return {
            "video_path": str(output_path), "duration_seconds": time.time() - t0, "status": "success",
            "metadata": {
                "checkpoint_variant": self.checkpoint, "pipeline": "ltx_pipelines.distilled + 9 tail-frame keyframes (local extend)",
                "prefix_frames_dropped": PREFIX_FRAMES, "generated_frames": total, "requested": f"{ow}x{oh}",
                "target_seconds": round(tgt_s, 3), "capped": total < PREFIX_FRAMES + ext_frames,
                "output_geometry": f"{out['width']}x{out['height']}@{out['fps']:.3f}fps/{out['frames']}f",
                "seed": seed, "audio_stripped": True,
            },
        }


class Ltx23ExtendWrapper(ModelWrapper):
    def __init__(self, model: str = "Lightricks/LTX-2.3:extend-local", output_dir: str = "./outputs", **kwargs):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = Ltx23ExtendService()

    def generate(self, image_path, text_prompt, duration: float = 5.0, output_filename: Optional[str] = None,
                 video_path: Optional[Union[str, Path]] = None, **kwargs) -> Dict[str, Any]:
        start = time.time()
        if video_path is None:
            return {"success": False, "video_path": None, "error": "ltx-2.3-extend-local needs video_path",
                    "duration_seconds": 0.0, "generation_id": None, "model": self.model, "status": "failed",
                    "metadata": {"prompt": text_prompt}}
        qd = kwargs.get("question_data") or {}
        gt = qd.get("ground_truth_video")
        target_seconds = _probe(gt)["duration"] if gt and Path(gt).exists() else None
        output_path = self.output_dir / (output_filename or "video.mp4")
        try:
            r = self.service.generate_video(video_path, text_prompt, output_path, target_seconds,
                                            seed=int(kwargs.get("seed", 42) or 42))
        except Exception as e:  # noqa: BLE001
            return {"success": False, "video_path": None, "error": str(e)[:1500], "duration_seconds": time.time() - start,
                    "generation_id": None, "model": self.model, "status": "failed",
                    "metadata": {"prompt": text_prompt, "modality": "v2v", "video_path_input": str(video_path)}}
        return {"success": True, "video_path": r["video_path"], "error": None, "duration_seconds": r["duration_seconds"],
                "generation_id": f"ltx23ext_{int(start)}", "model": self.model, "status": "success",
                "metadata": {"prompt": text_prompt, "modality": "v2v", **r["metadata"]}}
