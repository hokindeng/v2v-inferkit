"""HY-OmniWeaving video-to-video editing inference (open-source, local GPU).

Shells out to `torchrun --nproc_per_node=1 generate.py --task editing` in the
Tencent-Hunyuan/OmniWeaving repo clone: generate.py initializes parallel state
from WORLD_SIZE/LOCAL_RANK at module import time, so torchrun is the supported
launch path even on a single GPU (8 GPUs in the README is just sequence
parallelism for speed; maintainer-stated minimum is HunyuanVideo-1.5's 14GB
with offloading).

Constraints handled here:
- condition video must have >= 33 frames (hard ValueError below) — short
  inputs are ffmpeg-padded by slowing them down;
- 480p is the only supported resolution; --aspect_ratio is matched to the
  source video via ffprobe;
- --overlap_group_offloading false (the default true balloons CPU RAM, and
  the repo's own generate.sh disables it).

Model dir layout: the tencent/HY-OmniWeaving snapshot PLUS four extra
components downloaded into it (Qwen2.5-VL-7B-Instruct, byt5-small,
Glyph-SDXL-v2, FLUX.1-Redux siglip — the last is gated on HF). setup.sh
handles this; see setup/models/hy-omniweaving-v2v/.

Runs inside its model venv (envs/hy-omniweaving-v2v) via _subprocess_worker.py.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .base import ModelWrapper

_MIN_FRAMES = 33
# aspect ratios generate.py buckets against (16:9 default)
_ASPECT_CHOICES = ["16:9", "9:16", "4:3", "3:4", "1:1"]


def _kit_root() -> Path:
    return Path(__file__).parent.parent.parent


def _default_model_path() -> Path:
    weights = os.environ.get("V2V_WEIGHTS_DIR", str(_kit_root() / "weights"))
    return Path(weights) / "HY-OmniWeaving"


def _default_repo_path() -> Path:
    return _kit_root() / "repos" / "OmniWeaving"


def _probe_video(video_path: Union[str, Path]) -> Tuple[int, int, int]:
    """Return (width, height, nb_frames) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets",
        "-of", "json", str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stream = json.loads(out.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_read_packets"])


def _nearest_aspect(width: int, height: int) -> str:
    ratio = width / height
    best = min(
        _ASPECT_CHOICES,
        key=lambda a: abs(ratio - Fraction(*(int(x) for x in a.split(":")))),
    )
    return best


def _pad_to_min_frames(video_path: str, nb_frames: int, workdir: Path) -> str:
    """Slow a too-short video down so it reaches _MIN_FRAMES frames."""
    factor = _MIN_FRAMES / max(nb_frames, 1) + 0.05
    padded = workdir / "padded_input.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-filter:v", f"setpts=PTS*{factor:.4f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
            str(padded),
        ],
        capture_output=True, check=True,
    )
    return str(padded)


class OmniWeavingService:
    """Builds and runs the torchrun editing command."""

    def __init__(self, model: str = "tencent/HY-OmniWeaving"):
        self.model_id = model
        self.model_path = Path(os.environ.get("OMNIWEAVING_MODEL_PATH", str(_default_model_path())))
        self.repo_path = Path(os.environ.get("OMNIWEAVING_REPO_PATH", str(_default_repo_path())))

    def generate_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        seed: int = 0,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if not (self.repo_path / "generate.py").exists():
            raise FileNotFoundError(
                f"OmniWeaving repo not found at {self.repo_path} — run "
                "setup/install_model.sh --model hy-omniweaving-v2v"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {self.model_path}")

        width, height, nb_frames = _probe_video(video_path)
        aspect = _nearest_aspect(width, height)

        workdir = Path(tempfile.mkdtemp(prefix="omniweaving_"))
        try:
            input_video = str(video_path)
            if nb_frames < _MIN_FRAMES:
                input_video = _pad_to_min_frames(input_video, nb_frames, workdir)

            venv_torchrun = _kit_root() / "envs" / "hy-omniweaving-v2v" / "bin" / "torchrun"
            torchrun = str(venv_torchrun) if venv_torchrun.exists() else (shutil.which("torchrun") or "torchrun")
            # --standalone picks a free rendezvous port, so parallel workers
            # on one host don't collide on torchrun's default 29500
            cmd = [
                torchrun, "--standalone", "--nproc_per_node=1", "generate.py",
                "--task", "editing",
                "--prompt", text_prompt,
                "--condition_video_paths", input_video,
                "--aspect_ratio", aspect,
                "--seed", str(seed),
                "--sparse_attn", "false",
                "--use_sageattn", "false",
                "--enable_cache", "false",
                # group offloading calls diffusers' enable_group_offload, which
                # doesn't exist in the repo's own pinned diffusers==0.32.2 —
                # plain offloading is sufficient on a 48GB GPU (bf16 DiT ~17GB)
                "--group_offloading", "false",
                "--overlap_group_offloading", "false",
                "--output_path", str(output_path),
                "--model_path", str(self.model_path),
            ]
            if num_inference_steps is not None:
                cmd += ["--num_inference_steps", str(num_inference_steps)]

            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                cmd, cwd=str(self.repo_path), env=env,
                capture_output=True, text=True, timeout=7200,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"generate.py failed (exit {proc.returncode}): {proc.stderr[-800:]}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    f"generate.py exited 0 but produced no video at {output_path}. "
                    f"stdout tail: {proc.stdout[-400:]}"
                )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return {
            "video_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "status": "success",
            "metadata": {
                "aspect_ratio": aspect,
                "input_frames": nb_frames,
                "padded": nb_frames < _MIN_FRAMES,
                "seed": seed,
                "resolution": "480p",
            },
        }


class OmniWeavingWrapper(ModelWrapper):
    """v2v-inferkit wrapper for HY-OmniWeaving --task editing."""

    def __init__(
        self,
        model: str = "tencent/HY-OmniWeaving",
        output_dir: str = "./outputs",
        **kwargs,
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = OmniWeavingService(model=model)

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
                "error": "hy-omniweaving-v2v is video-to-video only: video_path is required",
                "duration_seconds": 0.0,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt},
            }

        kwargs.pop("question_data", None)
        kwargs.pop("num_frames", None)  # video_length is fixed by the task; input video governs
        output_path = self.output_dir / (output_filename or "video.mp4")

        try:
            result = self.service.generate_video(
                video_path=video_path,
                text_prompt=text_prompt,
                output_path=output_path,
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
            "generation_id": f"omniweaving_{int(start_time)}",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v",
                "video_path_input": str(video_path),
                **result["metadata"],
            },
        }
