"""LTX-2.3 video-to-video inference via IC-LoRA (open-source, local GPU).

⚠️ CHECKPOINT CAVEAT: LTX-2.3's only officially documented v2v pipeline
(ICLoraPipeline — video conditioning + IC-LoRA restyling) requires the
DISTILLED checkpoint ("ICLoraPipeline can only be used with a distilled
model", docs/pipelines.md); running the dev checkpoint through its
distilled-sigma/no-CFG denoiser produces garbage (issue #250). This collides
with the team's flagship-only rule. Pending a team ruling, this wrapper runs
the official distilled IC-LoRA path and records the checkpoint variant in the
result metadata. A dev-checkpoint SDEdit-style pipeline can be built from
official primitives (~100 lines) if the team wants strict-dev v2v.

Shells out to `python -m ltx_pipelines.ic_lora` in the Lightricks/LTX-2 repo
clone. Audio note: LTX-2.3 co-generates audio and the CLI muxes it into the
mp4; the benchmark compares video only, so audio is stripped afterwards.

Input constraints handled here: frame count snapped to 8k+1, H/W divisible by
64 (two-stage pipeline) via ffmpeg pre-scale.

Runs inside its model venv (envs/ltx-2.3-dev-v2v) via _subprocess_worker.py.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .base import ModelWrapper


def _kit_root() -> Path:
    return Path(__file__).parent.parent.parent


def _weights_dir() -> Path:
    return Path(os.environ.get("V2V_WEIGHTS_DIR", str(_kit_root() / "weights"))) / "LTX-2.3"


def _repo_dir() -> Path:
    return _kit_root() / "repos" / "LTX-2"


def _probe(video_path: Union[str, Path]) -> Tuple[int, int, int]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets", "-of", "json",
        str(video_path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_read_packets"])


class Ltx23Service:
    """Prepares input and runs ltx_pipelines.ic_lora."""

    def __init__(self, model: str = "Lightricks/LTX-2.3:ic-lora"):
        self.model_id = model
        self.repo_path = Path(os.environ.get("LTX2_REPO_PATH", str(_repo_dir())))
        self.weights_path = Path(os.environ.get("LTX2_WEIGHTS_PATH", str(_weights_dir())))
        self.gemma_root = Path(os.environ.get(
            "LTX2_GEMMA_ROOT", str(self.weights_path / "gemma-3-12b")
        ))
        self.checkpoint = os.environ.get(
            "LTX2_IC_LORA_CHECKPOINT", "ltx-2.3-22b-distilled-1.1.safetensors"
        )
        self.upsampler = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

    def _prepare_input(self, video_path: str, workdir: Path) -> Tuple[str, int, int, int]:
        """Rescale to 64-divisible dims and 8k+1 frames."""
        w, h, n = _probe(video_path)
        tw = max(64, (w // 64) * 64)
        th = max(64, (h // 64) * 64)
        tn = max(9, ((n - 1) // 8) * 8 + 1)

        if (tw, th, tn) == (w, h, n):
            return video_path, w, h, n

        prepared = workdir / "prepared_input.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", f"scale={tw}:{th},select='lt(n,{tn})'",
                "-vsync", "0", "-c:v", "libx264", "-preset", "fast",
                "-crf", "18", "-an", str(prepared),
            ],
            capture_output=True, check=True,
        )
        return str(prepared), tw, th, tn

    def generate_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        conditioning_strength: float = 0.8,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        checkpoint_path = self.weights_path / self.checkpoint
        for req, hint in [
            (self.repo_path / "packages", "repo clone missing"),
            (checkpoint_path, "checkpoint missing"),
            (self.gemma_root, "gemma text encoder missing (gated — accept HF terms)"),
        ]:
            if not req.exists():
                raise FileNotFoundError(
                    f"{req} not found ({hint}) — run setup/install_model.sh --model ltx-2.3-dev-v2v"
                )

        workdir = Path(tempfile.mkdtemp(prefix="ltx23_"))
        try:
            prepared, w, h, n = self._prepare_input(str(video_path), workdir)
            raw_output = workdir / "raw_output.mp4"

            python = str(_kit_root() / "envs" / "ltx-2.3-dev-v2v" / "bin" / "python")
            if not Path(python).exists():
                python = "python3"

            cmd = [
                python, "-m", "ltx_pipelines.ic_lora",
                "--distilled-checkpoint-path", str(checkpoint_path),
                "--spatial-upsampler-path", str(self.weights_path / self.upsampler),
                "--gemma-root", str(self.gemma_root),
                "--video-conditioning", prepared, str(conditioning_strength),
                "--prompt", text_prompt,
                "--seed", str(seed),
                "--output-path", str(raw_output),
            ]

            env = dict(os.environ)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            proc = subprocess.run(
                cmd, cwd=str(self.repo_path), env=env,
                capture_output=True, text=True, timeout=7200,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ic_lora failed (exit {proc.returncode}): {proc.stderr[-800:]}")
            if not raw_output.exists():
                raise RuntimeError(f"ic_lora exited 0 but produced no video. stdout tail: {proc.stdout[-400:]}")

            # Strip the co-generated audio track — benchmark compares video only
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw_output), "-c:v", "copy", "-an", str(output_path)],
                capture_output=True, check=True,
            )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return {
            "video_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "status": "success",
            "metadata": {
                "checkpoint_variant": self.checkpoint,
                "pipeline": "ic_lora (official v2v; distilled-only — pending team ruling)",
                "conditioning_strength": conditioning_strength,
                "prepared_dims": f"{w}x{h}x{n}",
                "seed": seed,
                "audio_stripped": True,
            },
        }


class Ltx23Wrapper(ModelWrapper):
    """v2v-inferkit wrapper for LTX-2.3 IC-LoRA video conditioning."""

    def __init__(
        self,
        model: str = "Lightricks/LTX-2.3:ic-lora",
        output_dir: str = "./outputs",
        **kwargs,
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = Ltx23Service(model=model)

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
                "error": "ltx-2.3-dev-v2v is video-to-video only: video_path is required",
                "duration_seconds": 0.0,
                "generation_id": None,
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt},
            }

        kwargs.pop("question_data", None)
        kwargs.pop("num_frames", None)
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
            "generation_id": f"ltx23_{int(start_time)}",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v",
                "video_path_input": str(video_path),
                **result["metadata"],
            },
        }
