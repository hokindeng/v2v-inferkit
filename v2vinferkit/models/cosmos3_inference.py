"""NVIDIA Cosmos3-Super video transfer inference (open-source, multi-GPU).

True spec-driven video2video: the wrapper writes a per-task spec JSON (edge
control derived on the fly from the conditioning video via vision_path) and
shells out to torchrun in the NVIDIA/cosmos-framework repo clone.

Hardware: Super (64B MoT, 32B diffusion tower, bf16) needs 128GB aggregate
GPU memory — 4+ GPUs officially ("does not fit on a single 80GB H100").
4xA100-80GB is arch-supported but unbenchmarked upstream. Does NOT run on a
single 48GB GPU. Checkpoint is 132.7GB, auto-downloaded to $HF_HOME on first
run (gated: accept the nvidia/Cosmos3-Super license and set HF_TOKEN).

Design notes:
- --no-guardrails: the guardrail stack needs extra gated downloads and its
  RetinaFace face-blur post-processor MODIFIES output pixels — unacceptable
  for benchmark outputs.
- edge control (preset medium) is the primary hint: strongest per-frame
  structure/motion lock, which is what a physics/object-permanence benchmark
  needs. Switch to edge+blur multi-control if object identity/color drifts.
- the framework loads the model once for multiple spec files; when running a
  full benchmark, prefer one launch over many (see generate_batch).

Runs inside its model venv via _subprocess_worker.py; the actual inference
environment is the cosmos-framework repo's own uv venv (created by setup.sh).
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .base import ModelWrapper


def _kit_root() -> Path:
    return Path(__file__).parent.parent.parent


def _repo_dir() -> Path:
    return _kit_root() / "repos" / "cosmos-framework"


class Cosmos3Service:
    """Writes per-task specs and runs cosmos_framework.scripts.inference."""

    def __init__(self, model: str = "Cosmos3-Super"):
        self.checkpoint = model  # registry name resolved by the framework
        self.repo_path = Path(os.environ.get("COSMOS_REPO_PATH", str(_repo_dir())))
        self.num_gpus = int(os.environ.get("COSMOS_NUM_GPUS", "4"))
        self.master_port = os.environ.get("COSMOS_MASTER_PORT", "29500")

    def _write_spec(
        self,
        workdir: Path,
        name: str,
        video_path: Union[str, Path],
        prompt: str,
        num_frames: int,
        fps: int,
    ) -> Path:
        prompt_file = workdir / f"{name}_prompt.txt"
        prompt_file.write_text(prompt)
        spec = {
            "name": name,
            "model_mode": "video2video",
            "resolution": "720",
            "aspect_ratio": "16,9",
            "num_frames": num_frames,
            "fps": fps,
            "num_steps": 50,
            "guidance": 3.0,
            "control_guidance": 1.5,
            "num_video_frames_per_chunk": num_frames,
            "num_conditional_frames": 1,
            "num_first_chunk_conditional_frames": 0,
            "share_vision_temporal_positions": True,
            # spec-relative path resolution — always emit absolute paths
            "prompt_path": str(prompt_file.resolve()),
            "vision_path": str(Path(video_path).resolve()),
            "edge": {"preset_edge_threshold": "medium"},
        }
        spec_file = workdir / f"{name}.json"
        spec_file.write_text(json.dumps(spec, indent=2))
        return spec_file

    def _probe_fps_frames(self, video_path: Union[str, Path]):
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                "-show_entries", "stream=r_frame_rate,nb_read_packets", "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, check=True,
        )
        s = json.loads(out.stdout)["streams"][0]
        num, den = (int(x) for x in s["r_frame_rate"].split("/"))
        fps = max(1, round(num / max(den, 1)))
        return fps, int(s["nb_read_packets"])

    def generate_video(
        self,
        video_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        seed: int = 2026,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        venv_torchrun = self.repo_path / ".venv" / "bin" / "torchrun"
        if not venv_torchrun.exists():
            raise FileNotFoundError(
                f"cosmos-framework venv not found at {venv_torchrun} — run "
                "setup/install_model.sh --model cosmos3-super-v2v"
            )

        fps, nb_frames = self._probe_fps_frames(video_path)
        # 720p caps at 200 frames; geometry/frames follow the control video
        num_frames = min(nb_frames, 121)

        workdir = Path(tempfile.mkdtemp(prefix="cosmos3_"))
        try:
            name = "task"
            self._write_spec(workdir, name, video_path, text_prompt, num_frames, fps)
            outdir = workdir / "out"

            gpus = ",".join(str(i) for i in range(self.num_gpus))
            cmd = [
                str(venv_torchrun),
                f"--nproc-per-node={self.num_gpus}",
                "--master-addr=127.0.0.1", f"--master-port={self.master_port}",
                "-m", "cosmos_framework.scripts.inference",
                "--parallelism-preset=latency",
                "-i", str(workdir / f"{name}.json"),
                "-o", str(outdir),
                "--checkpoint-path", self.checkpoint,
                "--no-guardrails",
                "--seed", str(seed),
            ]

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", gpus)
            env["LD_LIBRARY_PATH"] = ""  # bundled-libtorch conflict guard
            env.setdefault("PYTHONNOUSERSITE", "1")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("NCCL_NET_PLUGIN", "none")  # EFA/Libfabric segfault guard on AWS

            proc = subprocess.run(
                cmd, cwd=str(self.repo_path), env=env,
                capture_output=True, text=True, timeout=14400,
            )
            generated = outdir / name / "vision.mp4"
            if proc.returncode != 0:
                raise RuntimeError(f"cosmos inference failed (exit {proc.returncode}): {proc.stderr[-800:]}")
            if not generated.exists():
                raise RuntimeError(f"cosmos exited 0 but produced no video. stdout tail: {proc.stdout[-400:]}")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated), str(output_path))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        return {
            "video_path": str(output_path),
            "duration_seconds": time.time() - start_time,
            "status": "success",
            "metadata": {
                "control": "edge (preset medium)",
                "num_frames": num_frames,
                "fps": fps,
                "num_gpus": self.num_gpus,
                "seed": seed,
                "guardrails": False,
            },
        }


class Cosmos3Wrapper(ModelWrapper):
    """v2v-inferkit wrapper for Cosmos3-Super video transfer (edge control)."""

    def __init__(
        self,
        model: str = "Cosmos3-Super",
        output_dir: str = "./outputs",
        **kwargs,
    ):
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.service = Cosmos3Service(model=model)

    def generate(
        self,
        image_path: Optional[Union[str, Path]],
        text_prompt: str,
        duration: float = 4.0,
        output_filename: Optional[str] = None,
        video_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        start_time = time.time()

        if video_path is None:
            return {
                "success": False,
                "video_path": None,
                "error": "cosmos3-super-v2v is video-to-video only: video_path is required",
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
            "generation_id": f"cosmos3_{int(start_time)}",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "modality": "v2v",
                "video_path_input": str(video_path),
                **result["metadata"],
            },
        }
